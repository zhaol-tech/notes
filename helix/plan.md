# Helix Parallelism Implementation Plan for SGLang (Hop-A)

## Table of Contents

1. [Paper Summary](#1-paper-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [Current SGLang Logic & Gaps](#3-current-sglang-logic--gaps)
4. [Implementation Changes](#4-implementation-changes)
   - 4.1 [Configuration & Server Args](#41-configuration--server-args)
   - 4.2 [Process Groups](#42-process-groups)
   - 4.3 [Weight Sharding & Loading](#43-weight-sharding--loading)
   - 4.4 [KV Cache Sharding](#44-kv-cache-sharding)
   - 4.5 [Attention Phase](#45-attention-phase)
   - 4.6 [FFN Phase](#46-ffn-phase)
   - 4.7 [Model Layer Changes](#47-model-layer-changes)
   - 4.8 [CUDA Graphs](#48-cuda-graphs)
5. [Validation Plan](#5-validation-plan)

---

## 1. Paper Summary

**Paper:** Helix Parallelism: Rethinking Sharding Strategies for Interactive Multi-Million-Token LLM Decoding (arXiv:2507.07120v1)

**Core problem:** Standard TP shards both attention and FFN the same way. When TP width exceeds the number of KV heads K, KV cache must be duplicated across GPUs, creating a ceiling on attention speedup. Meanwhile, FFN weight reads (the other bottleneck) can only be parallelized across those same K GPUs.

**Core idea:** Decouple attention and FFN parallelism within each transformer layer:
- **Attention phase:** Use KVP (KV Parallelism) to shard KV cache along the sequence dimension across KVP GPUs, combined with TP across KV heads (TPA <= K). Total GPUs: N = KVP x TPA. No KV duplication.
- **FFN phase:** Reuse the same N GPUs for TP (dense models, TPF = N) or TP x EP (MoE models). All N GPUs collaborate to shard FFN weights.

**Key mechanism:** After local attention on each KVP shard, a single All-to-All over the query-head axis exchanges partial attention outputs + log-sum-exp scalars. Each GPU rescales and sums to reconstruct exact softmax-normalized attention. This transitions the data layout from KVP x TPA → TP=N for the subsequent o_proj and FFN computation.

**Hop-A** (this plan): The baseline Helix approach with exposed (non-overlapped) All-to-All communication after attention.

---

## 2. Architecture Overview

### 2.1 GPU Layout

```
N = KVP × TPA GPUs arranged in a 2D grid:

                  TPA=0    TPA=1    ...   TPA=(TPA-1)
               ┌─────────┬─────────┬─────┬─────────────┐
    KVP=0      │ GPU 0   │ GPU 1   │ ... │ GPU TPA-1   │  ← KV shard 0 (tokens 0..S/KVP-1)
               ├─────────┼─────────┼─────┼─────────────┤
    KVP=1      │ GPU TPA │ GPU     │ ... │ GPU 2*TPA-1 │  ← KV shard 1
               │         │ TPA+1   │     │             │
               ├─────────┼─────────┼─────┼─────────────┤
    ...        │   ...   │   ...   │ ... │   ...       │
               ├─────────┼─────────┼─────┼─────────────┤
    KVP=KVP-1  │ GPU     │ ...     │ ... │ GPU N-1     │  ← KV shard KVP-1
               └─────────┴─────────┴─────┴─────────────┘

For FFN: all N GPUs form a single TP=N group (dense) or TPF × EP grid (MoE)
```

### 2.2 Per-Layer Data Flow (Decode)

```
Layer Input: [B, H] replicated on all N GPUs
│
├─ 1. LayerNorm (local, all GPUs compute same result)
│
├─ 2. QKV Projection (redundant across KVP, sharded by TPA)
│     Each GPU: [B, H] × W_qkv → Q:[B, Q/TPA, Hsz], K:[B, K/TPA, Hsz], V:[B, K/TPA, Hsz]
│     GPUs with same TPA rank compute identical Q,K,V
│     Only the designated KVP rank stores new K,V in cache
│
├─ 3. Local Attention (each GPU: Q vs local KV shard)
│     GPU(kvp=i, tpa=j): Q_j × KVcache_i → partial_out:[B, Q/TPA, Hsz] + lse:[B, Q/TPA]
│
├─ 4. All-to-All across KVP dimension (within each TPA group)
│     Split Q/TPA heads into KVP chunks → exchange → receive KVP partial results for Q/N heads
│
├─ 5. Rescale & Sum using LSE (flash-decoding merge)
│     Each GPU: KVP partial results → exact attention for Q/N heads → [B, H/N]
│
├─ 6. o_proj (RowParallel, TP=N)
│     Each GPU: [B, H/N] × W_o → partial [B, H], All-Reduce over N GPUs → [B, H]
│
├─ 7. Residual + LayerNorm (local)
│
├─ 8. FFN gate_up_proj (ColumnParallel, TP=N)
│     Each GPU: [B, H] × W_gate_up → [B, 2F/N]
│
├─ 9. SiluAndMul → [B, F/N]
│
├─ 10. FFN down_proj (RowParallel, TP=N)
│      Each GPU: [B, F/N] × W_down → partial [B, H], All-Reduce over N GPUs → [B, H]
│
├─ 11. Residual (local)
│
└─ Layer Output: [B, H] replicated on all N GPUs
```

### 2.3 Weight Layout

| Layer | Sharding | TP Width | Weight per GPU | Replicated across KVP? |
|-------|----------|----------|---------------|----------------------|
| `qkv_proj` | Column-parallel | TPA | `H × (Q/TPA + 2·K/TPA)·Hsz` | **Yes** (redundant QKV) |
| `o_proj` | Row-parallel | N | `(H/N) × H` | No |
| `gate_up_proj` | Column-parallel | N | `H × (2F/N)` | No |
| `down_proj` | Row-parallel | N | `(F/N) × H` | No |
| `embed_tokens` | Vocab-parallel | N | `V/N × H` | No |
| `lm_head` | Column-parallel | N | `H × V/N` | No |

QKV weights are replicated across KVP ranks to avoid an All-Gather of queries before attention. This is a deliberate trade-off: extra weight memory vs. saved communication. For long-context decode, the KV cache memory savings from KVP far outweigh the extra QKV weight copies.

---

## 3. Current SGLang Logic & Gaps

This section maps each Helix requirement to the current SGLang implementation, identifying what exists and what's missing.

### 3.1 Parallelism Configuration

**Current logic:** SGLang uses a single TP width for the entire model. All linear layers (QKV, o_proj, FFN) share the same `tp_size` and `tp_rank` derived from `get_tensor_model_parallel_world_size()` / `get_tensor_model_parallel_rank()`. The one exception is dp_attention, which splits the global TP into an attention TP group and a DP dimension — but even there, all linear layers within a given phase use a single TP width.

**Relevant files:**
- `server_args.py` — `tp_size`, `dp_size`, `enable_dp_attention`
- `parallel_state.py` — `initialize_model_parallel()`, `_TP` global group
- `model_runner.py:810` — calls `initialize_model_parallel(tensor_model_parallel_size=self.tp_size, ...)`

**Gap:** Helix requires **two different TP widths within a single layer**: TPA for QKV projection and N (= KVP × TPA) for o_proj and FFN. There is no mechanism to configure per-layer or per-sublayer TP widths. No `helix_kvp_size` or `helix_tpa_size` parameters exist.

### 4.2 Process Groups

**Current logic:** `parallel_state.py` creates several process groups via `initialize_model_parallel()`:
- `_TP` — main tensor parallel group (all TP ranks)
- `_ATTN_TP` — attention-specific TP group (smaller when dp_attention enabled, otherwise same as `_TP`)
- `_ATTN_CP` — attention context parallel group (for context parallelism)
- `_MOE_EP`, `_MOE_TP` — expert parallel and MoE-specific TP groups
- `_PP` — pipeline parallel group

Accessor functions: `get_tp_group()`, `get_attn_tp_group()`, `get_attn_cp_group()`, etc.

**Relevant files:**
- `parallel_state.py:1595-1771` — group creation
- `parallel_state.py:175-299` — `GroupCoordinator` wrapper class

**Gap:** No KVP-specific process group exists. Helix requires:
- **KVP group** — GPUs with the same TPA rank but different KVP ranks (for All-to-All after attention). No existing group matches this topology.
- **TPA group** — GPUs with the same KVP rank but different TPA ranks (for potential intra-attention TP). Could reuse `_ATTN_TP` if configured correctly, but the rank layout math differs from dp_attention's layout.
- The existing `_TP` group (all N GPUs) can serve as the TPF group for FFN.

### 4.3 Weight Sharding & Loading

**Current logic:** Linear layers accept optional `tp_rank` and `tp_size` in their constructors (`linear.py:277-329` for ColumnParallel, `linear.py:1306-1321` for RowParallel). If not provided, they default to the global TP state. Weight loaders compute `start_idx = tp_rank * shard_size` to extract the correct shard from checkpoint weights.

`QKVParallelLinear` (`linear.py:840-1277`) handles the special case of Q, K, V having different head counts (GQA/MQA). It also accepts explicit `tp_rank`/`tp_size`.

**Relevant files:**
- `linear.py:370-417` — `ColumnParallelLinear.weight_loader()` uses `tp_rank` for shard selection
- `linear.py:1367-1428` — `RowParallelLinear.weight_loader()` uses `tp_rank` for shard selection

**Gap:** The infrastructure for per-layer TP rank/size already exists in the constructors, but **no model currently passes different values to different layers**. For Helix, `qkv_proj` must be constructed with `tp_size=TPA, tp_rank=tpa_rank` while `o_proj` uses `tp_size=N, tp_rank=global_rank`. The model's `__init__` must be modified to pass these explicitly.

Additionally, `RowParallelLinear.forward()` calls `tensor_model_parallel_all_reduce()` which always uses the global `_TP` group (`parallel_state.py`). For o_proj with TP=N, this is correct (global `_TP` is already N GPUs). But if any layer needed a different All-Reduce group, the current code would need a `tp_group` parameter — similar to how `use_dp_attention_reduce` switches to `get_attention_tp_group().all_reduce()`.

### 3.4 KV Cache Management

**Current logic:** Each GPU stores the **full** KV cache for all requests it handles:
- `MHATokenToKVPool` (`memory_pool.py:697-1038`) — allocates `k_buffer` and `v_buffer` of shape `[pool_size, head_num, head_dim]` per layer
- `ReqToTokenPool` (`memory_pool.py:126-184`) — maps `[max_req, max_context_len]` for token locations
- `prepare_for_decode()` (`schedule_batch.py:1948`) — allocates one KV slot per request via `alloc_for_decode(token_per_req=1)`, always on the local GPU
- `set_kv_buffer()` — stores new K,V at `out_cache_loc` on every decode step, on every GPU
- `init_forward_metadata()` — builds `kv_indptr`/`kv_indices` from `seq_lens` (full sequence length)

**Relevant files:**
- `memory_pool.py` — pool allocation and KV storage
- `schedule_batch.py:1948-2031` — decode preparation and KV allocation
- `forward_batch_info.py:376-526` — ForwardBatch construction

**Gap:** There is **no concept of KV sharding across GPUs**. Every GPU stores the complete KV history for its requests. Helix requires:
- Each KVP rank stores only 1/KVP of the KV history (sequence-dimension sharding)
- Round-robin assignment of new tokens to KVP ranks during decode
- `save_kv_cache` selectivity per-rank via a write mask (only designated rank stores new K,V)
- `alloc_for_decode()` must be conditional (only designated rank allocates)
- `seq_lens` passed to the attention backend must reflect **local** KV length, not global
- Prefill must distribute KV across KVP ranks by token position
- `ForwardBatch` needs new fields: `helix_local_kv_lens`, `helix_kv_write_mask`

**Head-count / topology gap:** The KV pool is sized with `head_num = model_config.get_num_kv_heads(get_attention_tp_size())` (see `model_runner_kv_cache_mixin.py:650`). `get_attention_tp_size()` returns the full attention TP width. For Helix, the KV pool's head count must be divided by **TPA** (not N), since each TPA rank stores its subset of KV heads. Similarly, the attention backends (`triton_backend.py:89`, `flashinfer_backend.py:915`) derive `num_head` and `num_kv_head` from `get_attention_tp_size()` — these must use TPA when Helix is enabled. Failure to fix this will cause shape/layout mismatches and incorrect cache sizing.

**Radix/prefix cache gap:** The radix cache (`radix_cache.py:460-500`) reads `req_to_token` indexed by **global** token lengths:
```python
# radix_cache.py:461-462
kv_indices = self.req_to_token_pool.req_to_token[req.req_pool_idx, : len(token_ids)]
```
where `token_ids` has global length. With compact-local `req_to_token`, reading up to `len(token_ids)` (global count) would read past `local_kv_len` into uninitialized/zero entries, causing incorrect cache insertions and pool corruption. Similarly, `cache_unfinished_req()` (`radix_cache.py:499-500`) uses the same global-length indexing pattern. See section 4.4g for the mitigation.

### 4.5 Attention Phase

**Current logic:** `RadixAttention.forward()` (`radix_attention.py:99-135`) delegates to `attn_backend.forward()`, which dispatches to `forward_decode()` for decode mode. The attention backend:
1. Stores new K,V in the KV cache
2. Runs the attention kernel (e.g., Triton two-stage: stage 1 computes partials per KV split, stage 2 merges)
3. Returns the **fully merged** attention output

The Triton backend's two-stage decode kernel (`decode_attention.py`) already separates:
- Stage 1: Partial attention computation per KV split → `attn_logits` + `attn_lse`
- Stage 2: Merge across splits → final output

`RadixAttention` stores `tp_q_head_num`, `tp_k_head_num` based on the global TP head assignment.

**Relevant files:**
- `radix_attention.py:47-135` — attention layer
- `base_attn_backend.py:79-121` — dispatch logic
- `triton_backend.py:997-1047` — `forward_decode()` runs both stages
- `decode_attention.py:45-249` (stage 1), `515-630` (stage 2)

**Gap:** Four missing capabilities:
1. **No partial output + LSE return path.** `forward_decode()` always runs both stages and returns only the merged output. Helix needs the partially-merged result (after intra-GPU stage 2, but before cross-GPU merge) plus the LSE. Stage 2 currently does **not** output its merged LSE.
2. **No All-to-All communication step.** There is no cross-GPU exchange of partial attention results. This is entirely new functionality.
3. **No rescale-and-sum merge across GPUs.** The flash-decoding merge formula exists in the stage 2 kernel but only operates on intra-GPU splits. A cross-GPU version is needed.
4. **Head counts are tied to global TP.** `RadixAttention` computes `tp_q_head_num = total_heads // tp_size`. For Helix, q heads per GPU should be `total_heads // TPA` during local attention, then `total_heads // N` after the All-to-All. The layer needs to be aware of both counts.

### 4.6 FFN Phase

**Current logic:** `LlamaMLP` (`llama.py:65-116`) uses:
- `MergedColumnParallelLinear` for `gate_up_proj` — shards output dim by global TP
- `RowParallelLinear` for `down_proj` — shards input dim by global TP, All-Reduces via global `_TP` group

The FFN path has no forward-mode branching and no communication beyond the All-Reduce in `down_proj`.

For MoE: `FusedMoE` (`moe/fused_moe_triton/layer.py`) handles expert routing and can be configured with separate `moe_tp_size` and `moe_ep_size`.

**Relevant files:**
- `llama.py:65-116` — `LlamaMLP`
- `linear.py:469-838` — `MergedColumnParallelLinear`
- `linear.py:1280-1497` — `RowParallelLinear`

**Gap:** **Minimal.** Since the global `_TP` group is already N GPUs, the FFN layers naturally shard across all N GPUs with no code changes. The All-Reduce in `down_proj` already covers N GPUs via the global `_TP` group. For MoE, the existing `moe_tp_size × moe_ep_size` configuration can be set to partition N GPUs appropriately.

### 3.7 Model Layer Orchestration

**Current logic:** `LlamaAttention.forward()` (`llama.py:218-242`) is a straight pipeline:
```
QKV proj → RoPE → RadixAttention → o_proj
```
No communication between attention output and o_proj. All layers use the same TP width.

`LlamaDecoderLayer.forward()` (`llama.py:296-318`):
```
input_layernorm → self_attn → post_attention_layernorm → mlp
```
Residual connections handled by fused RMSNorm.

**Relevant files:**
- `llama.py:218-242` — `LlamaAttention.forward()`
- `llama.py:296-318` — `LlamaDecoderLayer.forward()`

**Gap:** `LlamaAttention.forward()` needs to be modified to insert the All-to-All + rescale step between `RadixAttention` output and `o_proj`. The attention output shape changes from `[B, H/TP]` (standard) to `[B, Q/TPA * Hsz]` (partial, pre-merge) to `[B, H/N]` (post-merge, ready for o_proj). The model must be Helix-aware to orchestrate this transition. This pattern must be replicated in every model file that defines its own attention class (not just Llama).

### 4.8 CUDA Graphs

**Current logic:** `CudaGraphRunner` (`cuda_graph_runner.py`) captures the entire model forward pass as a CUDA graph for each batch size. During replay, input tensors are copied into pre-allocated buffers, the graph executes, and output is sliced. Graph capture calls `attn_backend.init_forward_metadata_capture_cuda_graph()`.

**Relevant files:**
- `cuda_graph_runner.py:476-537` — capture
- `cuda_graph_runner.py:776-889` — replay

**Gap:** CUDA graphs with NCCL collective operations (All-to-All, All-Reduce) require special handling. While NCCL operations can be captured in graphs on CUDA 12+, the Helix All-to-All introduces new communication within the captured region. This needs careful testing. Initial implementation should disable CUDA graphs when Helix is enabled.

### 3.9 Existing Analogues: dp_attention

SGLang's dp_attention (`dp_attention.py`) is the closest existing feature to Helix. Both use different parallelism for attention vs FFN:

| Aspect | dp_attention | Helix |
|--------|-------------|-------|
| Attention parallelism | Data-parallel (different **requests** per GPU) | KVP (different **KV shards** per GPU, same requests) |
| FFN parallelism | TP across attn_tp_size GPUs | TP across all N GPUs |
| Communication | All-Gather/Reduce-Scatter to transition between DP and TP | All-to-All to transition between KVP and TP |
| Process groups | `_ATTN_TP` (smaller) + `_TP` (full) | `_HELIX_KVP` (new) + `_HELIX_TPA` (new) + `_TP` (full) |
| RowParallel reduce | `use_dp_attention_reduce` flag switches group | o_proj uses global `_TP` (no flag needed) |
| Weight loading | Same TP rank for all layers within a phase | Different TP rank for QKV (tpa_rank) vs o_proj/FFN (global rank) |

dp_attention demonstrates that SGLang can support asymmetric parallelism, but Helix's specific requirements (KV sharding, partial attention + LSE, All-to-All merge) have no existing implementation.

### 3.10 Gap Summary

| Component | Current State | Gap for Helix | Severity |
|-----------|--------------|---------------|----------|
| **Server args** | Single TP width | No KVP/TPA config | Low — add params |
| **Process groups** | TP, ATTN_TP, PP, MOE_EP | No KVP group; must be PP-partition-local | Medium — new groups |
| **Weight sharding** | `tp_rank`/`tp_size` per layer (infra exists) | No model passes different values per sublayer | Low — plumbing |
| **KV cache** | Full KV on every GPU | No sequence-dim sharding, no round-robin, no conditional save | **High** — fundamental change |
| **Head-count / KV pool** | `get_attention_tp_size()` → N for head counts | KV pool, backends must use TPA (not N) for head division | **High** — shape mismatch if wrong |
| **Attention output** | Merged output only | No partial + LSE return path | **High** — backend changes |
| **All-to-All + merge** | Does not exist | Entirely new | **High** — new code |
| **Model orchestration** | Uniform TP per layer | Must insert All-to-All between attn and o_proj | Medium — per-model changes |
| **FFN** | Uses global TP | Already correct for TP=N | None |
| **Radix/prefix cache** | Reads `req_to_token` by global token length | Incompatible with compact-local `req_to_token`; disable initially | Medium — disable or redesign |
| **Backend coverage** | 15+ backends, all use `get_attention_tp_size()` | Initial bring-up: Triton only (merged LSE from modified stage 2); FlashInfer deferred (no decode LSE API); others gated | Low — startup validation |
| **CUDA graphs** | Captures full forward | Cannot capture All-to-All without work | Low — disable initially |

The four **high-severity** gaps are: (1) KV cache sharding across GPUs, (2) head-count/KV pool topology, (3) attention backends returning partial output + LSE, and (4) the All-to-All + merge operation. These are the core new functionality that Helix requires.

---

## 4. Implementation Changes

### 4.1 Configuration & Server Args

**File:** `python/sglang/srt/server_args.py`

Add new arguments:

```python
# Helix parallelism configuration
helix_kvp_size: int = 1          # KVP width (number of KV shards across sequence dim)
helix_tpa_size: int = 0          # TP width for attention (0 = auto: tp_size // helix_kvp_size)
helix_kvp_chunk_size: int = 16   # Round-robin chunk size for KV cache assignment
```

**Validation logic** (in `__post_init__` or a dedicated validator):

```python
if helix_kvp_size > 1:
    # Helix mode enabled

    # Auto-resolve TPA size before any assertions.
    # ModelConfig exposes get_total_num_kv_heads() (model_config.py:572) which
    # handles Falcon, GPTBigCode, and other model-specific edge cases.
    # The raw field is model_config.num_key_value_heads (model_config.py:528).
    total_kv_heads = model_config.get_total_num_kv_heads()
    if helix_tpa_size == 0:
        # Auto-resolve: TPA is uniquely determined by the topology equation
        #   kvp_size * tpa_size == tp_size  →  tpa_size = tp_size // kvp_size
        # There is no freedom to choose a different value — any other TPA
        # would violate the Helix topology invariant.
        helix_tpa_size = tp_size // helix_kvp_size
        self.helix_tpa_size = helix_tpa_size

    assert helix_kvp_size * helix_tpa_size == tp_size, \
        f"KVP({helix_kvp_size}) × TPA({helix_tpa_size}) must equal TP({tp_size})"
    assert helix_tpa_size <= total_kv_heads, \
        f"TPA({helix_tpa_size}) must be ≤ num_kv_heads({total_kv_heads})"
    # QKVParallelLinear (linear.py:891-895) uses divide(total_num_kv_heads, tp_size)
    # which asserts exact divisibility.  TPA is the tp_size for the attention
    # QKV projection, so total_kv_heads must be divisible by TPA.
    # If not, the configuration is unsupported — fail fast with a clear error
    # rather than letting QKVParallelLinear crash with an opaque assert.
    if total_kv_heads % helix_tpa_size != 0:
        raise ValueError(
            f"total_kv_heads ({total_kv_heads}) must be divisible by "
            f"helix_tpa_size ({helix_tpa_size}). QKVParallelLinear requires "
            f"exact divisibility for KV head sharding (linear.py:895). "
            f"Choose a different --helix-kvp-size so that "
            f"tp_size // kvp_size divides total_kv_heads."
        )

    # Backend gating: Helix requires the attention backend to return
    # partial output + merged LSE (section 4.5.2).  For initial bring-up,
    # only the Triton backend is supported — it uses a modified stage 2
    # kernel that writes merged LSE (see section 4.5.2 kernel change).
    #
    # FlashInfer's BatchDecodeWithPagedKVCacheWrapper.forward() does not
    # currently expose a merged-LSE return for decode
    # (flashinfer_backend.py:888; forward_return_lse exists only on
    # prefill wrappers at lines 837, 845).  FlashInfer support is deferred
    # until a decode LSE API is confirmed — see section 4.5.2 for the
    # design sketch.
    #
    # Other backends (wave_backend, aiter_backend, etc.) hardcode
    # get_attention_tp_size() for head counts and have no partial-output path.
    _HELIX_SUPPORTED_BACKENDS = {"triton"}
    if attention_backend not in _HELIX_SUPPORTED_BACKENDS:
        raise ValueError(
            f"Helix parallelism currently requires attention_backend in "
            f"{_HELIX_SUPPORTED_BACKENDS}, got '{attention_backend}'. "
            f"FlashInfer decode does not yet expose merged LSE return; "
            f"other backends lack Helix head-count plumbing."
        )

    # Radix cache is not compatible with compact-local req_to_token (section 4.4g).
    # ServerArgs uses disable_radix_cache (server_args.py:579), not enable_radix_cache.
    if not self.disable_radix_cache:
        logger.warning("Radix cache disabled: not yet compatible with Helix parallelism.")
        self.disable_radix_cache = True

    # DP-attention and context parallelism reshape the attention topology in
    # ways that conflict with Helix's KVP/TPA group layout.
    # - DP-attention (server_args.py:598) splits requests across attention
    #   groups; Helix assumes all KVP ranks attend the same request set.
    # - Context parallelism (server_args.py:422, attn_cp_size) shards
    #   sequences for prefill in a different dimension than KVP.
    if self.enable_dp_attention:
        raise ValueError(
            "Helix parallelism is not compatible with DP-attention "
            "(--enable-dp-attention). Disable one or the other."
        )
    if self.attn_cp_size > 1:
        raise ValueError(
            f"Helix parallelism is not compatible with context parallelism "
            f"(attn_cp_size={self.attn_cp_size}). Set --attn-cp-size 1."
        )

    # Paged KV allocation (page_size > 1) has page-boundary continuation
    # semantics in alloc_decode() / alloc_extend() (allocator.py:463, 398)
    # that are incompatible with Helix's selective per-request allocation.
    # The paged allocator's alloc() also requires page-aligned sizes.
    # Initial Helix implementation targets page_size == 1 only.
    if self.page_size > 1:
        raise ValueError(
            f"Helix parallelism requires --page-size 1 (got {self.page_size}). "
            f"Paged KV allocation is not yet supported with Helix."
        )

    # Speculative decoding bypasses the normal decode preparation path
    # (schedule_batch.py:1957-1960 returns early when spec is enabled)
    # and uses separate spec paths that allocate multiple tokens.  The
    # Helix decode allocation (section 4.4d) and release (section 4.4d
    # release_kv_cache_helix) assume the non-speculative decode flow.
    # Reject until full spec-path design is added.
    if self.speculative_algorithm is not None:
        raise ValueError(
            f"Helix parallelism is not yet compatible with speculative decoding "
            f"(speculative_algorithm={self.speculative_algorithm!r}). "
            f"Disable speculative decoding when using Helix."
        )
```

When `helix_kvp_size == 1`, Helix is disabled and everything falls back to standard TP.

**Helper property:**

```python
@property
def is_helix_enabled(self) -> bool:
    return self.helix_kvp_size > 1
```

### 4.2 Process Groups

**File:** `python/sglang/srt/distributed/parallel_state.py`

Create three new communication groups. With N = KVP × TPA GPUs mapped as `tp_local_rank = kvp_rank * TPA + tpa_rank`:

> **PP-safety note:** SGLang's existing TP groups are PP-partition-local — ranks are `[pp_idx * tp_size, ..., (pp_idx+1) * tp_size - 1]` (see `parallel_state.py:1662`). Helix groups must follow the same convention: use the **PP-partition-local TP rank** (not the global rank) as the basis for KVP/TPA sub-group construction. This ensures correctness when PP > 1.

**a) TPA group** — GPUs with the same KVP rank (for intra-attention TP All-Reduce if TPA > 1):
```
For KVP=2, TPA=4, N=8:
  TPA group 0: [GPU 0, 1, 2, 3]     (kvp_rank=0)
  TPA group 1: [GPU 4, 5, 6, 7]     (kvp_rank=1)
```

**b) KVP group** — GPUs with the same TPA rank (for All-to-All after attention):
```
For KVP=2, TPA=4, N=8:
  KVP group 0: [GPU 0, 4]           (tpa_rank=0)
  KVP group 1: [GPU 1, 5]           (tpa_rank=1)
  KVP group 2: [GPU 2, 6]           (tpa_rank=2)
  KVP group 3: [GPU 3, 7]           (tpa_rank=3)
```

**c) TPF group** — All N GPUs (for FFN All-Reduce). This is the existing `_TP` group when `tp_size = N`.

**Implementation:**

Add to `parallel_state.py` and call from `model_runner.py`:

**Call site:** In `ModelRunner.init_torch_distributed()` (`model_runner.py:813`), immediately after the existing `initialize_model_parallel(...)` and `initialize_dp_attention(...)` calls:

```python
# model_runner.py — after existing init calls (line ~821):
if self.server_args.is_helix_enabled:
    initialize_helix_groups(
        kvp_size=self.server_args.helix_kvp_size,
        tpa_size=self.server_args.helix_tpa_size,
    )
```

**Definition** in `parallel_state.py`:

```python
# New globals
_HELIX_KVP: Optional[GroupCoordinator] = None
_HELIX_TPA: Optional[GroupCoordinator] = None
# _TP already serves as TPF group (tp_size = N)

def initialize_helix_groups(kvp_size: int, tpa_size: int):
    """Create Helix-specific process groups.

    Groups are built relative to the PP-partition-local TP rank, NOT the
    global rank, so that Helix is compatible with PP > 1.

    Uses init_model_parallel_group() (parallel_state.py:1361) — the standard
    factory that constructs GroupCoordinator from a list of rank lists and
    configures NCCL/custom-allreduce/gloo backends.
    """
    global _HELIX_KVP, _HELIX_TPA

    # PP-partition-local TP rank (mirrors how _TP is built in
    # initialize_model_parallel — see parallel_state.py:1662)
    tp_rank = get_tensor_model_parallel_rank()      # 0..N-1 within PP partition
    tp_group_ranks = get_tp_group().ranks            # global ranks in this PP partition
    local_rank = get_tp_group().local_rank
    backend = torch.distributed.get_backend(get_tp_group().device_group)

    # TPA groups: GPUs with same kvp_rank
    tpa_all_ranks = []
    for kvp_r in range(kvp_size):
        local_indices = [kvp_r * tpa_size + t for t in range(tpa_size)]
        global_ranks = [tp_group_ranks[i] for i in local_indices]
        tpa_all_ranks.append(global_ranks)
    _HELIX_TPA = init_model_parallel_group(
        group_ranks=tpa_all_ranks,
        local_rank=local_rank,
        backend=backend,
        group_name="helix_tpa",
    )

    # KVP groups: GPUs with same tpa_rank
    kvp_all_ranks = []
    for tpa_r in range(tpa_size):
        local_indices = [k * tpa_size + tpa_r for k in range(kvp_size)]
        global_ranks = [tp_group_ranks[i] for i in local_indices]
        kvp_all_ranks.append(global_ranks)
    _HELIX_KVP = init_model_parallel_group(
        group_ranks=kvp_all_ranks,
        local_rank=local_rank,
        backend=backend,
        group_name="helix_kvp",
    )
```

Add accessor functions:

```python
def get_helix_kvp_group() -> GroupCoordinator
def get_helix_tpa_group() -> GroupCoordinator
def get_helix_kvp_rank() -> int      # rank within KVP group
def get_helix_kvp_size() -> int      # KVP width
def get_helix_tpa_rank() -> int      # rank within TPA group
def get_helix_tpa_size() -> int      # TPA width
```

**Rank calculation (PP-partition-local):**

```python
tp_rank = get_tensor_model_parallel_rank()  # 0..N-1 within PP partition
kvp_rank = tp_rank // tpa_size
tpa_rank = tp_rank % tpa_size
```

### 4.3 Weight Sharding & Loading

**File:** `python/sglang/srt/layers/linear.py`

The key challenge: `qkv_proj` uses TPA-based sharding (replicated across KVP) while `o_proj` and FFN use N-based sharding. Linear layers currently derive `tp_rank` and `tp_size` from the global TP state or constructor arguments.

**Approach:** Pass explicit `tp_rank` and `tp_size` to linear layers in the model constructor.

For `QKVParallelLinear` and `MergedColumnParallelLinear` used in QKV:
```python
# In LlamaAttention.__init__():
self.qkv_proj = QKVParallelLinear(
    hidden_size, head_dim, total_num_heads, total_num_kv_heads,
    tp_size=helix_tpa_size,      # Shard across TPA only
    tp_rank=helix_tpa_rank,      # Use TPA rank for shard selection
    ...
)
```

For `RowParallelLinear` (o_proj, down_proj):
```python
# Default tp_rank / tp_size already come from global TP state (= N)
# No change needed if global TP is already N
self.o_proj = RowParallelLinear(
    total_num_heads * head_dim, hidden_size,
    # tp_size=N (default), tp_rank=global_rank (default)
    ...
)
```

For `MergedColumnParallelLinear` (gate_up_proj):
```python
# Default tp_rank / tp_size = N (global TP)
self.gate_up_proj = MergedColumnParallelLinear(
    hidden_size, [intermediate_size] * 2,
    # tp_size=N (default), tp_rank=global_rank (default)
    ...
)
```

**Weight loading verification:** `QKVParallelLinear` and `ColumnParallelLinear` already accept `tp_rank` and `tp_size` constructor parameters (see `linear.py:277-329`). The weight loader uses `tp_rank * shard_size` as the start index. By passing `tpa_rank` and `tpa_size`, GPUs with the same `tpa_rank` but different `kvp_rank` will load identical weight shards. This is exactly the replication we need.

**RowParallelLinear All-Reduce group:** `RowParallelLinear.forward()` calls `tensor_model_parallel_all_reduce()` which uses the global `_TP` group. Since `_TP` is initialized with `tp_size = N`, the All-Reduce already covers all N GPUs. No change needed for o_proj and down_proj.

**However**, `qkv_proj` is a `ColumnParallelLinear` variant which does NOT have an All-Reduce (column-parallel outputs are already partitioned). So no communication issue there either.

**Head-count plumbing for KV pool and backends:**

The KV pool is created with `head_num = model_config.get_num_kv_heads(get_attention_tp_size())` (`model_runner_kv_cache_mixin.py:650`). With Helix, `get_attention_tp_size()` returns N (all GPUs), but the KV pool's head count should be divided by **TPA** — each TPA rank handles `K / TPA` KV heads. The backends similarly derive `num_head` and `num_kv_head` from `get_attention_tp_size()` (`triton_backend.py:89`, `flashinfer_backend.py:915`).

**Required change:** Explicitly pass TPA as the head-count divisor at each Helix-critical call site. **Do NOT override `get_attention_tp_size()` globally** — that helper is used beyond KV/head sizing (e.g., token padding alignment in `forward_batch_info.py:742`, dp_attention dispatch in `dp_attention.py:320`), so a global override would silently break unrelated code paths.

The explicit-argument approach modifies only these 4 call sites:
1. `MHATokenToKVPool.__init__()` — pass `head_num = model_config.get_num_kv_heads(tpa_size)` → KV buffer shape `[pool_size, K/TPA, head_dim]`
2. `TritonAttnBackend.__init__()` — pass `num_head = Q // tpa_size`, `num_kv_head = model_config.get_num_kv_heads(tpa_size)`
3. `FlashInferIndicesUpdaterDecode.__init__()` — pass `num_kv_heads = model_config.get_num_kv_heads(tpa_size)`
4. `RadixAttention.__init__()` — already receives TPA-local head counts from model constructor (see section 4.7.1)

Each site currently calls `get_attention_tp_size()` to derive head counts. With Helix, the caller passes the pre-computed TPA-based count instead. `get_attention_tp_size()` itself is left unchanged.

> **Note on other backends:** `wave_backend.py:142-147` and `aiter_backend.py:1881-1883` also hardcode `get_attention_tp_size()` for head counts. Rather than patching every backend, Helix mode is **gated to Triton only** at startup — see section 4.1 validation logic. FlashInfer is deferred until its decode wrapper exposes merged LSE return (see section 4.5.2). If future work extends Helix to more backends, they must also adopt explicit TPA-based head arguments and provide a partial-output + LSE return path.

This ensures the KV cache is correctly sized for TPA-based head sharding, and attention kernels receive the correct head dimensions, without global side effects.

### 4.4 KV Cache Sharding

**Files:**
- `python/sglang/srt/mem_cache/memory_pool.py`
- `python/sglang/srt/managers/schedule_batch.py`

Each KVP rank stores only S/KVP of the KV history per request. The key changes:

**a) KV cache pool sizing:**

In `MHATokenToKVPool.__init__()` (`memory_pool.py`), the pool size determines how many tokens' KV can be stored. With Helix, each GPU stores 1/KVP of the total KV:

```python
# In model_runner.py where pool_size is calculated:
if server_args.is_helix_enabled:
    # Each GPU stores only its KV shard
    effective_pool_size = total_pool_size // server_args.helix_kvp_size
```

This means each GPU can hold KVP times more requests (same total KV budget, but each request uses 1/KVP as much space per GPU).

**b) Round-robin KV assignment during decode:**

New tokens are assigned to KVP ranks in round-robin chunks. Add a helper to determine the designated KVP rank:

```python
def get_kvp_rank_for_token(seq_position: int, kvp_size: int, chunk_size: int) -> int:
    """Which KVP rank stores the KV for this token position."""
    return (seq_position // chunk_size) % kvp_size
```

**c) Selective KV cache save:**

During decode, only the designated KVP rank for each request's new token should write K,V to its cache. Different requests in a batch may designate different KVP ranks (because they have different `seq_len` values), so **per-request control** is needed.

> **Backend interface constraint:** The existing attention backends (`triton_backend.py:1019`, `flashinfer_backend.py:882`) use `save_kv_cache` as a **scalar bool** in control flow (`if save_kv_cache:`). Passing a per-request tensor directly would break this interface.

**Approach:** Keep `save_kv_cache` as a scalar bool (always `True` during normal decode). Add a separate **per-request write mask** (`helix_kv_write_mask`) to `ForwardBatch`. The mask is applied inside `set_kv_buffer()` to selectively write K,V only for designated requests:

```python
# In ForwardBatch (computed during prepare_for_decode):
my_kvp_rank = get_helix_kvp_rank()
designated_ranks = get_kvp_rank_for_token(batch.seq_lens - 1, kvp_size, chunk_size)
forward_batch.helix_kv_write_mask = (designated_ranks == my_kvp_rank)  # [bs] bool

# save_kv_cache remains a scalar True — the backend's control flow is unchanged
```

**Decode write masking — keep `set_kv_buffer()` ABI unchanged:**

The existing `set_kv_buffer()` signature (`memory_pool.py:951`) is `set_kv_buffer(self, layer, loc, cache_k, cache_v, ...)`. Backends call it as `forward_batch.token_to_kv_pool.set_kv_buffer(layer, out_cache_loc, k, v, ...)` (`triton_backend.py:1020`, `flashinfer_backend.py:883`). **Do not change this signature.**

Instead, filter `out_cache_loc`, `k`, and `v` **before** they reach `set_kv_buffer()`, in a thin wrapper called from the Helix attention path:

```python
def helix_filtered_set_kv_buffer(forward_batch, layer, cache_loc, k, v, **kwargs):
    """Write KV only for designated requests (decode path).

    Filters cache_loc/k/v by helix_kv_write_mask, then calls the
    standard set_kv_buffer() with the original signature.
    """
    mask = forward_batch.helix_kv_write_mask          # [bs] bool
    if mask is not None and not mask.all():
        cache_loc = cache_loc[mask]
        k = k[mask]
        v = v[mask]
    forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v, **kwargs)
```

This preserves the existing `set_kv_buffer()` ABI. The wrapper is called only in the Helix decode path (section 4.7.1); non-Helix paths are completely unaffected.

**d) KV cache allocation (`prepare_for_decode`):**

In `schedule_batch.py:prepare_for_decode()`, `alloc_for_decode(token_per_req=1)` allocates one KV slot per request and writes the allocated indices into `req_to_token` for ALL requests (`common.py:423`). With Helix, only the designated KVP rank needs to allocate. This requires a shard-aware allocation path.

> **Corruption risk (P0):** The current code always writes allocated indices into `req_to_token` for every request. If a non-designated rank skips allocation but `req_to_token` retains a default-initialized zero, the release path (`release_kv_cache`, `common.py:496`) will read that zero and free pool slot 0 — corrupting the free list. Additionally, `allocator.free()` (`allocator.py:155`) appends indices to the free list without validation.

**Approach — shard-aware `alloc_for_decode_helix()`:**

> **Timing note:** `alloc_for_decode_helix()` runs inside `ScheduleBatch.prepare_for_decode()` (`schedule_batch.py:1993`), which executes **before** `ForwardBatch` is constructed (`forward_batch_info.py:376`). Therefore `helix_local_kv_lens` must be computed at `ScheduleBatch` level, not in `ForwardBatch.init_new()`. Since the computation depends only on `seq_lens`, `kvp_rank`, `kvp_size`, and `chunk_size` — all available at scheduler time — this is straightforward. The `ScheduleBatch` stores these values and later propagates them into `ForwardBatch` (see section 4.7.4).

```python
def alloc_for_decode_helix(batch: ScheduleBatch, kvp_rank, kvp_size, chunk_size):
    """Allocate KV slots only for requests designated to this KVP rank.

    Called from ScheduleBatch.prepare_for_decode(), before ForwardBatch exists.
    Uses the same allocator API as the existing alloc_for_decode() (common.py:423):
      - batch.token_to_kv_pool_allocator  (ScheduleBatch field, line 1208)
      - batch.tree_cache                  (for eviction)
      - batch.req_to_token_pool           (for req_to_token writes)
    """
    # 0. Compute local KV lengths at scheduler time (NOT from ForwardBatch)
    local_kv_lens = compute_local_kv_lens(
        batch.seq_lens, kvp_rank, kvp_size, chunk_size
    )
    batch.helix_local_kv_lens = local_kv_lens  # stash for later ForwardBatch propagation

    # 1. Compute which requests belong to this rank
    designated = get_kvp_rank_for_token(batch.seq_lens, kvp_size, chunk_size)
    mask = (designated == kvp_rank)                  # [bs] bool
    designated_indices = mask.nonzero(as_tuple=True)[0]
    num_alloc = designated_indices.numel()

    if num_alloc == 0:
        return  # Nothing to allocate on this rank

    # 2. Allocate via the allocator (mirrors alloc_token_slots, common.py:201)
    allocator = batch.token_to_kv_pool_allocator
    evict_from_tree_cache(batch.tree_cache, num_alloc)
    new_locs = allocator.alloc(num_alloc)

    # 3. Write into req_to_token ONLY for designated requests
    #    Use local_kv_len as the write position (compact local indexing,
    #    see section 4.4f)
    for i, req_idx in enumerate(designated_indices):
        local_pos = local_kv_lens[req_idx]              # next local write pos
        batch.req_to_token_pool.req_to_token[
            batch.req_pool_indices[req_idx], local_pos
        ] = new_locs[i]

    # 4. Update out_cache_loc for the attention write mask
    #    (only designated requests have valid out_cache_loc)
    batch.out_cache_loc[designated_indices] = new_locs

    # 5. Increment local_kv_lens for designated requests.
    #    alloc_for_decode_helix() is called BEFORE seq_lens is incremented
    #    (schedule_batch.py:1993 vs 2001-2011), so local_kv_lens from step 0
    #    reflects pre-allocation lengths.  After allocating and writing the
    #    new slot into req_to_token, the designated requests now have one more
    #    local token.  Without this increment, backend metadata (kv_indptr /
    #    kv_indices) would miss the just-allocated decode token.
    batch.helix_local_kv_lens[designated_indices] += 1
```

**Shard-aware `release_kv_cache()`:**

Each KVP rank tracks which `req_to_token` positions it actually owns (positions 0..local_kv_len-1). On release, it only frees those entries. The release must align with the real request KV lifecycle tracked by `kv_committed_len` / `kv_allocated_len` (schedule_batch.py:836-854).

> **Why not delegate to `cache_finished_req()`:** With radix cache disabled (enforced in 4.1), ChunkCache is used. `ChunkCache.cache_finished_req()` (chunk_cache.py:65-71) reads `req_to_token[:kv_committed_len]` using **global** position coordinates, which is wrong for Helix's compact-local layout (section 4.4f). Instead, we replicate the lifecycle with local coordinate mapping.

```python
def release_kv_cache_helix(req, tree_cache, kvp_rank, kvp_size, chunk_size):
    """Free only KV slots owned by this KVP rank.

    Aligns with the real request KV lifecycle used by release_kv_cache()
    (common.py:465). Respects kv_committed_len / kv_allocated_len tracking
    and properly sets lifecycle flags via pop_committed_kv_cache() and
    pop_overallocated_kv_cache().

    Uses Req.seqlen property (schedule_batch.py:817) — NOT req.seq_len
    (which does not exist).
    """
    if req.req_pool_idx is None:
        return

    # Step 1: Free committed KV cache (mirrors ChunkCache.cache_finished_req,
    # chunk_cache.py:65, but with local coordinate mapping).
    # pop_committed_kv_cache() returns global kv_committed_len and sets the
    # kv_committed_freed flag (schedule_batch.py:836).
    global_committed = req.pop_committed_kv_cache()
    local_committed = compute_local_kv_len(
        global_committed, kvp_rank, kvp_size, chunk_size
    )
    if local_committed > 0:
        committed_indices = tree_cache.req_to_token_pool.req_to_token[
            req.req_pool_idx, :local_committed
        ]
        tree_cache.token_to_kv_pool_allocator.free(committed_indices)

    # Step 2: Free overallocated KV cache (mirrors common.py:481-499).
    # pop_overallocated_kv_cache() returns (kv_committed_len, kv_allocated_len)
    # and sets the kv_overallocated_freed flag (schedule_batch.py:844).
    # With no speculative decoding, start_p == end_p (no overallocation).
    start_p, end_p = req.pop_overallocated_kv_cache()
    if start_p < end_p:
        local_start = compute_local_kv_len(start_p, kvp_rank, kvp_size, chunk_size)
        local_end = compute_local_kv_len(end_p, kvp_rank, kvp_size, chunk_size)
        if local_start < local_end:
            overalloc_indices = tree_cache.req_to_token_pool.req_to_token[
                req.req_pool_idx, local_start:local_end
            ]
            tree_cache.token_to_kv_pool_allocator.free(overalloc_indices)

    # Step 3: Free the request pool slot (same as standard path, common.py:508).
    tree_cache.req_to_token_pool.free(req)
```

**Key invariants:**
- `req_to_token` positions beyond `local_committed + local_overallocated` are never read, so default zeros cannot leak into the free list.
- Each rank's `req_to_token` is a compact local view (see 4.4f), so positions 0..local_kv_len-1 are always valid.
- Lifecycle flags (`kv_committed_freed`, `kv_overallocated_freed`) are properly set via the `pop_*` APIs, preventing double-free.
- `out_cache_loc` for non-designated requests is unused (guarded by `helix_kv_write_mask` in the Helix decode path, see 4.4c).

**e) Prefill KV distribution:**

During prefill, the KV cache for the prompt must be distributed across KVP ranks using compact-local `req_to_token` semantics. The current prefill write path (`write_cache_indices`, `common.py:116-121`) writes to `req_to_token` using **global** position slices:

```python
# Current code (common.py:116-121):
req_to_token_pool.write((req_idx, slice(0, prefix_len)), prefix_tensors[i])
req_to_token_pool.write((req_idx, slice(prefix_len, seq_len)), out_cache_loc[...])
```

With the compact-local `req_to_token` design (section 4.4f), these global slices are incorrect. A Helix-specific prefill write path is needed.

**Approach — `alloc_for_extend_helix()` and `write_cache_indices_helix()`:**

```python
def alloc_for_extend_helix(batch: ScheduleBatch, kvp_rank, kvp_size, chunk_size):
    """Allocate prefill KV cache only for tokens owned by this KVP rank.

    Called from the extend/prefill path (common.py:alloc_for_extend).
    Uses batch.token_to_kv_pool_allocator (ScheduleBatch field, line 1208)
    for allocation — the same allocator used by alloc_token_slots().
    """
    allocator = batch.token_to_kv_pool_allocator

    owned_locs_per_req = []
    for i, req in enumerate(batch.reqs):
        prefix_len = batch.prefix_lens[i]
        seq_len = batch.seq_lens[i]

        # Determine which token positions this KVP rank owns
        owned_global_positions = [
            p for p in range(prefix_len, seq_len)
            if get_kvp_rank_for_token(p, kvp_size, chunk_size) == kvp_rank
        ]
        num_owned = len(owned_global_positions)

        if num_owned == 0:
            owned_locs_per_req.append(None)
            continue

        # Allocate only the owned token slots
        evict_from_tree_cache(batch.tree_cache, num_owned)
        new_locs = allocator.alloc(num_owned)
        owned_locs_per_req.append(new_locs)

        # Write to req_to_token at compact local positions
        for j, global_pos in enumerate(owned_global_positions):
            local_pos = global_to_local_pos(global_pos, kvp_rank, kvp_size, chunk_size)
            batch.req_to_token_pool.req_to_token[req.req_pool_idx, local_pos] = new_locs[j]

    # Compute and stash local KV lengths for later ForwardBatch propagation
    batch.helix_local_kv_lens = compute_local_kv_lens(
        batch.seq_lens, kvp_rank, kvp_size, chunk_size
    )

    return owned_locs_per_req  # used by build_helix_prefill_cache_loc()
```

**Prefix cache integration:** For cached prefix tokens, `prefix_tensors[i]` contains pool indices from a prior allocation. Under Helix, these must also be written to local positions:

```python
# Prefix tokens already have KV pool indices.  Write them at compact local
# positions instead of the global slice(0, prefix_len):
for p in range(prefix_len):
    if get_kvp_rank_for_token(p, kvp_size, chunk_size) == kvp_rank:
        local_p = global_to_local_pos(p, kvp_rank, kvp_size, chunk_size)
        req_to_token_pool.req_to_token[req_idx, local_p] = prefix_tensors[i][p]
```

**KV buffer writes during prefill — per-token filtering:**

During extend/prefill, `set_kv_buffer()` is called with the full `k` and `v` tensors covering **all tokens** in the batch and a token-aligned `out_cache_loc` (`triton_backend.py:816`, `flashinfer_backend.py:773`). The `helix_kv_write_mask` from section 4.4c is `[bs]` (per-request) and cannot filter individual tokens within a multi-token prefill request.

**Approach: Build a per-token `out_cache_loc` with sentinel values for non-owned positions.**

`alloc_for_extend_helix()` returns `out_cache_loc` sized to the **full** token count (same as standard path), but sets non-owned positions to a **sentinel pool index** that maps to a scratch/dummy KV slot:

```python
def build_helix_prefill_cache_loc(batch: ScheduleBatch, kvp_rank, kvp_size,
                                   chunk_size, owned_locs_per_req,
                                   helix_scratch_slot: int):
    """Build out_cache_loc for the full extend token stream.

    Owned positions get real pool indices from alloc_for_extend_helix().
    Non-owned positions get a sentinel index pointing to a scratch KV slot
    that is never read by any attention kernel (allocated once at pool init).

    Args:
        helix_scratch_slot: A single pool index reserved at KV pool creation
            time as a write-only scratch slot.  Allocated once via
            token_to_kv_pool_allocator.alloc(1) during pool initialization
            and stored on the pool object (e.g., MHATokenToKVPool.helix_scratch_slot).
    """
    sentinel = helix_scratch_slot
    out_cache_loc = torch.full(
        (batch.extend_num_tokens,), sentinel, dtype=torch.int64, device=batch.device
    )
    pt = 0
    for i, req in enumerate(batch.reqs):
        prefix_len = batch.prefix_lens[i]
        seq_len = batch.seq_lens[i]
        owned_idx = 0
        for p in range(prefix_len, seq_len):
            if get_kvp_rank_for_token(p, kvp_size, chunk_size) == kvp_rank:
                out_cache_loc[pt] = owned_locs_per_req[i][owned_idx]
                owned_idx += 1
            # else: stays sentinel — set_kv_buffer writes harmlessly to scratch
            pt += 1
    return out_cache_loc
```

This preserves token-for-token alignment between `k/v` tensors and `out_cache_loc`. The `set_kv_buffer()` call proceeds unmodified — writes to the sentinel slot are harmless overwrites to a scratch buffer. The attention kernel never indexes into the scratch slot because `req_to_token` only contains real pool indices for owned positions.

> **Note:** `helix_kv_write_mask` (per-request, `[bs]`) is still used for the **decode** path (section 4.4c), where each request contributes exactly one token and per-request granularity is sufficient. For prefill, the per-token sentinel approach above is needed instead.

**f) Metadata initialization (kv_indptr / kv_indices):**

> **Correctness note (P0):** The attention backend's index builders assume **contiguous** `req_to_token` ranges. Specifically, `create_flashinfer_kv_indices_triton` (`utils.py:17`) iterates `[kv_start : kv_start + kv_len)` per request and copies those `req_to_token` entries into `kv_indices`. If the round-robin assignment left interleaved gaps in `req_to_token` (owned/not-owned positions mixed together), the contiguous range assumption would produce incorrect indices.

**Design: Compact local `req_to_token` view.**

Each KVP rank maintains a **compact** local `req_to_token` mapping where positions `[0, 1, ..., local_kv_len-1]` correspond to the tokens this rank owns, in order of assignment. No gaps, no interleaving. This preserves the contiguous range assumption used by index builders.

The mapping from global token position → local position is:

```python
def global_to_local_pos(global_pos: int, kvp_rank: int, kvp_size: int, chunk_size: int) -> int:
    """Convert a global sequence position to a compact local position on this KVP rank."""
    chunk_idx = global_pos // chunk_size
    intra_chunk_offset = global_pos % chunk_size
    # How many full round-robin cycles before this chunk?
    full_cycles = chunk_idx // kvp_size
    # Local position = full_cycles * chunk_size + intra_chunk_offset
    return full_cycles * chunk_size + intra_chunk_offset

def compute_local_kv_len(global_seq_len: int, kvp_rank: int, kvp_size: int, chunk_size: int) -> int:
    """Number of tokens owned by this KVP rank for a request with global_seq_len tokens."""
    full_chunks = global_seq_len // chunk_size
    remainder = global_seq_len % chunk_size
    # Chunks assigned to this rank
    my_full_chunks = full_chunks // kvp_size + (1 if (full_chunks % kvp_size) > kvp_rank else 0)
    local_len = my_full_chunks * chunk_size
    # If remainder tokens exist and the partial chunk belongs to this rank
    if remainder > 0 and (full_chunks % kvp_size) == kvp_rank:
        local_len += remainder
    return local_len
```

**Writes to `req_to_token`:** During allocation (section 4.4d), KV slots are written at position `local_kv_len` (the next compact position), NOT at the global sequence position. This ensures `req_to_token[req, 0:local_kv_len]` is always a contiguous, gap-free array of valid pool indices.

**Backend usage:** `init_forward_metadata()` uses `helix_local_kv_lens` (from `ForwardBatch`) instead of `seq_lens` for `kv_indptr` construction. Since the local `req_to_token` is compact, the existing `create_flashinfer_kv_indices_triton` kernel works without modification:

```python
# In init_forward_metadata():
if is_helix_enabled:
    kv_lens = forward_batch.helix_local_kv_lens   # compact local lengths
else:
    kv_lens = forward_batch.seq_lens               # standard full lengths

# kv_indptr, kv_indices built from kv_lens and req_to_token as usual
# The contiguous range [0 : local_kv_len) in req_to_token is valid
```

**Important:** Global `seq_lens` are still maintained on ALL ranks (needed for position embeddings, scheduling decisions, and round-robin assignment). Only the attention backend indexing uses `local_kv_lens`.

**g) Radix/prefix cache compatibility:**

The radix cache (`radix_cache.py:460-500`) reads `req_to_token` indexed by global token counts. With compact-local `req_to_token`, this is incorrect — global lengths exceed `local_kv_len`, so reads would hit uninitialized entries.

**Initial approach: Disable radix/prefix caching when Helix is enabled.**

```python
# In server_args validation (ServerArgs uses disable_radix_cache, not enable):
if helix_kvp_size > 1:
    if not self.disable_radix_cache:
        logger.warning("Radix cache disabled: not yet compatible with Helix parallelism.")
        self.disable_radix_cache = True
```

This is the safest initial path. The radix cache relies on global `req_to_token` semantics in multiple code paths:
- `cache_finished_req()` (`radix_cache.py:460-462`) — reads `req_to_token[:kv_committed_len]`
- `cache_unfinished_req()` (`radix_cache.py:499-500`) — reads `req_to_token[:len(token_ids)]`
- `insert()` — inserts `kv_indices` as values, which are pool indices from `req_to_token`
- `free()` paths — free pool indices obtained from these same global reads

Making radix cache Helix-aware would require each KVP rank to maintain a local radix tree with local token-to-pool mappings, and coordinate prefix matches across KVP ranks. This is a significant design effort better deferred to a follow-up.

**Validation:** Add a test (section 5) that confirms Helix mode **auto-disables** radix cache — i.e., when `disable_radix_cache` is not explicitly set, Helix validation sets it to `True` and logs a warning. The test should verify that `server_args.disable_radix_cache is True` after validation, not that an error is raised.

### 4.5 Attention Phase

**New file:** `python/sglang/srt/layers/helix_attention.py`

This module implements the post-attention All-to-All + rescale step. It sits between `RadixAttention.forward()` output and `o_proj`.

#### 4.5.1 All-to-All + Rescale Operation

```python
def helix_kvp_alltoall_and_merge(
    partial_output: torch.Tensor,  # [B, Q/TPA, Hsz] partial attention from local KV shard
    lse: torch.Tensor,             # [B, Q/TPA] log-sum-exp from local attention
    kvp_group: GroupCoordinator,   # KVP communication group
    kvp_size: int,
) -> torch.Tensor:                 # [B, Q/N, Hsz] exact attention for this GPU's final heads
    """
    Exchange partial attention results across KVP ranks and merge
    using log-sum-exp rescaling (flash-decoding merge).
    """
    B, num_heads, Hsz = partial_output.shape
    heads_per_chunk = num_heads // kvp_size  # = Q/N heads per GPU after merge

    # 1. Split heads into KVP chunks and transpose KVP to dim-0
    #    all_to_all_single splits/concatenates along dim-0, so KVP must be the
    #    leading dimension.
    # [B, Q/TPA, Hsz] → [B, KVP, Q/N, Hsz] → [KVP, B, Q/N, Hsz]
    send_output = (partial_output
        .view(B, kvp_size, heads_per_chunk, Hsz)
        .permute(1, 0, 2, 3)                       # [KVP, B, Q/N, Hsz]
        .contiguous())
    send_lse = (lse
        .view(B, kvp_size, heads_per_chunk)
        .permute(1, 0, 2)                           # [KVP, B, Q/N]
        .contiguous())

    # 2. All-to-All: exchange KVP chunks across KVP ranks
    #    Each rank sends its heads_per_chunk partition destined for every other
    #    KVP rank and receives the corresponding partition from them.
    recv_output = torch.empty_like(send_output)  # [KVP, B, Q/N, Hsz]
    recv_lse = torch.empty_like(send_lse)        # [KVP, B, Q/N]
    dist.all_to_all_single(recv_output, send_output, group=kvp_group.device_group)
    dist.all_to_all_single(recv_lse, send_lse, group=kvp_group.device_group)

    # 3. Transpose back to [B, KVP, Q/N, ...] for the merge
    recv_output = recv_output.permute(1, 0, 2, 3).contiguous()  # [B, KVP, Q/N, Hsz]
    recv_lse = recv_lse.permute(1, 0, 2).contiguous()            # [B, KVP, Q/N]

    # 4. Rescale and merge using LSE (flash-decoding formula)
    # recv_output[b, k, h, :] = partial output from KVP rank k for head h
    # recv_lse[b, k, h] = log-sum-exp from KVP rank k for head h
    max_lse = recv_lse.max(dim=1, keepdim=True).values        # [B, 1, Q/N]
    exp_weights = torch.exp(recv_lse - max_lse)                # [B, KVP, Q/N]
    weighted = recv_output * exp_weights.unsqueeze(-1)         # [B, KVP, Q/N, Hsz]
    merged = weighted.sum(dim=1)                                # [B, Q/N, Hsz]
    normalizer = exp_weights.sum(dim=1, keepdim=True)          # [B, 1, Q/N]
    output = merged / normalizer.unsqueeze(-1)                  # [B, Q/N, Hsz]

    return output  # [B, Q/N, Hsz] = [B, H/N]
```

This merge logic is identical to the Triton decode attention stage 2 kernel — the same online softmax rescaling used in flash-decoding. The difference is that "splits" here are across GPUs (KVP ranks) rather than within a single GPU.

#### 4.5.2 Extracting Partial Output + LSE from Attention

The attention backend currently returns only the final merged output. For Helix, we need the **partial** output and the **LSE** before merging, since the merge now happens across GPUs via All-to-All.

**Approach:** Add a `return_partial` flag or a Helix-specific attention mode.

**For the Triton backend** (`triton_backend.py`):

The two-stage kernel already separates partial computation (stage 1: per-split attention) from merging (stage 2: reduce across splits within this GPU). For Helix, we still run **both stages** locally — stage 2 merges intra-GPU KV splits into a single partial result for this KVP rank's KV shard. The result is "partial" only relative to the full KV across all KVP ranks; the cross-GPU merge happens in `helix_kvp_alltoall_and_merge()`.

```python
def forward_decode_helix_partial(self, q, k, v, layer, forward_batch, save_kv_cache):
    """Run both stages of decode attention on this KVP rank's local KV shard.
    Returns partial output + merged LSE for cross-KVP All-to-All merge."""
    # ... same setup as forward_decode ...

    # Run stage 1 (compute partial attention per intra-GPU KV split)
    decode_attention_fwd_stage1(
        q, key_buffer, value_buffer,
        attn_logits, attn_lse,
        kv_indptr, kv_indices,
        num_kv_splits, ...
    )

    # Run stage 2 to merge intra-GPU splits into a single partial result
    # (merges across KV splits within this GPU, NOT across KVP ranks)
    decode_attention_fwd_stage2(
        attn_logits, attn_lse,
        output, output_lse,  # Merged partial result for this GPU's KV shard
        num_kv_splits, ...
    )

    return output, output_lse  # [B, Q/TPA, Hsz], [B, Q/TPA]
```

The key insight: stage 2 merges splits **within a single GPU**. We still run stage 2 locally, but the output is "partial" relative to the full KV cache — it only covers this KVP rank's KV shard. The cross-GPU merge happens in `helix_kvp_alltoall_and_merge()`.

**Important — merged LSE output from stage 2:**

Stage 2 (`_fwd_kernel_stage2`, decode_attention.py:516-582) currently computes `e_max` and `e_sum` during the reduce loop (lines 545-572) but only writes the merged output `acc / e_sum` to `O` (line 578-582). It does **not** write the merged LSE. For Helix, we need the merged LSE per (batch, head) pair.

**Required kernel change:** Add an `output_lse` buffer parameter to `_fwd_kernel_stage2` and write the merged LSE after the reduce loop:

```python
# At end of _fwd_kernel_stage2, after the existing tl.store(O, ...):
# Merged LSE = e_max + log(e_sum)
merged_lse_val = e_max + tl.log(e_sum)
tl.store(
    Output_LSE + cur_batch * stride_lse_b + cur_head,
    merged_lse_val,
)
```

The wrapper `_decode_softmax_reducev_fwd` (line 585) gains an `output_lse` parameter (shape `[B, num_heads]`, dtype `float32`) and passes it to the kernel. The Helix-specific decode function allocates this buffer:

```python
def forward_decode_helix_partial(self, q, k, v, layer, forward_batch, save_kv_cache):
    # ... same setup ...
    B = q.shape[0]
    num_heads = layer.tp_q_head_num
    output_lse = torch.empty(B, num_heads, dtype=torch.float32, device=q.device)

    # Run stage 1
    _decode_att_m_fwd(q, k_buffer, v_buffer, attn_logits, attn_lse, ...)

    # Run modified stage 2 — writes both merged output AND merged LSE
    _decode_softmax_reducev_fwd(
        attn_logits, attn_lse, q, output, v_buffer,
        kv_indptr, num_kv_splits, max_kv_splits,
        sinks=sinks,
        output_lse=output_lse,  # NEW: merged LSE output
    )

    return output, output_lse  # [B, Q/TPA, Hsz], [B, Q/TPA]
```

**Non-Helix paths are unaffected:** The existing `decode_attention_fwd_normal()` / `decode_attention_fwd_grouped()` call `_decode_softmax_reducev_fwd()` without `output_lse`; the kernel skips the LSE store when the pointer is null (gated by a `tl.constexpr HAS_OUTPUT_LSE` flag).

**For the FlashInfer backend** (`flashinfer_backend.py`) — **future work, not in initial bring-up:**

> FlashInfer is **gated out** of initial Helix bring-up (see section 4.1 validation). This section is a design sketch for when FlashInfer decode LSE support becomes available.

FlashInfer's `BatchDecodeWithPagedKVCacheWrapper.forward()` (flashinfer_backend.py:888) returns only the merged output `o`. The prefill wrappers have `forward_return_lse()` (flashinfer_backend.py:837, 845), but the decode wrapper does not expose this API.

When FlashInfer adds decode LSE return (e.g., `return_lse=True` parameter or a `forward_return_lse` method on the decode wrapper), the integration would be:

```python
# FlashInfer API (when available):
o, lse = decode_wrapper.forward(q, kv_buffer, ..., return_lse=True)
# or: o, lse = decode_wrapper.forward_return_lse(q, kv_buffer, ...)
```

To enable FlashInfer for Helix:
1. Add `"flashinfer"` back to `_HELIX_SUPPORTED_BACKENDS` in section 4.1 validation
2. Add a runtime check: `hasattr(decode_wrapper, 'forward_return_lse')` or version check
3. Implement `forward_helix_partial_decode` on `FlashInferIndicesUpdaterDecode` (section 4.7.5)

### 4.6 FFN Phase

The FFN phase requires minimal changes since it uses standard TP with `tp_size = N`.

**File:** `python/sglang/srt/models/llama.py` (and other model files)

**Current state:** `LlamaMLP` creates `gate_up_proj` and `down_proj` with the default TP settings from the global parallel state. The global TP is already N (= KVP × TPA) since that's how many GPUs we're using.

**Change needed:** None for FFN layers themselves — the default global TP group handles the sharding and All-Reduce correctly.

**For MoE models** (e.g., DeepSeek, Mixtral):

The paper describes repartitioning N GPUs into a TPF × EP grid:
- `TPF × EP = N`
- Choose TPF vs EP to match expert size and quantity

SGLang already supports configuring `--moe-ep-size` and `--moe-tp-size` independently. For Helix, these should be set such that `moe_tp_size × moe_ep_size = N`. No new code is needed; just configuration guidance.

### 4.7 Model Layer Changes

**File:** `python/sglang/srt/models/llama.py` (primary target; pattern applies to all models)

#### 4.7.1 LlamaAttention Changes

The attention module is the primary change point. It needs to:
1. Use TPA-based sharding for QKV
2. Run local attention (using existing backend)
3. Perform All-to-All + rescale across KVP ranks
4. Use N-based sharding for o_proj

```python
class LlamaAttention(nn.Module):
    def __init__(self, config, ..., quant_config, prefix):
        super().__init__()

        # Determine Helix config
        self.is_helix = server_args.is_helix_enabled
        self.kvp_size = server_args.helix_kvp_size
        self.tpa_size = server_args.helix_tpa_size

        if self.is_helix:
            tpa_rank = get_helix_tpa_rank()

            # Compute TPA-local head counts from TOTAL heads to avoid
            # double division.  In standard Llama code, self.num_heads is
            # already TP-local (total_num_heads // tp_size).  For Helix,
            # we derive from totals directly:
            self.num_heads = self.total_num_heads // self.tpa_size
            self.num_kv_heads = max(1, self.total_num_kv_heads // self.tpa_size)
            self.q_size = self.num_heads * self.head_dim
            self.kv_size = self.num_kv_heads * self.head_dim

            # QKV projection: sharded by TPA, replicated across KVP
            self.qkv_proj = QKVParallelLinear(
                hidden_size, self.head_dim,
                self.total_num_heads, self.total_num_kv_heads,
                tp_size=self.tpa_size,
                tp_rank=tpa_rank,
                quant_config=quant_config,
                prefix=add_prefix("qkv_proj", prefix),
            )
            # o_proj: sharded by N (global TP), uses global TP group for All-Reduce
            self.o_proj = RowParallelLinear(
                self.total_num_heads * self.head_dim,
                hidden_size,
                # tp_size=N, tp_rank=global_rank (defaults)
                quant_config=quant_config,
                prefix=add_prefix("o_proj", prefix),
            )
            # RadixAttention with TPA-local head counts.
            # Signature: RadixAttention(num_heads, head_dim, scaling,
            #     num_kv_heads, layer_id, ...) — see radix_attention.py:52-57
            self.attn = RadixAttention(
                self.num_heads,                       # Q heads per TPA rank
                self.head_dim,
                self.scaling,
                num_kv_heads=self.num_kv_heads,       # KV heads per TPA rank
                layer_id=layer_id,
                quant_config=quant_config,
                prefix=add_prefix("attn", prefix),
            )
        else:
            # Standard TP (existing code, unchanged)
            ...

    def forward(self, positions, hidden_states, forward_batch):
        q, k, v = self.forward_prepare_native(positions, hidden_states)

        if self.is_helix and self.kvp_size > 1:
            # save_kv_cache remains scalar True — per-request selectivity is
            # handled by helix_filtered_set_kv_buffer() BEFORE set_kv_buffer()
            # is called, inside the forward_helix_partial backend method.
            # See section 4.7.5 for the concrete backend integration.

            # Local attention on this KVP rank's KV shard
            # Returns partial output + LSE (not fully merged)
            partial_out, lse = self.attn.forward_helix_partial(
                q, k, v, forward_batch, save_kv_cache=True
            )

            # All-to-All + rescale across KVP ranks
            attn_output = helix_kvp_alltoall_and_merge(
                partial_out, lse,
                get_helix_kvp_group(),
                self.kvp_size,
            )
        else:
            attn_output = self.attn(q, k, v, forward_batch)

        output, _ = self.o_proj(attn_output)
        return output
```

#### 4.7.2 LlamaDecoderLayer Changes

Minimal changes — the layer just calls attention and MLP:

```python
class LlamaDecoderLayer(nn.Module):
    # No structural changes needed.
    # LlamaAttention handles Helix internally.
    # LlamaMLP uses default global TP (= N), which is correct.
    pass
```

#### 4.7.3 Embedding and LM Head

`VocabParallelEmbedding` and `lm_head` use the global TP = N. All N GPUs collaborate on embedding lookup and logits computation. No changes needed.

#### 4.7.4 ForwardBatch Extensions

**File:** `python/sglang/srt/model_executor/forward_batch_info.py`

Add Helix-specific fields:

```python
class ForwardBatch:
    # Existing fields...

    # Helix fields
    helix_kv_write_mask: Optional[torch.Tensor] = None   # [bs] bool — per-request KV write mask
    helix_local_kv_lens: Optional[torch.Tensor] = None   # [bs] local KV lengths on this KVP rank
```

Populated in `ForwardBatch.init_new()`:

```python
if server_args.is_helix_enabled:
    kvp_rank = get_helix_kvp_rank()
    kvp_size = server_args.helix_kvp_size
    chunk_size = server_args.helix_kvp_chunk_size

    # Per-request KV write mask: True for requests whose new token is
    # designated to this KVP rank.  Used by set_kv_buffer() to selectively
    # write K,V (see section 4.4c).  save_kv_cache itself stays scalar True.
    designated_ranks = (batch.seq_lens - 1) // chunk_size % kvp_size
    forward_batch.helix_kv_write_mask = (designated_ranks == kvp_rank)

    # Local KV lengths: propagated from ScheduleBatch where they were
    # computed during prepare_for_decode() / alloc_for_extend_helix().
    # NOT recomputed here — ScheduleBatch is the source of truth because
    # allocation (which updates local_kv_lens) happens before ForwardBatch
    # construction (see section 4.4d timing note).
    forward_batch.helix_local_kv_lens = batch.helix_local_kv_lens
```

#### 4.7.5 Backend `forward_helix_partial` — KV Write-Mask Integration

The `forward_helix_partial` backend method is where `helix_filtered_set_kv_buffer` (defined in section 4.4c) is concretely invoked. This section shows the explicit call site for both supported backends.

**Triton backend** (`triton_backend.py`):

The standard decode path (line 1019-1022) calls `set_kv_buffer()` directly:
```python
# Standard decode (triton_backend.py:1019-1022):
if save_kv_cache:
    forward_batch.token_to_kv_pool.set_kv_buffer(
        layer, forward_batch.out_cache_loc, k, v
    )
```

The Helix decode path replaces this with the write-mask wrapper:
```python
def forward_helix_partial_decode(self, q, k, v, layer, forward_batch, save_kv_cache):
    """Helix decode: save KV (filtered), compute local attention + merged LSE."""

    # --- KV save with write-mask filtering (section 4.4c) ---
    if save_kv_cache:
        helix_filtered_set_kv_buffer(
            forward_batch, layer, forward_batch.out_cache_loc, k, v
        )
        # helix_filtered_set_kv_buffer filters out_cache_loc/k/v by
        # forward_batch.helix_kv_write_mask BEFORE calling set_kv_buffer().
        # set_kv_buffer() itself is called with the original signature.

    # --- Local attention with merged LSE output (section 4.5.2) ---
    num_heads = layer.tp_q_head_num
    # Reshape q to 3D once — _decode_att_m_fwd and _decode_softmax_reducev_fwd
    # both derive (batch, head_num) from q.shape[0:2] (decode_attention.py:596),
    # so q MUST be [B, num_heads, head_dim] for both stages.
    q_3d = q.view(-1, num_heads, layer.qk_head_dim)
    B = q_3d.shape[0]
    o = torch.empty(B, num_heads * layer.v_head_dim, dtype=q.dtype, device=q.device)
    # Allocate merged LSE buffer — filled by modified stage 2 kernel
    output_lse = torch.empty(B, num_heads, dtype=torch.float32, device=q.device)

    # kv_indptr / kv_indices are built from helix_local_kv_lens and the
    # compact-local req_to_token (section 4.4f), so the kernel sees only
    # this KVP rank's KV shard.

    # Stage 1: per-split partial attention
    _decode_att_m_fwd(
        q_3d,
        forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
        forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
        self.forward_metadata.attn_logits,
        self.forward_metadata.attn_lse,
        self.forward_metadata.kv_indptr,
        self.forward_metadata.kv_indices,
        self.forward_metadata.num_kv_splits,
        ...
    )

    # Stage 2: merge intra-GPU splits → partial output + merged LSE
    # Uses modified _decode_softmax_reducev_fwd with output_lse parameter
    # (see section 4.5.2 kernel change).
    # q_3d is passed (not raw q) so that q.shape[0:2] gives (B, num_heads).
    _decode_softmax_reducev_fwd(
        self.forward_metadata.attn_logits,
        self.forward_metadata.attn_lse,
        q_3d, o.view(-1, num_heads, layer.v_head_dim),
        forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
        self.forward_metadata.kv_indptr,
        self.forward_metadata.num_kv_splits,
        self.max_kv_splits,  # from TritonAttnBackend, not ForwardMetadata (triton_backend.py:1041)
        output_lse=output_lse,  # merged LSE written here
    )

    # Return partial output [B, Q/TPA, Hsz] + merged LSE [B, Q/TPA]
    # Reshape from flat [B, num_heads * v_head_dim] to 3D [B, num_heads, v_head_dim]
    # to match helix_kvp_alltoall_and_merge() input contract (section 4.5.1).
    return o.view(B, num_heads, layer.v_head_dim), output_lse
```

**FlashInfer backend** (`flashinfer_backend.py`) — **future work, not in initial bring-up:**

> FlashInfer is gated out of initial Helix (section 4.1). This sketch shows the integration once decode LSE return is available (see section 4.5.2 design note).

```python
def forward_helix_partial_decode(self, q, k, v, layer, forward_batch, save_kv_cache):
    """Helix decode: save KV (filtered), compute local attention + LSE.
    Requires FlashInfer decode wrapper with forward_return_lse() support."""

    # --- KV save with write-mask filtering ---
    if save_kv_cache:
        helix_filtered_set_kv_buffer(
            forward_batch, layer, forward_batch.out_cache_loc, k, v
        )

    # --- Local attention via FlashInfer wrapper (requires LSE return) ---
    # The wrapper's paged_kv_indptr / paged_kv_indices are built from
    # helix_local_kv_lens and compact-local req_to_token.
    o, lse = self.decode_wrapper.forward_return_lse(
        q.contiguous().view(-1, layer.tp_q_head_num, layer.qk_head_dim),
        forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id),
        ...
    )

    return o, lse
```

**RadixAttention dispatch** (`radix_attention.py`):

`RadixAttention.forward()` dispatches to the backend. For Helix, a new `forward_helix_partial()` method on `RadixAttention` calls the backend's `forward_helix_partial_decode`:
```python
class RadixAttention:
    def forward_helix_partial(self, q, k, v, forward_batch, save_kv_cache=True):
        """Called from LlamaAttention.forward() in Helix mode (section 4.7.1).
        Returns (partial_output, lse) for All-to-All merge."""
        return forward_batch.attn_backend.forward_helix_partial_decode(
            q, k, v, self, forward_batch, save_kv_cache
        )
```

**Call chain summary (decode, Triton backend — initial bring-up):**
1. `LlamaAttention.forward()` → `self.attn.forward_helix_partial(q, k, v, forward_batch)` (section 4.7.1)
2. `RadixAttention.forward_helix_partial()` → `forward_batch.attn_backend.forward_helix_partial_decode(...)` (above)
3. `TritonAttnBackend.forward_helix_partial_decode()` → `helix_filtered_set_kv_buffer(...)` → `set_kv_buffer()` (section 4.4c)
4. Same backend method → `_decode_att_m_fwd(...)` (stage 1) + modified `_decode_softmax_reducev_fwd(...)` (stage 2 with `output_lse`) → returns `(partial_out, merged_lse)` with shapes `[B, Q/TPA, Hsz]`, `[B, Q/TPA]`
5. Back in `LlamaAttention.forward()` → `helix_kvp_alltoall_and_merge(partial_out, merged_lse, ...)` (section 4.5.1)

### 4.8 CUDA Graphs

**File:** `python/sglang/srt/model_executor/cuda_graph_runner.py`

CUDA graphs with Helix add complexity because graph capture needs to include NCCL All-to-All and All-Reduce operations.

**Approach for initial implementation: Disable CUDA graphs when Helix is enabled.**

```python
# In model_runner.py:
if server_args.is_helix_enabled:
    self.cuda_graph_max_bs = 0  # Disable CUDA graphs
```

**Future optimization:** NCCL operations can be captured in CUDA graphs on CUDA 12+. This would require:
1. Capturing All-to-All in the graph (fixed tensor sizes, fixed group)
2. Ensuring buffer pointers don't change between capture and replay
3. Testing with NCCL graph capture APIs

---

## 5. Validation Plan

### 5.1 Unit Tests: All-to-All + Merge Correctness

**Test:** Verify that `helix_kvp_alltoall_and_merge()` produces bit-identical results to single-GPU full attention.

```
Setup:
- Generate random Q [B, Q_total, Hsz] and KV cache [S, K_total, Hsz]
- Run full attention on single GPU → reference output

Helix simulation:
- Split KV cache into KVP shards along sequence dimension
- On each "GPU", compute partial attention + LSE for its KV shard
- Call helix_kvp_alltoall_and_merge() to exchange and merge
- Compare merged output to reference output

Pass criteria: max absolute difference < 1e-3 (fp16) or < 1e-5 (fp32)
```

This can be tested with `torch.distributed` on a single node with multiple processes, or simulated in-process by manually splitting tensors and calling the merge function without actual communication.

### 5.2 Unit Tests: KV Cache Round-Robin Assignment

**Test:** Verify `get_kvp_rank_for_token()` distributes tokens correctly.

```
For kvp_size=4, chunk_size=16:
  Tokens 0-15   → rank 0
  Tokens 16-31  → rank 1
  Tokens 32-47  → rank 2
  Tokens 48-63  → rank 3
  Tokens 64-79  → rank 0 (wraps)
  ...

Verify: all tokens assigned, no gaps, balanced distribution
```

### 5.3 Unit Tests: Weight Loading Sharding

**Test:** Verify QKV weights are correctly replicated across KVP and sharded by TPA.

```
Setup: 4 GPUs, KVP=2, TPA=2
  GPU 0 (kvp=0, tpa=0): QKV shard 0
  GPU 1 (kvp=0, tpa=1): QKV shard 1
  GPU 2 (kvp=1, tpa=0): QKV shard 0  ← must match GPU 0
  GPU 3 (kvp=1, tpa=1): QKV shard 1  ← must match GPU 1

  GPU 0 o_proj shard 0 ≠ GPU 1 o_proj shard 1 ≠ GPU 2 shard 2 ≠ GPU 3 shard 3

Verify: torch.equal(gpu0.qkv_proj.weight, gpu2.qkv_proj.weight) == True
Verify: torch.equal(gpu0.o_proj.weight, gpu1.o_proj.weight) == False
```

### 5.4 Integration Test: Single-Layer Correctness

**Test:** Run one LlamaDecoderLayer with Helix and compare to standard TP.

```
Setup:
- 8 GPUs
- Standard TP=8: reference output
- Helix KVP=2, TPA=4: test output

For both prefill and decode:
  Same input hidden_states, same weights (loaded with respective sharding)
  Compare layer output tensors

Pass criteria: max absolute difference < 1e-2 (accumulated floating-point differences
from different computation order are expected but should be small)
```

### 5.5 Integration Test: End-to-End Generation

**Test:** Full model generation with Helix produces correct text.

```
Setup:
- Llama-8B (8 KV heads) on 8 GPUs
- Config A: Standard TP=8 (baseline)
- Config B: Helix KVP=2, TPA=4

Test:
  1. Same prompt → compare generated tokens (should match within sampling tolerance)
  2. With temperature=0 (greedy): tokens must be identical
  3. Compare logits at each step: max absolute difference < threshold

Prompts to test:
  - Short context (< 1K tokens)
  - Medium context (~ 10K tokens)
  - Long context (~ 100K tokens, if feasible)
```

### 5.6 Integration Test: Variable Batch Sizes

**Test:** Verify correctness across different batch sizes, especially edge cases.

```
Batch sizes to test: 1, 2, 7, 16, 32, 64
  - bs=1: Minimum batch
  - bs=7: Non-power-of-2
  - bs=64: Large batch

For each: compare greedy decode output to standard TP baseline
```

### 5.7 Multi-GPU Communication Tests

**Test:** Verify process groups are correctly formed and communication works.

```
Setup: 8 GPUs, KVP=4, TPA=2

Verify:
  1. TPA group sizes: 4 groups of 2 GPUs each
     (one group per kvp_rank; each has TPA=2 members)
  2. KVP group sizes: 2 groups of 4 GPUs each
     (one group per tpa_rank; each has KVP=4 members)
  3. All-to-All across KVP group: send/recv correct data
  4. All-Reduce across TPF group (all 8 GPUs): correct sum
  5. Rank mapping is PP-partition-local (test with PP=2 if applicable)
  6. No deadlocks or hangs
```

### 5.8 KV Cache Consistency Test

**Test:** After a decode sequence, verify total KV across all KVP ranks equals the full KV cache.

```
Setup: KVP=4, decode 100 tokens

After decoding:
  - Gather KV cache from all 4 KVP ranks
  - Reassemble into full sequence order using round-robin mapping
  - Compare to single-GPU KV cache (run same model without Helix)

Pass criteria: KV values identical (no missing or duplicated tokens)
```

### 5.9 Performance Benchmarks (Post-Correctness)

Once correctness is established:

```
Benchmark matrix:
  Model: Llama-8B, Llama-70B (if GPUs available)
  Context: 4K, 32K, 128K, 1M tokens
  Batch: 1, 8, 32
  Config: TP=8 baseline vs Helix (KVP=2/4/8, TPA=4/2/1)

Metrics:
  - Token-to-Token Latency (TTL)
  - Throughput (tokens/sec/GPU)
  - GPU memory utilization per rank
  - Communication time breakdown (All-to-All vs All-Reduce)
```

### 5.10 Test Execution Order

1. **Unit tests (no multi-GPU required):** 5.1 (simulated merge), 5.2 (KV assignment)
2. **Weight loading tests (multi-process):** 5.3
3. **Communication tests (multi-GPU):** 5.7
4. **Single-layer integration (multi-GPU):** 5.4
5. **KV cache consistency (multi-GPU):** 5.8
6. **End-to-end generation (multi-GPU):** 5.5, 5.6
7. **Performance benchmarks:** 5.9
