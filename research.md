# SGLang Attention & FFN Implementation Research (Decode Path)

## Table of Contents

1. [End-to-End Decode Flow](#1-end-to-end-decode-flow)
2. [Attention Architecture](#2-attention-architecture)
3. [Attention Backends](#3-attention-backends)
4. [Decode Attention Kernels](#4-decode-attention-kernels)
5. [KV Cache Management](#5-kv-cache-management)
6. [FFN / MLP Architecture](#6-ffn--mlp-architecture)
7. [Mixture of Experts (MoE)](#7-mixture-of-experts-moe)
8. [Quantization in FFN & Attention](#8-quantization-in-ffn--attention)
9. [CUDA Graph Optimization](#9-cuda-graph-optimization)
10. [Key Data Structures](#10-key-data-structures)

---

## 1. End-to-End Decode Flow

### 1.1 Scheduling → Forward Pass Pipeline

The decode step starts in the scheduler and flows through the model runner to produce logits:

```
Scheduler                          ModelRunner                         Model (Llama)
────────                          ───────────                         ─────────────
get_next_batch_to_run()
  │
  ├─ update_running_batch()
  │    ├─ filter_batch()
  │    ├─ check_decode_mem()
  │    └─ prepare_for_decode()
  │         ├─ input_ids ← output_ids (prev sampled token)
  │         ├─ seq_lens += 1
  │         └─ alloc_for_decode(token_per_req=1)
  │
  └─ get_model_worker_batch()
       │
       └─ ForwardBatch.init_new() ──────►  forward_decode()
            ├─ positions = clamp(seq_lens-1)    │
            └─ move to GPU                      ├─ attn_backend.init_forward_metadata()
                                                └─ model.forward(input_ids, positions, ...)
                                                        │
                                                        ├─ embed_tokens(input_ids)
                                                        ├─ for each layer:
                                                        │   ├─ input_layernorm
                                                        │   ├─ self_attn(q,k,v → KV cache)
                                                        │   ├─ post_attention_layernorm
                                                        │   └─ mlp(gate_up → act → down)
                                                        ├─ final norm
                                                        └─ lm_head → logits [bs, vocab_size]
```

**Key files:**
- Scheduler: `python/sglang/srt/managers/scheduler.py` — `get_next_batch_to_run()` (line ~1875), `update_running_batch()` (line ~2203)
- Schedule batch: `python/sglang/srt/managers/schedule_batch.py` — `prepare_for_decode()` (line ~1948)
- Model runner: `python/sglang/srt/model_executor/model_runner.py` — `forward_decode()` (line ~2288), `_forward_raw()` (line ~2450)
- Forward batch: `python/sglang/srt/model_executor/forward_batch_info.py` — `ForwardBatch.init_new()` (line ~376)

### 1.2 Decode Batch Preparation

In `prepare_for_decode()` (`schedule_batch.py:1948`):

Note: With speculative decoding, this function returns early (`schedule_batch.py:1957`) before any of the steps below — the batch is prepared through spec-algorithm-specific paths instead.

For the normal (non-speculative) path:
- `input_ids` is set to `output_ids` (the tokens sampled in the previous step)
- `seq_lens += 1` (one new token per request)
- KV cache slots are allocated via `alloc_for_decode(token_per_req=1)`
- Each request's `kv_allocated_len` and `kv_committed_len` are incremented

Position computation in `ForwardBatch.init_new()` (`forward_batch_info.py:483`):
```python
if forward_batch.forward_mode.is_decode():
    positions = clamp_position(batch.seq_lens)  # = max(seq_lens - 1, 0)
```

### 1.3 ForwardMode Enum

**File:** `forward_batch_info.py:74`

```python
class ForwardMode(IntEnum):
    EXTEND = auto()           # Prefill with prefix caching
    DECODE = auto()           # Single-token autoregressive decode
    MIXED  = auto()           # Chunked prefill (both prefill + decode)
    IDLE   = auto()
    TARGET_VERIFY = auto()    # Speculative verify
    DRAFT_EXTEND  = auto()    # Draft model extend
    ...
```

The `is_decode()` check routes to `forward_decode()` throughout the stack.

### 1.4 Layer-by-Layer Execution (Llama Example)

**File:** `python/sglang/srt/models/llama.py`

**LlamaForCausalLM.forward()** (line ~503):
1. **Embedding**: `hidden_states = self.embed_tokens(input_ids)` — shape `[batch_size] → [batch_size, hidden_size]`
2. **Decoder loop** (line ~381): For each of 32 (or N) layers:
   - `input_layernorm(hidden_states, residual)` → RMSNorm
   - `self_attn(positions, hidden_states, forward_batch)` → attention + KV cache
   - `post_attention_layernorm(hidden_states, residual)` → RMSNorm
   - `mlp(hidden_states)` → FFN
3. **Final norm** (line ~399): `self.norm(hidden_states, residual)`
4. **LM head** (line ~536): `self.logits_processor(input_ids, hidden_states, self.lm_head, forward_batch)` → `[batch_size, vocab_size]`

---

## 2. Attention Architecture

### 2.1 RadixAttention Layer

**File:** `python/sglang/srt/layers/radix_attention.py` (lines 47-135)

`RadixAttention` is the main attention layer used across all models. Key attributes:
- `tp_q_head_num`, `tp_k_head_num`, `tp_v_head_num`: Tensor-parallel head counts
- `head_dim`, `qk_head_dim`, `v_head_dim`: Support for different Q/K vs V dimensions
- `scaling`: Softmax scale (typically `1/sqrt(head_dim)`)
- `logit_cap`: Optional tanh-based logit capping (used in Gemma-style models)
- `sliding_window_size`: Optional sliding window attention
- `layer_id`: Layer index for KV cache addressing

**Forward routing** (lines 99-135):
```python
def forward(self, q, k, v, forward_batch, save_kv_cache=True, **kwargs):
    if forward_batch.forward_mode.is_extend() and get_forward_context() is not None:
        # Special path for torch.compile / custom op contexts
        unified_attention_with_output(q, k, v, output, save_kv_cache, self.layer_id, ...)
        return output
    else:
        # All other modes (decode, extend, mixed, idle, etc.)
        # Decode/extend branching happens inside attn_backend.forward()
        return forward_batch.attn_backend.forward(q, k, v, self, forward_batch, ...)
```

### 2.2 Attention in Model Layers

**LlamaAttention** (`llama.py:218-242`):
```python
def forward(self, positions, hidden_states, forward_batch):
    qkv, _ = self.qkv_proj(hidden_states)          # [bs, hidden] → [bs, q_size + 2*kv_size]
    q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)  # q_size != kv_size for GQA/MQA
    q, k = self.rotary_emb(positions, q, k)         # Apply RoPE
    attn_output = self.attn(q, k, v, forward_batch) # RadixAttention
    output, _ = self.o_proj(attn_output)             # Output projection
    return output
```

**RoPE (Rotary Position Embedding):**
- Uses absolute positions (for decode: `positions = [seq_len_i - 1]`)
- Rotates Q and K vectors: `angle = position / (base^(2d/d_model))`
- Enables extrapolation to longer sequences

### 2.3 Base Attention Backend Interface

**File:** `python/sglang/srt/layers/attention/base_attn_backend.py` (lines 17-170)

Key abstract methods:
- `init_forward_metadata()` — prepare per-batch metadata
- `forward_decode(q, k, v, layer, forward_batch)` — decode attention
- `forward_extend(q, k, v, layer, forward_batch)` — prefill attention
- `forward_mixed(q, k, v, layer, forward_batch)` — chunked prefill

The base `forward()` method (lines 79-121) dispatches based on `forward_mode`: IDLE returns an empty tensor, DECODE calls `forward_decode`, MIXED on NPU calls `forward_mixed`, and everything else (including EXTEND and non-NPU MIXED) falls through to `forward_extend`.

---

## 3. Attention Backends

SGLang uses a registry-based backend system with 15+ attention backends.

**File:** `python/sglang/srt/layers/attention/attention_registry.py`

### 3.1 Backend Summary Table

| Backend | File | Primary Use | Key Features |
|---------|------|-------------|--------------|
| **FlashInfer** | `flashinfer_backend.py` | Default NVIDIA | Paged KV, sliding window, CUDA graphs |
| **Triton** | `triton_backend.py` | NVIDIA fallback | 2-stage decode, dynamic KV splits |
| **FlashAttention** | `flashattention_backend.py` | FA3/FA4 | High-performance full attention |
| **FlashMLA** | `flashmla_backend.py` | MLA models | Multi-head Latent Attention |
| **CUTLASS MLA** | `cutlass_mla_backend.py` | MLA models | Custom tile sizes |
| **TRT-LLM MHA** | `trtllm_mha_backend.py` | Enterprise | TensorRT-LLM MHA kernels |
| **TRT-LLM MLA** | `trtllm_mla_backend.py` | Enterprise | TensorRT-LLM MLA kernels |
| **Wave** | `wave_backend.py` | AMD (ROCm) | Wave-lang kernels, 2-phase |
| **AITER** | `aiter_backend.py` | AMD (ROCm) | MLA, FP8 KV, persistent cache |
| **NSA** | `nsa_backend.py` | DeepSeek V3.2 | Native Sparse Attention |
| **Double Sparsity** | `double_sparsity_backend.py` | Sparse models | Heavy token approximation |
| **Torch Native** | `torch_native_backend.py` | Fallback | Pure PyTorch |
| **Torch Flex** | `torch_flex_backend.py` | PyTorch 2.5+ | Flex Attention API |
| **Intel AMX** | `intel_amx_backend.py` | Intel CPU | AMX acceleration |
| **Intel XPU** | `xpu_backend.py` | Intel GPU | XPU support |
| **Hybrid** | `hybrid_attn_backend.py` | Mamba/Linear | MHA + linear attention mixing |

### 3.2 FlashInfer Backend (Default)

**File:** `python/sglang/srt/layers/attention/flashinfer_backend.py` (910 lines)

**Decode forward** (lines 862-898):
```python
def forward_decode(self, q, k, v, layer, forward_batch, save_kv_cache=True):
    decode_wrapper = self.forward_metadata.decode_wrappers[self._get_wrapper_idx(layer)]

    if k is not None and save_kv_cache:
        forward_batch.token_to_kv_pool.set_kv_buffer(
            layer, forward_batch.out_cache_loc, k, v,
            layer.k_scale, layer.v_scale
        )

    o = decode_wrapper.forward(
        q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
        forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id),
        sm_scale=layer.scaling,
        logits_soft_cap=layer.logit_cap,
        k_scale=layer.k_scale_float,
        v_scale=layer.v_scale_float,
    )
    return o.view(-1, layer.tp_q_head_num * layer.head_dim)
```

Uses `BatchDecodeWithPagedKVCacheWrapper` from the FlashInfer library. Supports dual-wrapper mode for sliding window or cross-attention dispatch via `_get_wrapper_idx()`.

**Metadata initialization** (lines 424-437): Uses `indices_updater_decode` to configure FlashInfer wrappers with KV page tables, sequence lengths, and optional split tile sizes.

### 3.3 Triton Backend

**File:** `python/sglang/srt/layers/attention/triton_backend.py` (1048 lines)

**Decode forward** (lines 997-1047):
```python
def forward_decode(self, q, k, v, layer, forward_batch, save_kv_cache=True, sinks=None):
    q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)
    o = torch.empty_like(q)  # or new_empty for different v_head_dim

    if save_kv_cache:
        forward_batch.token_to_kv_pool.set_kv_buffer(layer, forward_batch.out_cache_loc, k, v)

    # Select kv_indptr/kv_indices (full or sliding window)
    self.decode_attention_fwd(
        q, key_buffer, value_buffer, o,
        kv_indptr, kv_indices,
        attn_logits, attn_lse, num_kv_splits,
        max_kv_splits, layer.scaling,
        logit_cap=logit_cap, sinks=sinks,
    )
    return o
```

**Metadata initialization** (lines 234-437): Builds `kv_indptr` (cumulative sum of seq_lens), `kv_indices` (token pool locations), and computes `num_kv_splits` dynamically.

---

## 4. Decode Attention Kernels

### 4.1 Triton Two-Stage Decode Kernel

**File:** `python/sglang/srt/layers/attention/triton_ops/decode_attention.py` (777 lines)

The Triton decode kernel uses a two-stage architecture for efficient parallel reduction:

#### Stage 1: Attention Computation (lines 45-249)

```
Kernel: _fwd_kernel_stage1
Grid: (batch_size, num_heads, max_kv_splits)

For each (batch, head, split):
    1. Load Q[batch, head, :] from query tensor
    2. Loop over KV chunks assigned to this split:
        a. Load K[kv_indices[idx], head, :] from paged buffer
        b. Compute QK = Q @ K^T * sm_scale
        c. (Optional) Apply logit_cap: QK = logit_cap * tanh(QK / logit_cap)
        d. (Optional) Apply temperature scaling (XAI)
        e. Update running max and accumulator: acc += exp(QK - max) * V
    3. Output: Att_Out[batch, head, split, :] = acc / sum
    4. Output: Att_Lse[batch, head, split] = log(sum) + max
```

Key parameters:
- `kv_indptr`: Pointer to start of each sequence's KV in the paged buffer
- `kv_indices`: Actual page/token indices for paged KV cache access
- `num_kv_splits`: How many chunks to split the KV sequence for this query
- `BLOCK_N`: Block size for KV chunks (64 for normal, 32 for grouped/GQA)

#### Stage 2: Softmax Reduction (lines 515-630)

```
Kernel: _fwd_kernel_stage2
Grid: (batch_size, num_heads)

For each (batch, head):
    Merge partial results from all splits:
        acc = old_acc * exp(old_lse - new_lse) + new_acc
        lse = log(sum_of_exps)
    Output: O[batch, head, :] = final_acc / final_sum
```

#### MHA vs GQA Dispatch (lines 633-777)

- **Normal (MHA):** `decode_attention_fwd_normal()` — one-to-one Q-head to KV-head mapping
- **Grouped (GQA/MQA/MLA):** `decode_attention_fwd_grouped()` — multiple Q heads share KV heads

### 4.2 Dynamic KV Splits

**File:** `triton_backend.py:182-232`

KV splits control parallelism within each query's attention computation:

```python
def get_num_kv_splits(num_kv_splits, seq_lens):
    # Also handles speculative decoding: num_token may be topk * num_seq
    num_group = num_token // num_seq

    if (self.static_kv_splits or self.device_core_count <= 0) and not self.enable_deterministic:
        num_kv_splits.fill_(self.max_kv_splits)     # Fixed splits (legacy non-deterministic)
    elif self.split_tile_size is not None and self.enable_deterministic:
        # For speculative/grouped cases, expand seq_lens to match num_token
        if num_group > 1:
            expanded_seq_lens = seq_lens.repeat_interleave(num_group)
        else:
            expanded_seq_lens = seq_lens
        num_kv_splits = ceil(expanded_seq_lens / split_tile_size)  # Deterministic with fixed tile
    else:
        # Dynamic: Triton kernel balances load across SM cores using
        # seq_lens, num_head, num_kv_head, device_core_count, and max_kv_splits
        get_num_kv_splits_triton(num_kv_splits, seq_lens, num_seq, num_group, ...)
```

### 4.3 Special Decode Optimizations

| Optimization | Description | Location |
|---|---|---|
| **Logit capping** | `tanh(QK/cap) * cap` bounds attention scores | `decode_attention.py:129` |
| **Temperature scaling (XAI)** | Position-dependent temperature for XAI models | `decode_attention.py:89-93, 132-133` |
| **Sink tokens** | Extra contribution from designated "sink" tokens | `decode_attention.py:574-576` |
| **Sliding window** | Separate KV indices for windowed layers | `triton_backend.py:1024-1029` |
| **Speculative decode** | Pre-computed KV indices from `spec_info` | `triton_backend.py:246-283` |
| **Quantized KV** | FP8 KV cache with scale factors | `flashinfer_backend.py:888-896` |

---

## 5. KV Cache Management

### 5.1 Memory Pool Architecture

**File:** `python/sglang/srt/mem_cache/memory_pool.py` (2025 lines)

Two-level mapping:

```
Request → Token locations → KV buffers

ReqToTokenPool                     TokenToKVPool (MHATokenToKVPool)
├── req_to_token: [max_req,        ├── k_buffer: [num_layers] each [pool_size, head_num, head_dim]
│                  max_ctx_len]     ├── v_buffer: [num_layers] each [pool_size, head_num, v_head_dim]
├── free_slots: List[int]          └── alloc/free via BaseTokenToKVPool
└── alloc(reqs) → req indices
```

### 5.2 KV Cache Store During Decode

When a new token is decoded, its K and V vectors are stored:

```python
# In forward_decode (triton_backend.py:1019-1022):
if save_kv_cache:
    forward_batch.token_to_kv_pool.set_kv_buffer(
        layer, forward_batch.out_cache_loc, k, v
    )
```

**`set_kv_buffer` implementation** (`memory_pool.py:951-988`):
```python
def set_kv_buffer(layer, loc, cache_k, cache_v, k_scale=None, v_scale=None):
    _set_kv_buffer_impl(cache_k, cache_v,
        self.k_buffer[layer_id], self.v_buffer[layer_id],
        loc, row_dim, store_dtype, ...)
```

For CUDA with matching K/V dimensions, this uses a JIT `store_cache` kernel (`memory_pool.py:102-110`):
```python
store_cache(k.view(-1, row_dim), v.view(-1, row_dim),
            k_cache.view(-1, row_dim), v_cache.view(-1, row_dim),
            indices, row_bytes=row_bytes)
```

### 5.3 KV Cache Retrieval During Decode

The attention kernel reads cached KV via:
```python
key_buffer = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
value_buffer = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)
```

The `kv_indices` tensor maps from logical sequence positions to physical buffer locations, enabling paged access. This is built during metadata initialization using `req_to_token` mappings.

---

## 6. FFN / MLP Architecture

### 6.1 Dense FFN (LlamaMLP)

**File:** `python/sglang/srt/models/llama.py` (lines 65-116)

```python
class LlamaMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, ...):
        self.gate_up_proj = MergedColumnParallelLinear(    # Fused gate + up
            hidden_size, [intermediate_size] * 2)
        self.down_proj = RowParallelLinear(                # Down projection
            intermediate_size, hidden_size)
        self.act_fn = SiluAndMul()                         # Fused SiLU + multiply

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)     # [bs, hidden] → [bs, 2*intermediate]
        x = self.act_fn(gate_up)               # SiLU(gate) * up → [bs, intermediate]
        x, _ = self.down_proj(x)               # [bs, intermediate] → [bs, hidden]
        return x
```

**Architecture:** SwiGLU variant
```
x ─┬─ gate_proj ─→ SiLU ─┐
   │                      ├─ element-wise multiply ─→ down_proj ─→ output
   └─ up_proj ────────────┘
```

The gate and up projections are fused into a single `MergedColumnParallelLinear` GEMM for efficiency.

### 6.2 Linear Layer Implementations

**File:** `python/sglang/srt/layers/linear.py` (1602 lines)

| Class | Lines | Parallelism | Used For |
|-------|-------|-------------|----------|
| `ReplicatedLinear` | 182-274 | None (replicated) | MoE gate projection |
| `ColumnParallelLinear` | 277-467 | Column-parallel (shard output dim) | Individual projections |
| `MergedColumnParallelLinear` | 469-838 | Column-parallel (fused) | gate_up_proj |
| `QKVParallelLinear` | 840-1277 | Column-parallel (QKV fused) | qkv_proj in attention |
| `RowParallelLinear` | 1280-1497 | Row-parallel (shard input dim, all-reduce) | down_proj, o_proj |

**Tensor parallelism pattern:**
- Column-parallel: Each GPU computes a slice of the output dimension
- Row-parallel: Each GPU computes with a slice of the input dimension, then all-reduce

**`RowParallelLinear.forward()` (lines 1460-1489):**
```python
def forward(self, input_, skip_all_reduce=False):
    output_parallel = self.quant_method.apply(self, input_)
    if self.reduce_results and not skip_all_reduce:
        output = tensor_model_parallel_all_reduce(output_parallel)
    return output, output_bias
```

### 6.3 Activation Functions

**File:** `python/sglang/srt/layers/activation.py` (384 lines)

**`SiluAndMul`** (lines 63-96) — Primary activation for SwiGLU:
```python
class SiluAndMul(MultiPlatformOp):
    def forward_native(self, x):
        d = x.shape[-1] // 2
        return F.silu(x[..., :d]) * x[..., d:]  # SiLU(gate) * up

    def forward_cuda(self, x):
        out = torch.empty(x.shape[:-1] + (d,), ...)
        sgl_kernel.silu_and_mul(x, out)  # Fused CUDA kernel
        return out
```

Backend-specific dispatching:
- **CUDA/XPU:** `sgl_kernel.silu_and_mul()` — fused kernel from sgl-kernel
- **CPU:** `torch.ops.sgl_kernel.silu_and_mul_cpu()` — Intel AMX accelerated
- **NPU:** `torch_npu.npu_swiglu()` — Ascend NPU kernel

**`GeluAndMul`** (lines 99-143) — Used in some models (GPT-J, etc.):
- Supports "tanh" and "none" approximations
- Similar multi-backend dispatch

### 6.4 Normalization

**RMSNorm** — Used before both attention and FFN:
```python
# In LlamaDecoderLayer:
self.input_layernorm = RMSNorm(hidden_size, eps)          # Before attention
self.post_attention_layernorm = RMSNorm(hidden_size, eps)  # Before FFN
```

Computation: `output = x / sqrt(mean(x^2) + eps) * weight`

The fused variant handles residual connections in-place:
```python
hidden_states, residual = self.input_layernorm(hidden_states, residual)
# Combines: residual += hidden_states; hidden_states = norm(residual)
```

---

## 7. Mixture of Experts (MoE)

### 7.1 FusedMoE Layer

**File:** `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` (1416 lines)

**`FusedMoE` class** (lines 156-1130):
```python
class FusedMoE(nn.Module):
    def forward(self, hidden_states, topk_output):
        if is_in_piecewise_cuda_graph():
            # CUDA graph mode — special dispatch
            return moe_forward_piecewise_cuda_graph_impl(...)
        return self.forward_impl(hidden_states, topk_output)

    def forward_impl(self, hidden_states, topk_output):
        # Core: route tokens → experts → fused GEMM
        run_moe_core() → quant_method.apply()
```

### 7.2 Routing (Top-K Selection)

**File:** `python/sglang/srt/layers/moe/topk.py`

```python
class TopK:
    # Routes tokens to top-k experts
    # Uses moe_fused_gate kernel from sgl_kernel
    # Supports: topk_softmax, topk_sigmoid, aiter_biased_grouped_topk
```

Output formats: `StandardTopKOutput`, `TritonKernelTopKOutput`, `BypassedTopKOutput`

### 7.3 Fused MoE Triton Kernel

**File:** `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py`

Implements a fused pipeline:
1. **Token sorting**: Map tokens to assigned experts
2. **GEMM1**: `input × w1` (gate+up projection, quantization-aware)
3. **Fused SiLU + MUL**: Activation
4. **GEMM2**: `activated × w2` (down projection)
5. **Weighted accumulation**: Multiply by router weights, sum across experts

Supports quantization formats: FP8 W8A8, INT8 W8A16, INT4 W4A16, with per-block/per-token scaling.

### 7.4 CUTLASS MoE Implementations

**FP8:** `python/sglang/srt/layers/moe/cutlass_moe.py`
```python
def cutlass_fused_experts_fp8(a, w1_q, w2_q, w1_scale, w2_scale, topk_weights, topk_ids):
    # 1. Per-token quantize activations → FP8
    # 2. fp8_blockwise_scaled_grouped_mm() for GEMM1
    # 3. Fused SiLU + MUL
    # 4. fp8_blockwise_scaled_grouped_mm() for GEMM2
```

**W4A8:** `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py`
```python
def cutlass_w4a8_moe(a, w1_q, w2_q, ...):
    # INT4 weights (packed as INT8) + FP8 activations
    # Uses cutlass_w4a8_moe_mm() from sgl_kernel
```

### 7.5 Expert Parallel MoE (DeepEP)

**File:** `python/sglang/srt/layers/moe/ep_moe/layer.py`

`DeepEPMoE` extends `FusedMoE` for expert parallelism across GPUs:
- Deep GEMM wrapper for optimized computation
- All-to-all (A2A) communication for token dispatch
- Low-latency mode via DeepEP framework
- Supports FP8 W8A8 and W4A8

---

## 8. Quantization in FFN & Attention

### 8.1 FP8 Quantization

**File:** `python/sglang/srt/layers/quantization/fp8.py` (1674 lines)

**Configuration (`Fp8Config`):**
- `activation_scheme`: "dynamic" (per-token) or "static" (pre-computed)
- `weight_block_size`: Block-wise quantization granularity
- `use_mxfp8`: MXFP8 format (block_size=[1,32])

**For linear layers:** `Fp8LinearMethod`
- Dynamic per-token activation quantization
- Static per-block weight quantization
- Supported on all dense linear layers (gate_up, down, qkv, o)

**For MoE layers:** `Fp8MoEMethod`
- Three quantization info types: `TritonMoeQuantInfo`, `DeepGemmMoeQuantInfo`, `FlashInferTrtllmFp8MoeQuantInfo`
- Per-token group quantization for activations
- Per-block weight quantization for expert weights

### 8.2 INT8 Quantization

**Files:**
- `python/sglang/srt/layers/quantization/int8_kernel.py` — Per-token and per-group INT8 quantization
- `python/sglang/srt/layers/quantization/blockwise_int8.py` — Block-wise INT8

### 8.3 Other Quantization Methods

Supported via the linear layer `quant_method` abstraction:
- GPTQ (INT4/INT8 with group quantization)
- AWQ (Activation-aware Weight Quantization)
- Marlin (fast INT4 dequantization)
- GGUF (llama.cpp format)
- ModelOpt FP8/FP4
- CompressedTensors
- 15+ other methods (see `linear.py:54-74`)

### 8.4 Quantized KV Cache

In FlashInfer backend:
```python
o = decode_wrapper.forward(
    q, kv_buffer,
    k_scale=layer.k_scale_float,  # FP8 KV dequantization
    v_scale=layer.v_scale_float,
)
```

---

## 9. CUDA Graph Optimization

### 9.1 Graph Capture

**File:** `python/sglang/srt/model_executor/cuda_graph_runner.py`

Captured for discrete batch sizes (1, 2, 4, 8, 16, ...):
```python
self.capture_bs = get_batch_sizes_to_capture(model_runner, num_tokens_per_bs=1)
# For decode: 1 token per request
```

Capture process (lines 556-738):
1. Create ForwardBatch with fill values
2. Initialize attention backend: `attn_backend.init_forward_metadata_capture_cuda_graph()`
3. Run model forward to record all GPU operations
4. Store graph: `self.graphs[bs] = graph`

### 9.2 Graph Replay (Decode Fast Path)

**File:** `model_runner.py:2463-2475`

```python
can_run_graph = (forward_batch.forward_mode.is_cuda_graph()
                 and self.graph_runner.can_run(forward_batch))
if can_run_graph:
    ret = self.graph_runner.replay(forward_batch)  # ~100x faster than normal forward
```

**Replay process** (lines 776-889):
1. Find nearest captured batch size: `bs = ceil_to_captured(raw_bs)`
2. Copy input tensors into pre-allocated graph buffers
3. Pad remaining slots with fill values
4. `self.graphs[bs].replay()` — zero CPU overhead
5. Slice output to actual batch size: `logits[:raw_bs]`

### 9.3 Piecewise CUDA Graphs for MoE

MoE layers use piecewise CUDA graph capture to handle dynamic token-expert routing:
```python
# In FusedMoE.forward():
if is_in_piecewise_cuda_graph():
    return moe_forward_piecewise_cuda_graph_impl(
        hidden_states, topk_weights, topk_ids, router_logits, self.layer_id)
```

---

## 10. Key Data Structures

### 10.1 ForwardBatch

```
ForwardBatch
├── forward_mode: ForwardMode           # DECODE for decode phase
├── batch_size: int                     # Number of requests
├── input_ids: Tensor [bs]             # Last sampled tokens
├── positions: Tensor [bs]             # Absolute positions (seq_len - 1)
├── seq_lens: Tensor [bs]             # Current sequence lengths
├── seq_lens_sum: int                  # Total tokens across batch
├── req_pool_indices: Tensor [bs]     # Request → pool slot mapping
├── out_cache_loc: Tensor [bs]        # Where to store new KV
├── req_to_token_pool: ReqToTokenPool  # Request → token location mapping
├── token_to_kv_pool: TokenToKVPool    # KV cache buffers
├── attn_backend: AttentionBackend     # Selected attention backend
├── sampling_info: SamplingBatchInfo   # Temperature, top-p, etc.
└── spec_info: Optional[SpecInput]     # Speculative decoding info
```

### 10.2 Decode Metadata (Triton)

```
DecodeForwardMetadata
├── kv_indptr: Tensor [bs+1]          # Cumulative KV lengths
├── kv_indices: Tensor [total_kv]     # Physical KV buffer locations
├── attn_logits: Tensor [bs, heads, max_splits, v_dim]  # Stage 1 output
├── attn_lse: Tensor [bs, heads, max_splits]             # Log-sum-exp
├── num_kv_splits: Tensor [bs]        # Splits per sequence
├── window_kv_indptr: Optional        # For sliding window
└── window_kv_indices: Optional       # For sliding window
```

### 10.3 Decode Metadata (FlashInfer)

```
DecodeMetadata
└── decode_wrappers: List[BatchDecodeWithPagedKVCacheWrapper]
    ├── [0]: Full attention wrapper (or sliding window)
    └── [1]: Optional second wrapper (full/cross-attention)
```

### 10.4 Memory Pools

```
ReqToTokenPool
├── req_to_token: Tensor [max_req, max_ctx_len]  # Request → token locations
└── free_slots: List[int]

MHATokenToKVPool
├── k_buffer: List[Tensor]  # [num_layers] × [pool_size, head_num, head_dim]
├── v_buffer: List[Tensor]  # [num_layers] × [pool_size, head_num, v_head_dim]
└── store_dtype: dtype       # Storage precision (may differ from compute)
```

---

## Summary of Decode-Specific Characteristics

1. **Single token per request**: During normal (non-speculative) decode, each request contributes exactly 1 new token. Batch shape is `[batch_size]` not `[batch_size, seq_len]`. With speculative decoding, the regular decode path is bypassed entirely — `prepare_for_decode()` returns early (`schedule_batch.py:1957`) and the batch is prepared through spec-algorithm-specific paths (e.g., Eagle v1/v2) rather than a single function.

2. **Memory-bound, not compute-bound**: The main bottleneck is KV cache reads. Attention must load all cached K,V for each new query token, making it proportional to sequence length.

3. **Two-stage attention**: The Triton backend splits long KV sequences across GPU cores (stage 1), then merges partial results (stage 2), maximizing parallelism.

4. **Paged KV cache**: Physical KV buffer locations are decoupled from logical sequence positions via `kv_indices`, enabling memory-efficient sharing and eviction (radix tree).

5. **CUDA graphs**: Pre-recorded GPU execution eliminates CPU overhead. Padded batches with sliced outputs handle variable batch sizes.

6. **Fused operations**: `MergedColumnParallelLinear` (gate+up), `SiluAndMul` (activation+multiply), and fused MoE kernels minimize memory round-trips.

7. **Backend flexibility**: The registry system allows swapping attention implementations (FlashInfer, Triton, TRT-LLM, etc.) without changing model code.
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTUwODY1MzcwMV19
-->