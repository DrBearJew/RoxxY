# Packed16 FA V-Layout Research

Status: **research only** — no patches planned until all cells filled.

## 1. V cache physical storage

**Allocation** (`llama-kv-cache.cpp:269`):
```cpp
ggml_tensor * v = ggml_new_tensor_3d(ctx, type_v_layer, n_embd_v_gqa, kv_size, n_stream);
// ne = [D_total, K, streams]
// Physical: D-contiguous rows, each row = ggml_row_size(type, D_total) bytes
```

This is true for all V types: f16, q4_0, q8_0, tbq4_0. Same layout regardless of `v_trans`.

**v_trans origin** (`llama-model.cpp:2090`):
```cpp
!cparams.flash_attn
```

| FA state | v_trans | meaning |
|----------|---------|---------|
| enabled | false | V exposed D-contiguous to graph |
| disabled | true | V exposed as [K,H,D] transposed view |

---

## 2. get_v() output layout matrix

Three branches in `get_v()` (`llama-kv-cache.cpp:1318-1382`):

| Branch | Condition | ne shape | nb[0] | Layout name |
|--------|-----------|----------|-------|-------------|
| TBQ | `v->type == TBQ3_0 \|\| TBQ4_0` | `[D_total, K, ns]` | `sizeof(type)` | FA_DKHB (3D) |
| v_trans=false | FA enabled | `[D_head, H_kv, K, ns]` | `sizeof(type)` | FA_DKHB |
| v_trans=true | FA disabled | `[K, H_kv, D_head, ns]` | `row_size(Kv × D_head)` | NATIVE_KHDB (bad nb0) |

**Key detail**: v_trans=true branch returns `ne=[K, H, D]` but `nb[0] = row_size(kv_size*D_head)` — this is NOT `sizeof(type)`. FA backends that assert `nb[0] == ts` will reject this.

**For quantized V (q4_0):**
```
block_q4_0 = { ggml_half d; uint8_t qs[16]; }  // 2 + 16 = 18 bytes
QK4_0 = 32 quantized values per block
nb per block = sizeof(block_q4_0) = 18 bytes
```
- v_trans=false: `ne=[D_head, H, K]` where D_head = n_embd_head_v blocks × QK4_0 scalars
  `nb[0] = 18` (block stride), `nb[1] = 18 * n_embd_head_v/blocks` (head stride)
  ✅ block-granular contiguous — each block object is contiguous, D_head scalars split across blocks
- v_trans=true: `ne=[K, H, D_head]` with `nb[0] = ggml_row_size(v->type, kv_size * n_embd_head_v)`
  This is ~ kv_size × D_head_blocks × 18 — NOT 18. ❌ Not FA-safe.

**For quantized V (q8_0):**
```
block_q8_0 = { ggml_half d; int8_t qs[32]; }  // 2 + 32 = 34 bytes
QK8_0 = 32 quantized values per block
nb per block = sizeof(block_q8_0) = 34 bytes
```
- Same pattern as q4_0: nb[0] = 34 for v_trans=false, nb[0] = huge for v_trans=true

**Address formula for d=0,31,32,255** (q4_0, D=256, v_trans=false):
```
d=0:   block = 0, offset_in_block = 0 → V + 0*nb1 + 0*18
          nibble = qs[0] & 0x0f
d=31:  block = 0, offset_in_block = 31 → V + 0*nb1 + 31?...
          Wait: nb[0]=18 (block granularity), not scalar granularity.
          d=31: same block (0), accessed via same block_q4_0 struct.
          Actual scalar: block_q4_0 at nb[0]*0, then decode scalar 31 from nibbles.
d=32:  block = 1, offset_in_block = 0 → V + 0*nb1 + 1*18
d=255: block = 7 (256/32-1), offset_in_block = 31 → V + 0*nb1 + 7*18
```
Key: FA backends that do per-scalar V reads need to go through block decode. FA backends
that do per-block reads (DOT4-MMQ with block granularity) are fine as-is.

**For TBQ**: always `ne=[D_total, K, ns]` (3D), `nb[0]=sizeof(type)`. Skip permute chain.

---

## 3. build_attn_mha() transform chain

Full sequence in `llama-graph.cpp:1986-2146`:

```
Step 1: TBQ reshape
  If v_is_tbq: reshape to 4D [D_head, H_kv, K, ns]
  Layout: FA_DKHB

Step 2: Global permute
  v = ggml_permute(ctx0, v, 0, 2, 1, 3)
  ∀ paths: [D|K, K|D, H, ns]

Step 3: FA path decision
  For use_flash_attn && v_trans && !pwmma_forced:
    v_for_fa = ggml_transpose(ctx0, v)   → restores FA_DKHB
  For use_flash_attn && (!v_trans || pwmma_forced):
    v_for_fa = v                          → keeps permuted layout
```

**Layouts at each step** (v_trans=true, f16 V):

| Step | Operation | Shape | Layout | nb[0] | Why this exists | Who expects it |
|------|-----------|-------|--------|-------|-----------------|----------------|
| input | get_v() | [K, H, D] | NATIVE_KHDB | kv*D*ts | Cache stores D-contiguous rows; v_trans reshapes to [K,H,D] for non-FA consumers | Non-FA ggml_mul_mat path |
| step 2 | permute(0,2,1,3) | [K, D, H] | PERMUTED_KDHB | kv*D*ts | Swaps H↔D so FA backends see [*,D,H] — K and V need same dim ordering for QK+PV | ALL paths — done unconditionally |
| step 3 | transpose (non-PWMMA) | [D, H, K] | FA_DKHB | ts | Restores D-contiguous layout after the universal permute | VEC/TILE/MMA/WMMA/DOT4 — all require nb0=ts |
| step 3 | skip (PWMMA) | [K, D, H] | PERMUTED_KDHB | kv*D*ts | PWMMA has custom V loader for this layout → skip transpose to avoid extra ggml op | PWMMA only (PWMMA_V_LAYOUT_TRANS) |

**PWMMA inherits the PERMUTED_KDHB** layout because `ggml_permute` happens before the `pwmma_forced` check at step 3. The comment at line 2065 says "Skip the global permute for PWMMA+v_trans to keep native layout" but the code does NOT skip the permute — it still runs. The skip is on the transpose, not the permute. PWMMA's `PWMMA_V_LAYOUT_TRANS` loader handles PERMUTED_KDHB.

---

## 4. V consumer map

Only 3 call sites of `get_v()` in `llama-graph.cpp`:

| Line | Function | v_trans | Layout received |
|------|----------|---------|-----------------|
| 2378 | `build_attn_inp_k()` (self-attn) | per cache | FA_DKHB or NATIVE_KHDB |
| 2557 | `build_attn_inp_cross()` (cross-attn) | per cache | FA_DKHB or NATIVE_KHDB |
| 2645 | `build_attn_assist()` (MTP assistant) | src cache | FA_DKHB or NATIVE_KHDB |

All go through `build_attn_mha()`. No other consumers read V from cache.

**MTP assistant path** (line 2645): Uses source context's V directly. May need stream slicing for kv_unified=false.

---

## 5. Backend capability matrix

**V layout support per backend:**

| Backend | FA_DKHB (nb0=ts) | NATIVE_KHDB (bad nb0) | PERMUTED_KDHB | q4_0 V | q8_0 V | f16 V |
|---------|-------------------|----------------------|---------------|--------|--------|-------|
| CPU FA | ✅ (required) | ❌ | ❌ | ✅ (via block decode) | ✅ (via block decode) | ✅ |
| VEC | ✅ (assert) | ❌ | ❌ | ✅ | ✅ | ✅ |
| TILE | ✅ (assert) | ❌ | ❌ | no | no | ✅ |
| MMA f16 | ✅ (assert) | ❌ | ❌ | no | no | ✅ |
| WMMA f16 | ✅ (assert) | ❌ | ❌ | no | no | ✅ |
| DOT4 packed16 | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ |
| DOT4-MMQ | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ |
| PWMMA packed16 | ✅ (FA layout) | ❌ (loader uses TRANS via nb detection) | ✅ (TRANS layout) | ✅ | ✅ | ✅ |

**Key constraints:**

| Constraint | File:Line | What |
|-----------|-----------|------|
| VEC/TILE/MMA/WMMA nb0 check | `fattn-common.cuh:1573` | `GGML_ASSERT(V->nb[0] == ggml_element_size(V))` |
| Same check for TBQ packed V | `fattn-common.cuh:1654` | `GGML_ASSERT(V->nb[0] == ts)` |
| CPU FA nb0 check | `ops.cpp:8900` | `GGML_ASSERT(nbv0 == ggml_type_size(v->type))` |
| PWMMA V layout detection | `fattn-packed16-wmma-tile.cuh:356-371` | Heuristic: nb1>nb2 detection, then enum assignment |
| PWMMA FA loader | `fattn-packed16-wmma-tile.cuh:205` | `p + d*nb10` (D-contiguous read) |
| PWMMA TRANS loader | `fattn-packed16-wmma-tile.cuh:208` | `p + d*nb12 + k*nb11` (strided read) |
| PWMMA NATIVE_KDH loader | `fattn-packed16-wmma-tile.cuh:209-214` | `p + k*nb11 + d*nb10` variant |
| DOT4 V access | `fattn-dot4-q8q4.cuh:580` | `v_head + k*nb21` then `nb20` for block stride |
| DOT4-MMQ V access | `fattn-packed16-dot4-mmq.cuh:199` | `V + b*nb13 + hk*nb12 + k*nb11`, then `blk*nb10` for blocks |
| CUDA FA support | `ggml-cuda.cu:5259` | PWMMA override or `fattn.cu:3329` support probe |
| CPU FA support | `ggml-cpu.cpp:423` | Default `true` — no layout check |
| CPU FA compute | `ggml-cpu.c:2001` | No pre-check, assert at `ops.cpp:8900` |

**CPU FA is the dangerous backend**: it claims support for ALL FA ops (default: true),
then asserts `nbv0 == ts` at compute time. If scheduler routes v_trans V to CPU,
it crashes at the assert, not at a clean error.

**V access formulas per backend:**

| Backend | Formula | Source |
|---------|---------|--------|
| CPU FA | `v_row[d] = base + k_row*nb1 + d*nb0` | ops.cpp (nb0 must = ts) |
| VEC | `V + k*nb21` → `dequantize_V(ptr, ...)` | fattn-vec.cuh:425, nb21=row stride |
| DOT4 | `v_head + k*nb21 + d*nb20` | fattn-dot4-q8q4.cuh:580 |
| DOT4-MMQ | `pdmq_decode_v(V, nb10..nb13, k, d)` | fattn-packed16-dot4-mmq.cuh |
| PWMMA FA | `V + k*nb11 + d*nb10` | fattn-packed16-wmma-tile.cuh:91 |
| PWMMA TRANS | `V + d*nb12 + k*nb11` | fattn-packed16-wmma-tile.cuh:203 |

---

## 6. Scheduler routing matrix

**Support probe chain** (`ggml-cuda.cu:5259-5272`):
```
GGML_OP_FLASH_ATTN_EXT → ggml_cuda_flash_attn_ext_supported(device, op)
```

**`ggml_cuda_flash_attn_ext_supported()`** (`fattn.cu:3329`):
- Calls `ggml_cuda_get_best_fattn_kernel()`
- Returns `true` if kernel != BEST_FATTN_KERNEL_NONE
- Support probe runs in `GGML_CUDA_FATTN_SELECT_SUPPORT_PROBE` context

**Route enforcement** (for packed16 I32 K):
```
require_packed16_dot4_mmq: ABORT if !ggml_cuda_packed16_dot4_mmq_supported()
require_packed16_wmma:    ABORT if not selected (with nq==1 DOT4 fallback allowed)
```

**CPU fallback risk**: CPU FA accepts FLASH_ATTN_EXT but requires `nbv0 == ts`. If V reaches CPU with bad nb0, CPU FA will hit the assert. If V is reshaped between dispatch and execution, CPU may silently accept it.

---

## 7. Proposed V layout enum

**Target contract** — add to FLASH_ATTN_EXT op_params:

```cpp
enum ggml_fattn_v_layout {
    GGML_FATTN_V_LAYOUT_FA_DKHB     = 0,  // [D, K, H, B], nb0=type_size
    GGML_FATTN_V_LAYOUT_NATIVE_KHDB = 1,  // [K, H, D, B], nb0=strided
    GGML_FATTN_V_LAYOUT_PERM_KDHB   = 2,  // [K, D, H, B], nb0=strided
};
```

**op_params slot availability**:

| Slot | Content | Size | User |
|------|---------|------|------|
| [0] | attention_scale | float (reinterpret_cast) | All |
| [1] | max_bias | float | All |
| [2] | logit_softcap | float | All |
| [3] | precision | GGML_PREC_* | CPU FA only |
| [4] | instruction | ggml_fattn_instruction | CUDA HIP |
| **[5]** | **FREE** | **int32** | **Available for v_layout** |

`GGML_MAX_OP_PARAMS = 64` bytes → 16 int32 slots. Slots 0-4 used. Slot [5] free.

Writer: `build_attn_mha()` sets it after op creation.
Reader: each backend checks it before deciding support.

---

## 8. Test matrix

**Layout correctness harness** — dump V ne/nb at each transform step:

| Scenario | v_trans | Cache V type | pp | Self/Cross/Assist |
|----------|---------|-------------|-----|-------------------|
| A1 | false | f16 | 16 | self |
| A2 | false | f16 | 512 | self |
| A3 | false | q8_0 | 512 | self |
| B1 | true | f16 | 16 | self |
| B2 | true | f16 | 512 | self |
| B3 | true | q4_0 | 512 | self |
| C1 | false | f16 | 16 | assist (MTP) |
| C2 | true | f16 | 512 | assist (MTP) |

**Backend routing harness**:

| Route | pp | Expected backend | V layout |
|-------|-----|-----------------|----------|
| baseline | 16 | VEC | FA_DKHB |
| baseline | 512 | VEC | FA_DKHB |
| `packed16_vec` | 512 | VEC (shadow) | FA_DKHB |
| `packed16_wmma_tile` | 16 | PWMMA | PERMUTED_KDHB |
| `packed16_wmma_tile` | 512 | PWMMA | PERMUTED_KDHB ⚠️ (crash target) |
| `rocm_packed16_dot4_mmq` | 16 | DOT4-MMQ | FA_DKHB |
| `rocm_packed16_dot4_mmq` | 512 | DOT4-MMQ | FA_DKHB |

⚠️ = known blocker: pp512+ PWMMA with native v_trans. Root cause:
```
1. packed16 I32 K → fattn.cu selects BEST_FATTN_KERNEL_PACKED16_WMMA_TILE
2. But cparams.flash_attn is false (packed16 K doesn't require FA flag)
3. So ggml_cuda_flash_attn_ext_supported() returns true for PWMMA
4. GGML scheduler may still route to CPU FA:
   - GPU backend might reject the op (VRAM, stride check, or other)
   - CPU FA gets it, hits: ASSERT(nbv0 == ggml_type_size(v->type))
   - Crash: V has nb[0] = kv_size*D*ts, not ts
5. Alternatively: fattn-common.cuh assert fires before PWMMA even runs
   - VEC/TILE path checks V->nb[0] == ts before checking for PWMMA
```

**The crash is NOT in the PWMMA kernel itself.** It is in the routing: V with bad nb0
reaches a backend (CPU/VEC) that requires nb0==ts, before PWMMA gets a chance to handle it.

---

## 9. Known invariants that must hold

1. **Physical V is D-contiguous**: `[D_total, K, streams]` in cache. Never changes.
2. **get_v() with v_trans=false**: Returns D-contiguous view. Safe for all backends.
3. **get_v() with v_trans=true**: Returns strided view. Only PWMMA handles it.
4. **fattn-common.cuh asserts nb0==ts**: Blocks native layouts from VEC/TILE/MMA/WMMA.
5. **CPU FA asserts nbv0==ts**: Same constraint.
6. **ggml_permute happens before PWMMA check**: PWMMA inherits intermediate layout via permute→skip-transpose.
7. **All V consumers go through build_attn_mha()**: Three call sites, one function. Patch once.

---

## 10. Patch staging (for reference only — not implemented)

| Stage | What | Risk |
|-------|------|------|
| **Stage 1** | Add `ggml_fattn_v_layout` enum + op_params[5] writer in build_attn_mha(). No backend behavior change. Add logging. | Zero risk. |
| **Stage 2** | PWMMA reads layout from op_params instead of stride heuristics. | Medium — PWMMA already layout-aware, just formalize. |
| **Stage 3** | Scheduler: CUDA supports_op returns true for PWMMA routes; never falls back to CPU for packed16. | Medium — affects backend assignment. |
| **Stage 4** | DOT4/DOT4-MMQ read layout from op_params. | Low — already stride-based access. |
| **Stage 5** | VEC/TILE/CPU: if layout != FA_DKHB, reject early with clear error instead of assert crash. | Low — just improves error messages. |

---

## 11. The deeper contract problem

**Root cause**: Physical V storage is always D-contiguous (`[D_total, K, streams]`). The problem is
entirely in which VIEW `get_v()` returns and how `build_attn_mha()` transforms it.

### 11.1 Why FA consumers receive non-FA views

`get_v()` decides its output based on `cache.v_trans` which is a **global cache property**:

```cpp
v_trans = !cparams.flash_attn  // llama-model.cpp:2090
```

When `cparams.flash_attn = false` but packed16 I32 K forces FA:
```cpp
// llama-graph.cpp:2001
const bool use_flash_attn = (cparams.flash_attn || k_is_packed16_i32) && kq_b == nullptr;
```
The graph CORRECTLY takes the FA path. But `get_v()` still returns the non-FA v_trans view
because `cache.v_trans` is still `true`. The mismatch propagates.

**The fix is NOT in PWMMA** — it's upstream:
- Either `get_v()` should accept a `desired_layout` parameter
- Or `build_attn_mha()` should call a separate `get_v_fa()` when FA will be used
- Or `cache.v_trans` should be overridable per-graph-node

### 11.2 v_trans is consumer-specific, not cache-global

`get_v()` with v_trans=true serves the NON-FA path at line 2200:
```cpp
// llama-graph.cpp:2200
if (!v_trans) {
    v = ggml_cont(ctx0, ggml_transpose(ctx0, v));  // convert to [K,H,D] for mul_mat
}
ggml_tensor * kqv = ggml_mul_mat(ctx0, v, kq);  // expects [K,H,D]
```

The non-FA path NEEDS `[K,H,D]` for `ggml_mul_mat`. But the FA path NEEDS D-contiguous.
One view cannot serve both. The current design picks the non-FA view at `get_v()` time
and then the FA path permutes it back. PWMMA is the only backend that skips the transpose.

### 11.3 v_trans detection is heuristic, not explicit

```cpp
// llama-graph.cpp:1996 — detected from tensor strides, not from cache metadata!
const bool v_trans = v->nb[1] > v->nb[2];
```

This heuristic is fragile:
- If `get_v()` changes its output layout, the detection silently changes
- At pp16 where D==K==256, the heuristic may give wrong results (shape aliasing)
- There is no explicit `v_layout` flag in the tensor or op

### 11.4 Physical storage IS D-contiguous — the view is the only problem

```
v_trans=false get_v():  view_4d(D_head, H, K, ns) with nb0=row_size(D_head) → FA_DKHB ✅
v_trans=true  get_v():  view_4d(K, H, D_head, ns) with nb0=row_size(kv_size×D_head) → NATIVE_KHDB
Physical cache:         tensor_3d(D_total, K, streams) — same bytes either way
```

`get_v()` with v_trans=false and `get_v()` with v_trans=true produce DIFFERENT VIEWS
of the SAME physical bytes. The FA-compatible view exists; it's just gated behind `v_trans=false`.

### 11.5 PACKED16_K_CACHE graph contamination

`PACKED16_K_CACHE=1` replaces the shadow K tensor with I32 packed16:
```cpp
// llama-kv-cache.cpp:268
k = nullptr;  // no shadow K when packed16_active
kp = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, D/4, kv_size * H_kv, streams);  // payload
ks = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, D/32, kv_size * H_kv, streams);  // scales
```

`get_k()` returns I32 view when `!k && kp`:
```cpp
// llama-kv-cache.cpp:1276
return ggml_view_4d(ctx, kp, D/4, n_kv, H_kv, ns, ...)
// ne = [D/4, K, H, ns] — I32 type
```

V is NEVER affected — `packed16_active` only touches K allocation.

The `ggml_can_mul_mat` failure: if the non-FA path at line 2200 receives I32 K (from packed16 K cache),
`ggml_mul_mat` cannot handle I32. The safeguard is `k_is_packed16_i32 → use_flash_attn=true`,
but if the scheduler overrides FA routing to CPU, I32 K reaches a backend that can't handle it.

---

## 12. Mask contract (under-researched)

```cpp
// llama-graph.cpp:24-38
kq_mask: ne = [n_kv, n_tokens/n_stream, 1, n_stream]
kq_mask type: F32 (created as F32, cast to F16 for FA at line 2094)
```

```cpp
// llama-graph.cpp:2093-2095 — FA path mask type coercion
if (kq_mask && kq_mask->type != GGML_TYPE_F16) {
    kq_mask = ggml_cast(ctx0, kq_mask, GGML_TYPE_F16);
}
```

**Mask invariants for FA:**
- Type: F16 (FA requires F16 mask; F32 is cast at graph time)
- Shape: `[n_kv, nq, 1, n_stream]`
- Values: `-inf` for masked positions, `0.0f` for unmasked
- Causal diagonal: applied separately via `q_pos_base`/`kv_pos_base` in backend

**Mask risks:**
- No explicit mask layout enum — assumed shape is always `[n_kv, nq]`
- Batch broadcast: `mask->ne[2] == 1`, `mask->ne[3] == n_stream`
- If n_stream changes between layers, mask shape may mismatch

---

## 13. Backend support contract — scheduler coupling

**Current flow:**
```
1. Graph builds FLASH_ATTN_EXT node with K=I32, V=[K,D,H]
2. ggml_backend_cuda_supports_op() → ggml_cuda_flash_attn_ext_supported()
3. Returns true if get_best_fattn_kernel() != NONE
4. Scheduler may still route to CPU if CUDA rejects or if CPU is available
5. CPU FA: ASSERT(nbv0 == ts) → CRASH on v_trans V
```

**Required invariant:**
```
For K=I32 packed16 + v_trans=true V:
  → CUDA must claim support (PWMMA or DOT4-MMQ)
  → CPU must reject (nbv0 != ts)
  → Scheduler must NOT fall back to CPU
```

Currently, CPU FA has NO layout-aware rejection — it just asserts. If the scheduler
sends v_trans V to CPU FA, it crashes at the assert, not at a clean error.

---

## 14. Layout contract struct proposal

Current implicit assumptions to make explicit:

```cpp
struct fattn_layout_contract {
    // Populated by build_attn_mha() from cache metadata + route hints
    ggml_fattn_v_layout   v_layout;     // op_params[5]
    ggml_fattn_k_layout   k_layout;     // op_params[6] (future)
    ggml_fattn_route_class route_class;  // op_params[7] (future: baseline/DOT4/PWMMA/MMQ)
};
```

**Q: Is a full struct needed, or just v_layout?**

A: v_layout alone fixes the immediate PWMMA crash. But K layout (I32 vs f16 vs q8_0)
and route class (DOT4 vs PWMMA vs baseline) determine backend dispatch. If only v_layout
is explicit, the next bug will be a K layout mismatch.

**Minimum viable contract for Option D stage 1:**
- `op_params[5] = v_layout` (ggml_fattn_v_layout)
- Everything else stays implicit but gets assertions in support checks

---

## 15. Artifacts still needed

### A. Graph layout trace
For each (route, pp, V_type) combination, dump:
- get_v() output ne/nb
- build_attn_mha() V after permute
- build_attn_mha() V_for_fa
- op_params[4] (instruction) assigned
- Selected backend in support probe
- Actual backend in dispatch

### B. Quantized V loader proofs
For q4_0/q8_0 in v_trans=false:
- Exact scalar decode for d=0,1,31,32,255
- Block boundary behavior
- Verify nb[0] = sizeof(block) works for all backends

### C. Minimal non-aliasing test grid
```
D=128, K=512  (D != K)
D=256, K=128  (D != K)
D=256, K=512  (D != K)
pp16 avoided (D==K aliasing hides axis swaps)
```

**Research finding:** The v_trans heuristic `v->nb[1] > v->nb[2]` at line 1996
is only fragile when D==K and H==K (all dimensions equal). In practice with
GQA (H_kv < H_q), the heuristic is usually correct because heads dimension
separates the axes. But pp16 with 1 head and D=256,K=256 would alias.

The bigger risk: `get_v()` returns ne=[K, H, D] with nb0=strided. If a backend
only checks ne[0]==D (not nb[0]), it passes. Must always check nb[0]==ts AND
layout enum together.

### C2. Mask layout contract (researched)

**Mask creation** (`build_attn_inp_kq_mask`, line 24-42):
```cpp
type = cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32
ne = [n_kv, n_tokens/n_stream, 1, n_stream]
```

**Hybrid mode mask type bug:**
```
cparams.flash_attn = false → mask created as F32
k_is_packed16_i32 → use_flash_attn = true
FA path casts mask to F16 at line 2093: ggml_cast(ctx0, kq_mask, GGML_TYPE_F16)
```
This adds an extra `ggml_cast` op in the graph when packed16 forces FA.
F32→F16 of -inf/0.0f values is lossless, so the cast itself is safe.
But the extra graph node is avoidable.

**Mask reuse** (`can_reuse_kq_mask`, line 43-62):
Checks ne[0..3] but NOT type. If the mask was F32 from a previous layer
and the current layer needs F16 (because packed16 forced FA), the reuse
check says "yes" but the type is wrong → triggers the cast again.

**Mask values:**
- `-inf` for masked positions, `0.0f` for unmasked
- Created as: data[i] = hparams.use_alibi ? -abs(p0-p1) : 0.0f
- Causal diagonal applied per-backend via q_pos_base/kv_pos_base

**Mask invariants for FA backends:**
- Type: F16 (cast if F32 at graph time)
- ne[0] = n_kv, ne[1] = nq, ne[2] = 1, ne[3] = n_stream
- Values: -inf/0.0f only (no intermediate soft values from ALiBi in FA mode)

### D. PACKED16_K_CACHE graph diff
Run same model with/without PACKED16_K_CACHE=1, dump all attention ops:
- K type, V type, V ne/nb
- FA vs non-FA path
- Backend assignment

**Research completed (static analysis):**

PACKED16_K_CACHE has TWO modes, neither touches V:

1. **Shadow mode** (`k != nullptr`): K cache has shadow q8_0 tensor. get_k() returns
   standard f16/q8_0 view. Packed16 payload/scales are side-channel only.

2. **Packed16-only mode** (`k == nullptr, kp != nullptr`): No shadow K. get_k() returns
   I32 view: `ne=[D/4, n_kv, H_kv, ns]`. K type = GGML_TYPE_I32.

V is NEVER affected: `ggml_new_tensor_3d(ctx, type_v_layer, D_total, kv_size, n_stream)`
regardless of packed16_active. The `ggml_can_mul_mat` failure at pp512 is a K-type
routing issue (I32 K reaching a non-MMQ matmul), not a V-layout issue.

Graph trace with packed16-only mode (K=I32):
```
get_k() → ne=[D/4, n_kv, H_kv, ns], type=I32
get_v() → ne=[K, H_v, D_head, ns], type=f16/q4/q8 (depends on v_trans)
build_attn_mha():
  k_is_packed16_i32 = true
  use_flash_attn = (false || true) && kq_b==nullptr → true
  → FA path always taken (assert kq_b==nullptr)
  v_trans = v->nb[1] > v->nb[2]  (heuristic, NOT cache.v_trans)
  global permute v
  if v_trans && !pwmma: transpose to FA_DKHB
  if v_trans && pwmma:  skip transpose → PERMUTED_KDHB
  FLASH_ATTN_EXT created
  instruction = PREFILL_QK / DECODE_QK / MTP_VERIFY_QK
```
Edge case: if kq_b != nullptr with I32 K, use_flash_attn=false, non-FA path
runs ggml_mul_mat(k,q) with I32 K → would crash. But kq_b is ALiBi bias, essentially
always nullptr for modern LLaMA-style models. Protected by FA assert at line 2073.

---

## 20. Final validation: get_v(FOR_FA) prerequisites

### 20.1 Zero-copy for quantized V types

**q4_0**: `get_v(FOR_FA)` nb0 = `ggml_row_size(GGML_TYPE_Q4_0, D_head)` = D_head/32 × 18 bytes
- v_trans=false already uses this exact formula → ✅ same bytes, different view_4d args

**q8_0**: `get_v(FOR_FA)` nb0 = `ggml_row_size(GGML_TYPE_Q8_0, D_head)` = D_head/32 × 34 bytes
- Same formula as v_trans=false → ✅

**TBQ4_0**: Always uses `ggml_view_3d` with `nb0=ggml_row_size(type, D_total) == sizeof(type)`
- Only one branch, not affected by v_trans → ✅ already FA-compatible

**f16**: `get_v(FOR_FA)` nb0 = `ggml_row_size(GGML_TYPE_F16, D_head)` = D_head × 2 = sizeof(half)
- v_trans=false nb0 = sizeof(type) for single-head → ✅

**Conclusion**: `get_v(FOR_FA)` workspace identically for all V types. The
`ggml_row_size(type, D_head)` formula is type-agnostic and already proven in
the v_trans=false code path (which has runtime asserts verifying nb0 == type_size).

### 20.2 Can call sites decide use_flash_attn before get_v()?

**Decision formula** (from build_attn_mha line 2001):
```cpp
use_flash_attn = (cparams.flash_attn || k->type == GGML_TYPE_I32) && kq_b == nullptr
```
Factors needed:
- `cparams.flash_attn` — available at all call sites (member function, this->cparams)
- `k->type` — known AFTER get_k() returns
- `kq_b == nullptr` — always true for FA-capable paths

**All 3 get_v() call sites have get_k() on the preceding line:**

| Site | File:Line | get_k() line | get_v() line | K type known? | cparams? |
|------|-----------|-------------|-------------|---------------|----------|
| self-attn (kv_impl) | 2377-2378 | 2377 | 2378 | ✅ | ✅ (member fn) |
| cross-attn | 2556-2557 | 2556 | 2557 | ✅ | ✅ |
| assist (MTP) | 2644-2645 | 2644 | 2645 | ✅ | ✅ (src_cur context) |

**Exception**: `build_attn_inp_k_impl` (line 2469) does NOT call get_v() — it creates
a V view from K tensor: `ggml_view_4d(ctx0, k, ...)`. This is a K/V shared-layout
optimization. Not relevant to get_v(FOR_FA).

**Required code change at each site** (conceptual, not implemented):
```cpp
ggml_tensor * k = mctx_cur->get_k(ctx0, il);
bool use_fa = cparams.flash_attn || k->type == GGML_TYPE_I32;
ggml_tensor * v = mctx_cur->get_v(ctx0, il, use_fa ? LLAMA_V_LAYOUT_FOR_FA : LLAMA_V_LAYOUT_FOR_NON_FA);
```

**Conclusion**: All three call sites can determine FA usage before get_v().
The K type is known from get_k() on the preceding line. cparams is available.
kq_b is always nullptr for these paths.

---

## 16. Refined architecture plan

### Root bug class

```
packed16 forced FA → hybrid mode:
    cparams.flash_attn = false    (global)
    cache.v_trans = true          (set from !cparams.flash_attn)
    get_v() → non-FA view         (decided by cache.v_trans)
    k_is_packed16_i32 → use_flash_attn = true  (per-node override)
    → FA op receives non-FA V view
```

The fix is not a layout enum in op_params. The fix is making V layout a **consumer request**:

```cpp
// Current (global side-effect):
get_v(ctx, il)  // decides from cache.v_trans

// Target (consumer-driven):
get_v(ctx, il, LLAMA_V_LAYOUT_FOR_FA)     // returns FA_DKHB
  or
get_v_fa(ctx, il)                           // explicit FA view
get_v_non_fa(ctx, il)                       // explicit non-FA view
```

Because physical storage is D-contiguous and all consumers go through `build_attn_mha()`,
the same cache bytes can serve both views. The consumer chooses.

### Staging (research is prerequisite, no implementation yet)

| Stage | What | Prerequisite research |
|-------|------|----------------------|
| 0 | Commit research doc as-is | ✅ Done |
| 1 | `op_params[5] = v_layout` + logging only | None |
| 2 | CPU FA rejects non-DKHB in support (not assert-crash) | None |
| 3 | Scheduler: packed16 PWMMA → CUDA only, never CPU/DOT4 fallback | Graph diff (artifact D) |
| 4 | PWMMA reads `op_params[5]` instead of stride heuristics | V layout dump matrix |
| 5 | `get_v(layout_request)` API — consumer-driven views | Non-aliasing tests, consumer audit |
| 6 | DOT4/DOT4-MMQ native layout support | Quantized V loader proofs |

### Top 3 remaining research priorities

1. **Graph diff with/without PACKED16_K_CACHE**
   - Dump all attention nodes: K type, V ne/nb, v_layout, backend support, backend assignment
   - Explains the `ggml_can_mul_mat` failure at pp512
   - Validates that packed16 only touches K, never V

2. **Non-aliasing layout tests**
   - D=128,K=512, D=256,K=128, D=256,K=512
   - Stop trusting pp16 where D==K hides axis swaps
   - Verify nb[0]==ts for FA_DKHB at non-trivial sizes

3. **Mask layout contract**
   - mask ne/nb per route
   - causal diagonal ownership (graph vs backend)
   - broadcast rules
   - F32→F16 cast timing

### Key insight to preserve

```text
V layout must be requested by the consumer.
It cannot be derived globally from cparams.flash_attn.
```

That is the architectural bug class. Everything else — PWMMA layout, CPU fallback,
pp512 crash — is downstream.

---

## 17. get_v(FOR_FA): Proof that FA-compatible view is zero-copy

**Claim**: `get_v()` can return a D-contiguous `[D_head, H_kv, K, ns]` view with
`nb[0]==type_size` even when `cache.v_trans=true`. Zero copy, view-only.

**Physical storage** (always the same):
```
ggml_new_tensor_3d(ctx, type, D_total, kv_size, n_stream)
ne  = [D_total, kv_size, n_stream]
nb0 = type_size (f16) or sizeof(block_qX_0) (quantized)
nb1 = nb0 * D_total          ← token-row stride
nb2 = nb1 * kv_size            ← stream stride
```

**v_trans=false get_v()** (FA-compatible, already correct):
```
ggml_view_4d(ctx, v,
  D_head, H_kv, n_kv, ns,                    ← ne shape
  row_size(type, D_head),                    ← nb0 = type_size for single head ✅
  row_size(type, D_total),                   ← nb1 = head stride in cache
  row_size(type, D_total * kv_size),         ← nb2 = token stride
  row_size(type, D_total * kv_size) * s0)    ← nb3 = sequence offset
```
Produces FA_DKHB with `nb[0] == type_size` ✅

**v_trans=true get_v()** (current, non-FA):
```
ggml_view_4d(ctx, v,
  n_kv, H_kv, D_head, ns,                    ← ne shape
  row_size(type, kv_size * D_head),          ← nb0 = K×D_head×type_size ❌
  row_size(type, kv_size),                   ← nb1 = token stride
  row_size(type, kv_size * D_total),         ← nb2 = GQA head-pair stride
  row_size(type, kv_size * D_total) * s0)    ← nb3
```
Produces NATIVE_KHDB with nb0 ≫ type_size ❌

**Hypothetical get_v(FOR_FA) on v_trans=true cache** — just use v_trans=false's view_4d args:
```
ggml_view_4d(ctx, v,
  D_head, H_kv, n_kv, ns,
  row_size(type, D_head),
  row_size(type, D_total),
  row_size(type, D_total * kv_size),
  row_size(type, D_total * kv_size) * s0)
```
**SAME physical bytes as v_trans=true's view.** Different `ne` and `nb`.
No `ggml_cont`. No allocation. No copy. ✅

**Conclusion**: `get_v(FOR_FA)` is a one-line change per branch — use the v_trans=false
view_4d arguments regardless of `cache.v_trans`. The physical layout is already D-contiguous.

---

## 18. Scheduler routing findings

### CUDA supports_op for FLASH_ATTN_EXT (`ggml-cuda.cu:5259`)
```cpp
// Special PWMMA force: if route requires PWMMA and K=I32/V=F16, claim support
if (require_pwmma && K==F32 && K_src==I32 && V==F16) return true;
return ggml_cuda_flash_attn_ext_supported(dev_ctx->device, op);
```
CUDA has a targeted override for PWMMA route. Other routes fall through to
`ggml_cuda_flash_attn_ext_supported()` which calls `get_best_fattn_kernel()`.

### CPU supports_op for FLASH_ATTN_EXT (`ggml-cpu.cpp:423`)
```cpp
// No explicit FLASH_ATTN_EXT case → falls through to default:
default: return true;  // CPU claims to support EVERY op
```
**CPU ALWAYS claims to support FLASH_ATTN_EXT**, regardless of K type or V layout.
The rejection happens at compute time via `GGML_ASSERT(nbv0 == ts)` — an assert crash,
not a clean rejection.

### Compute path for CPU FA (`ggml-cpu.c:2001`)
```cpp
case GGML_OP_FLASH_ATTN_EXT:
    ggml_compute_forward_flash_attn_ext(params, tensor);
    break;
```
No type/layout check before dispatch. The assert at `ops.cpp:8900` fires inside compute.

### Scheduler assignment risk
```
1. Graph builds FA node with K=I32, V=NATIVE_KHDB
2. CUDA supports_op returns true (PWMMA override or fattn kernel selected)
3. CPU supports_op returns true (default: return true)
4. Scheduler has CHOICE between CUDA and CPU
5. If scheduler picks CPU → crash at compute-time assert
```

### Required fix
CPU supports_op must reject FLASH_ATTN_EXT when:
```
K type == I32 || V layout != FA_DKHB
```
Otherwise the scheduler can route packed16 FA to CPU.

---

## 19. Graph layout trace — logging injection points

Seven trace points needed, no behavior changes:

| # | File | Line | What to log |
|---|------|------|-------------|
| 1 | `llama-kv-cache.cpp` | after view_4d | il, v_trans, V ne/nb, nb0==ts |
| 2 | `llama-graph.cpp` | 1996 | v_trans detected, k_is_packed16, use_flash_attn, V in ne/nb |
| 3 | `llama-graph.cpp` | after 2068 | V after global permute: ne/nb |
| 4 | `llama-graph.cpp` | after 2079 | v_for_fa ne/nb, path (transpose/skip) |
| 5 | `llama-graph.cpp` | after 2097 | op_params[4], mask ne/nb/type |
| 6 | `fattn.cu` | `_supported()` | selected kernel, nq, n_kv, V ne/nb |
| 7 | `fattn.cu` | dispatch | actual kernel, V ne/nb at dispatch |

**Test matrix for trace**:
- baseline, PACKED16_K_CACHE, PWMMA forced, DOT4 forced
- pp16, pp512, pp4096
- V types: f16, q4_0, q8_0
