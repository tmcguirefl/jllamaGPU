# GPU J engine guide — rewrite jllama on ggml

This tree is a **GPU fork workspace** of `jllama_dev`. The J 9.8 runtime on this machine already has the GPU `libj`. Start here; do not invent a second engine.

**Session goal:** keep tokens/embeddings on the CPU as needed, keep **weights and activations on the GPU**, and implement a small LLM (TinyStories / stories15M, then Llama-scale) using the verbs below. CPU transfer is for viewing, tokenize, and sample — not for every matmul.

---

## 1. Runtime (read this first)

| What | Path |
|------|------|
| J console | `/Users/tomdevel/j9.8/bin/jconsole` |
| Engine | `/Users/tomdevel/j9.8/bin/libj.dylib` (gpu-resident, 2026-08-29 beta7) |
| ggml dylibs | same `bin/` (`libggml*.dylib`, rpath `@loader_path`) |
| Stock J backup | `/Users/tomdevel/j9.8/bin/libj.dylib.jsoftware-beta5` |
| Engine source | `/Users/tomdevel/jdev/jsource` branch `gpu-resident` (commit `a17db8b9`) |
| Named wrappers | `jgpu.ijs` in this directory (copy of `jsource/gpu-phase0/jgpu.ijs`) |
| Engine tests | `/Users/tomdevel/jdev/jsource/gpu-phase0/test_phase{1..5}.ijs` |

**Never** `/usr/bin/jconsole` (Java). **Never** assume `/Applications/j9.8` — this machine uses `/Users/tomdevel/j9.8`.

Do **not** pass `-lib` unless you are testing a freshly built `jsource` dylib. Default jconsole already loads the GPU engine.

Check:

```
/Users/tomdevel/j9.8/bin/jconsole
   9!:14''
```

Expect `j9.8.0-beta7` and date `2026-08-29` (clang-21). Stock Jsoftware was `beta5` / `2026-07-06`.

Working type on GPU: **F32**. J `FL` on CPU is IEEE double. Upload converts; download of F32 is exact enough for F32; vs F64 expect ~1e-3 on GEMM, ~1e-5 on unary.

Display of a GPU noun does **not** densify. `8!:0 y` is a one-line summary like `gpu q4_0 4096 atoms, 2304 bytes`.

---

## 2. What a GPU noun is

The engine **hijacks the sparse header**. Classic sparse is not used in this fork.

- `P.a[0] = _1` sentinel
- `2 $. y` → ggml type id (`0` F32, `1` F16, `2` Q4_0, `6` Q5_0, `8` Q8_0, `12` Q4_K, …)
- `4 $. y` → `nbytes , natom` (packed size, not `4 * natom` for quants)
- `$ y` and `# y` are the J shape (last axis = **K** for quantized weights)

```j
g =. $. x          NB. dense numeric → GPU F32
h =. $.^:_1 g      NB. download to FL (Q4/F16 dequant to float)
```

Quantized nouns stay **packed**. Do not dequant at load. The only v1 consumer of a packed quant noun is **left argument of** `+/ .*` (and download for debugging).

---

## 3. Verb table (what the rewrite should call)

### 3.1 Identity / inspect

| Phrase | Meaning |
|--------|---------|
| `$. y` | upload dense B01/INT/FL → GPU F32 |
| `$.^:_1 y` / `0 $. y` | download; quants dequant to FL |
| `2 $. y` | ggml type id |
| `4 $. y` | packed `nbytes , natom` |
| `8!:0 y` | summary string; **no** auto densify |

### 3.2 Linear algebra

| Phrase | Meaning |
|--------|---------|
| `A +/ .* B` | both GPU. **Left** F32/F16/quant; **right and result F32** |
| Last axis of left | contracting **K**; must be a multiple of the ggml block size |

Block sizes: F32/F16 = 1; Q4_0/Q5_0/Q8_0 = **32**; Q4_K/Q5_K/Q6_K = **256**.

### 3.3 Fused kernels (ggml/Metal)

| Phrase | Meaning |
|--------|---------|
| `128!:35 x` | SiLU, any rank, F32 |
| `128!:36 x` | softmax on **last** axis |
| `w 128!:37 x` | RMSNorm, eps=`1e_5`; `w` gain length = last axis of `x` |
| `128!:37 x` | RMSNorm without gain |
| `128!:38 x` | RoPE, `pos = i. n_tok`, θ=10000, NORMAL, full last axis |
| `pos 128!:38 x` | RoPE; `x` rank 2 `n_tok × d` or rank 3 `n_tok × n_head × d` |
| `(pos;theta;n_rot;mode) 128!:38 x` | `mode` 0 = Llama NORMAL, 2 = NeoX; `n_rot` even, `≤ d_head` |

Dense numeric args to these foreigns are auto-uploaded; result is always GPU F32.

Named locale (load `jgpu.ijs`):

```j
load ROOT , 'jgpu.ijs'          NB. after setroot
silu =: silu_jgpu_
softmax =: softmax_jgpu_
rmsnorm =: rmsnorm_jgpu_
rope =: rope_jgpu_
```

### 3.4 F32 array verbs that stay on device

These are the primitives jllama already writes. They work when the nouns are GPU F32 (and `+` `*` `-` `%` also auto-upload a dense other arg: mask, scale, bias).

| Phrase | Use |
|--------|-----|
| `+` `*` `-` `%` | residual, SwiGLU mul, score scale, mask add |
| `|: y` | reverse axes (linear helper) |
| `1 0 2 \|: y` | head permute `n_tok × n_head × d` ↔ `n_head × n_tok × d` |
| `,` | catenate **items** (axis 0). KV: `kc , K` |
| `((0,n_kv,d)$0) , K` | empty dense + GPU = start of cache |
| `{` | item from: `h { qb`, `idx { y` (INT atom or list; negatives OK) |
| `$` | reshape; **atom count must match** (no fill) |
| `, y` | ravel |
| `,: y` | add a leading 1 (via reshape) |
| `< y` and `x ; y` | boxing GPU nouns is allowed |

`#` and `$` on a GPU noun read the header (no download).

### 3.5 Load / quantize

| Phrase | Meaning |
|--------|---------|
| `128!:33 'file.gguf'` | boxed table shape `(#tensors, 2)` of `(name ; GPU noun)` |
| `128!:34 W` | dense → **Q4_0** GPU; last axis `% 32` |
| `tid 128!:34 W` | dense → ggml type `tid`. Write `Q8_0 128!:34 W` or `8 (128!:34) W` — **not** `8 128!:34 W` (parses as `(8 128)!:34`) |

Type ids: `F16=1 Q4_0=2 Q4_1=3 Q5_0=6 Q5_1=7 Q8_0=8 Q4_K=12 Q5_K=13 Q6_K=14` (also in `jgpu.ijs`).

GGUF load does **not** densify. J shape is ggml dims **reversed**, so **last axis = ggml `ne[0]` = K**. A 2-D ggml weight `[n_in, n_out]` becomes J `n_out × n_in`.

---

## 4. Layout — the one thing that will break a naive port

Current jllama (`docs/conventions.md`):

```
x:  n_tok × n_in
W:  n_in  × n_out
y =. x +/ . * w
```

GPU / GGUF / ggml `mul_mat`:

```
W:  n_out × n_in     NB. last axis = K, quantized along K
x:  n_tok × n_in
y =. |: W +/ .* |: x
```

You **cannot** `|: ` a quantized noun. Quantized `W` must stay **left** of `+/ .*`. Activations (F32) are transposed in and the F32 result is transposed out. That is `linear_jgpu_`:

```j
linear =: 4 : '|: y +/ .* |: x'   NB. x activations, y = W (n_out × n_in)
```

Existing `core/tensor.ijs` `linear` (`b +"1 x +/ . * w` with `w` = n_in × n_out) is the **CPU** convention. For GPU, either:

- keep GGUF-native `n_out × n_in` and use `linear_jgpu_`, or
- upload `|:` of a CPU `n_in × n_out` **before** `128!:34` so the packed last axis is `n_in`.

Do not `x +/ .* Wq` with `Wq` quantized.

SwiGLU:

```j
h =. (silu xv linear wg) * (xv linear wu)
h linear wd
```

(`swiglu_jgpu_`). Residual is ordinary `+` of two GPU F32 nouns.

---

## 5. Map from current jllama files

| Current | GPU change |
|---------|------------|
| `core/tensor.ijs` `silu` `softmax` `rmsnorm` | bind to `128!:35-37` (or `*_jgpu_`) when args are GPU |
| `linear` / every `+/ . *` on weights | `linear_jgpu_` / `|: W +/ .* |: x` |
| `causal_mask` | stay dense FL; `scores + mask` auto-uploads mask |
| `core/rope.ijs` | `pos 128!:38 x` (NORMAL). NeoX: `(pos;theta;n_rot;2) 128!:38 x` |
| `split_heads` / `merge_heads` | already `$ ,` — works on GPU F32 |
| `kv_empty` | `((0, n_kv, d)$0)` dense empty; `kc , K` starts the cache |
| `expand_kv` | keep `1 0 2 \|: idx { 1 0 2 \|: y` |
| `attention_heads` box loop | **do not** `heads , < …` then `> heads`. Open of a list of GPU boxes will not assemble one array. Cat items: `O =. ((0, n_q, dh)$0)` then `O =. O , attention1 …` then `1 0 2 \|: O` |
| `mha_step` `kc , K` | works on GPU |
| `,: xv` | works |
| `io/gguf.ijs` | either call `128!:33` and keep packed GPU nouns, or keep the J parser for **metadata/tokenizer only** and take tensor bytes from `128!:33`. Do not promote weight tensors to f64. |
| `ffn_swiglu` / `block_*` | GPU silu + linear + `+` `*` |

Tokenizer, sampling, GGUF **metadata**, and token ids stay CPU.

---

## 6. What is still nonce / illegal

- Classic sparse
- GPU `+/ .*` with quantized **right** arg (right must be F32)
- `|:` of a quantized noun
- `>` of a **list** of boxed GPU arrays (scalar box `> <g` is fine)
- IQ-type `128!:34` (need imatrix). GGUF **load** of those types still stores packed data; matmul may work if Metal has the kernel
- Reshape that changes atom count (no fill)
- Mixed GPU + classic-sparse
- Most other primitives (`^` `%.` `+/` on GPU, etc.) — use the named kernels

If a verb  nonces, the noun is probably still GPU; download only to debug.

---

## 7. Suggested first slice

1. Smoke: `$.` roundtrip, `128!:33` on `test/fixtures/tiny_parity_f16.gguf` or jsource `gpu-phase0/tiny_q4.gguf`.
2. Replace `silu` `softmax` `rmsnorm` `rope` with GPU foreigns; keep CPU fallbacks for dense.
3. GGUF weights: packed GPU nouns, shape `n_out × n_in`.
4. FFN + RMSNorm + residual on GPU (SwiGLU).
5. Attention: split/merge + `1 0 2 \|:` + per-head `{` + cat (no `> heads`) + `scores % %: d` + dense causal mask.
6. KV decode: empty `,` append.
7. TinyStories `stories15M.F16.gguf` — models live in `jllama_dev/models/` (not copied here). Symlink if needed:

```
ln -s /Users/tomdevel/jdev/jllama_dev/models /Users/tomdevel/jdev/jllamaGPU/models
```

Engine self-tests (already passing on this machine):

```
/Users/tomdevel/j9.8/bin/jconsole /Users/tomdevel/jdev/jsource/gpu-phase0/test_phase3.ijs
/Users/tomdevel/j9.8/bin/jconsole /Users/tomdevel/jdev/jsource/gpu-phase0/test_phase4.ijs
/Users/tomdevel/j9.8/bin/jconsole /Users/tomdevel/jdev/jsource/gpu-phase0/test_phase5.ijs
```

---

## 8. Engine internals (only if libj must change)

Source: `/Users/tomdevel/jdev/jsource` on branch `gpu-resident`.

- `jsrc/gpu.c` — J-facing (includes `j.h`, talks only to `gpu_api.h`)
- `jsrc/gpu_ggml.c` / `gpu_metal.m` — **never** include `j.h`; Metal/ggml stay out of JE headers
- Rebuild: `cd /Users/tomdevel/jdev/jsource/make2 && NOCLEAN=1 jplatform=darwin j64x=j64arm ./build_libj.sh`
- Install into the runtime: copy `bin/darwin/j64arm/libj.dylib` and `libggml*.dylib` to `/Users/tomdevel/j9.8/bin/`

Do not treat this as restoring classic Jsoftware sparse. This fork hijacks that slot on purpose.

---

## 9. Boxing packs (still true)

jllama `docs/conventions.md`: chained `x ; y ; z` re-boxes. For mixed GPU nouns + integers use:

```j
(<x) , (<n_head) , layerbox , (<theta)
```

GPU nouns **may** be boxed (`<Wq` works).

---

## 10. Licenses (for a GitHub push)

| Piece | License | Where |
|-------|---------|--------|
| This jllama tree (J scripts, docs) | MIT | `LICENSE` |
| J engine / jsource (including GPU glue in `jsrc/gpu*.c`) | Jsoftware commercial **or** GPL-3 | `jsource/LICENSE`, `LICENSE-GPL3` |
| ggml | MIT | `jsource/third_party/ggml.LICENSE` |

Details: `THIRD_PARTY.md`. Shipping `libj.dylib` is a jsource distribution, not covered by this repo's MIT alone.
