# GPU J engine guide — rewrite jllama on ggml

This tree is **jllamaGPU** (it replaces the old `jllama_dev` directory). The J 9.8 runtime on this machine already has the GPU `j.dll`. Start here; do not invent a second engine.

**Session goal:** keep tokens/embeddings on the CPU as needed, keep **weights and activations on the GPU**, and implement a small LLM (TinyStories / stories15M, then Llama-scale) using the verbs below. CPU transfer is for viewing, tokenize, and sample — not for every matmul.

---

## 1. Runtime (read this first)

| What | Path |
|------|------|
| J console | `C:\Users\tmcguire\j9.8\bin\jconsole.exe` |
| Engine | `C:\Users\tmcguire\j9.8\bin\j.dll` (gpu-resident, Windows Vulkan) |
| ggml DLLs | same `bin/` (`ggml.dll`, `ggml-base.dll`, `ggml-cpu.dll`, `ggml-vulkan.dll`) |
| Stock J backup | `C:\Users\tmcguire\j9.8\bin\j.dll.jsoftware-beta7` |
| Engine source | `C:\Users\tmcguire\jdev\jsource` branch `gpu-resident` |
| ggml sources | sibling `C:\Users\tmcguire\jdev\llama.cpp\ggml` |
| Named wrappers | `jgpu.ijs` in this directory |
| Engine tests | `C:\Users\tmcguire\jdev\jsource\gpu-phase0\test_phase{1..5}.ijs` |

Never the Java `jconsole`. Use the full path to J's `jconsole.exe`.

Do **not** pass `-lib` unless you are testing a freshly built `jsource` `j.dll`. Default jconsole already loads the GPU engine.

Check:

```
"C:\Users\tmcguire\j9.8\bin\jconsole.exe"
   9!:14''
```

Expect `j9.8.0-beta7` and `j64avx2/windows`.

Working type on GPU: **F32**. J `FL` on CPU is IEEE double. Upload converts; download of F32 is exact enough for F32; vs F64 expect ~1e-3 on GEMM, ~1e-5 on unary.

Display of a GPU noun does **not** densify. `8!:0 y` is a one-line summary like `gpu q4_0 4096 atoms, 2304 bytes`.

---

## 2. What a GPU noun is

GPU nouns are a first-class type (`3!:0` is `524288`). Classic sparse is unused in this fork.

- `2 G. y` → ggml type id (`0` F32, `1` F16, `2` Q4_0, `6` Q5_0, `8` Q8_0, `12` Q4_K, …)
- `4 G. y` → `nbytes , natom` (packed size, not `4 * natom` for quants)
- `$ y` and `# y` are the J shape (last axis = **K** for quantized weights)

```j
g =. G. x          NB. dense numeric → GPU F32 (new cookie)
m =. G: x          NB. same upload, mutable; later m=. kernel y reuses m
h =. G.^:_1 g      NB. download to FL (Q4/F16 dequant to float)
1 G. m             NB. 1 iff mutable
```

Quantized nouns stay **packed**. Do not dequant at load. The only v1 consumer of a packed quant noun is **left argument of** `+/ .*` (and download for debugging).

---

## 3. Verb table (what the rewrite should call)

### 3.1 Identity / inspect

| Phrase | Meaning |
|--------|---------|
| `G. y` | upload dense B01/INT/FL → GPU F32 (new buffer) |
| `G: y` | upload (or copy) to a **mutable** GPU noun; assignment in explicit defs writes that buffer |
| `1 G. y` | 1 iff `y` is a `G:` noun |
| `G.^:_1 y` / `0 G. y` | download; quants dequant to FL |
| `2 G. y` | ggml type id |
| `4 G. y` | packed `nbytes , natom` |
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
| `'silu' g. x` | SiLU, any rank, F32 |
| `'softmax' g. x` | softmax on **last** axis |
| `w ('rmsnorm' g.) x` | RMSNorm, eps=`1e_5`; `w` gain length = last axis of `x` |
| `'rmsnorm' g. x` | RMSNorm without gain |
| `'rope' g. x` | RoPE, `pos = i. n_tok`, θ=10000, NORMAL, full last axis |
| `pos ('rope' g.) x` | RoPE; `x` rank 2 `n_tok × d` or rank 3 `n_tok × n_head × d` |
| `(pos;theta;n_rot;mode) ('rope' g.) x` | `mode` 0 = Llama NORMAL, 2 = NeoX; `n_rot` even, `≤ d_head` |

Dense numeric args to these kernels are auto-uploaded; result is always GPU F32. `g.` is an adverb: `'silu' g.` is a verb.

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
| `'gguf' g. 'file.gguf'` | boxed table shape `(#tensors, 2)` of `(name ; GPU noun)` |
| `('quant' g.) W` | dense → **Q4_0** GPU; last axis `% 32` |
| `tid ('quant' g.) W` | dense → ggml type `tid`. `Q8_0 ('quant' g.) W` or `8 ('quant' g.) W` |

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
- upload `|:` of a CPU `n_in × n_out` **before** `('quant' g.)` so the packed last axis is `n_in`.

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
| `core/tensor.ijs` `silu` `softmax` `rmsnorm` | bind to `'silu' g.` / `'softmax' g.` / `'rmsnorm' g.` (or `*_jgpu_`) when args are GPU |
| `linear` / every `+/ . *` on weights | `linear_jgpu_` / `|: W +/ .* |: x` |
| `causal_mask` | stay dense FL; `scores + mask` auto-uploads mask |
| `core/rope.ijs` | `pos ('rope' g.) x` (NORMAL). NeoX: `(pos;theta;n_rot;2) ('rope' g.) x` |
| `split_heads` / `merge_heads` | already `$ ,` — works on GPU F32 |
| `kv_empty` | `((0, n_kv, d)$0)` dense empty; `kc , K` starts the cache |
| `expand_kv` | keep `1 0 2 \|: idx { 1 0 2 \|: y` |
| `attention_heads` box loop | **do not** `heads , < …` then `> heads`. Open of a list of GPU boxes will not assemble one array. Cat items: `O =. ((0, n_q, dh)$0)` then `O =. O , attention1 …` then `1 0 2 \|: O` |
| `mha_step` `kc , K` | works on GPU |
| `,: xv` | works |
| `io/gguf.ijs` | either call `'gguf' g.` and keep packed GPU nouns, or keep the J parser for **metadata/tokenizer only** and take tensor bytes from `'gguf' g.`. Do not promote weight tensors to f64. |
| `ffn_swiglu` / `block_*` | GPU silu + linear + `+` `*` |

Tokenizer, sampling, GGUF **metadata**, and token ids stay CPU.

---

## 6. What is still nonce / illegal

- Classic sparse
- GPU `+/ .*` with quantized **right** arg (right must be F32)
- `|:` of a quantized noun
- `>` of a **list** of boxed GPU arrays (scalar box `> <g` is fine)
- IQ-type `('quant' g.)` (need imatrix). GGUF **load** of those types still stores packed data; matmul may work if Metal has the kernel
- Reshape that changes atom count (no fill)
- Mixed GPU + classic-sparse
- Most other primitives (`^` `%.` `+/` on GPU, etc.) — use the named kernels

If a verb  nonces, the noun is probably still GPU; download only to debug.

---

## 7. Suggested first slice

1. Smoke: `G.` roundtrip, `'gguf' g.` on `test/fixtures/tiny_parity_f16.gguf` or jsource `gpu-phase0/tiny_q4.gguf`.
2. Replace `silu` `softmax` `rmsnorm` `rope` with `'silu' g.` etc.; keep CPU fallbacks for dense.
3. GGUF weights: packed GPU nouns, shape `n_out × n_in`.
4. FFN + RMSNorm + residual on GPU (SwiGLU).
5. Attention: split/merge + `1 0 2 \|:` + per-head `{` + cat (no `> heads`) + `scores % %: d` + dense causal mask.
6. KV decode: empty `,` append.
7. TinyStories `stories15M.F16.gguf` — GGUFs live in this tree's `models/` (gitignored).

Engine self-tests:

```
"C:\Users\tmcguire\j9.8\bin\jconsole.exe" C:\Users\tmcguire\jdev\jsource\gpu-phase0\test_phase3.ijs
"C:\Users\tmcguire\j9.8\bin\jconsole.exe" C:\Users\tmcguire\jdev\jsource\gpu-phase0\test_phase4.ijs
"C:\Users\tmcguire\j9.8\bin\jconsole.exe" C:\Users\tmcguire\jdev\jsource\gpu-phase0\test_phase5.ijs
```

---

## 8. Engine internals (only if libj must change)

Source: `C:\Users\tmcguire\jdev\jsource` on branch `gpu-resident`.

- `jsrc/gpu.c` — J-facing (includes `j.h`, talks only to `gpu_api.h`)
- `jsrc/gpu_ggml.c` — **never** include `j.h`; ggml stays out of JE headers
- Rebuild (Windows Vulkan): `third_party\build_ggml.ps1`, then `makemsvc\jdll`. Copy `j.dll` and `ggml*.dll` from `bin\windows\j64avx2` into `C:\Users\tmcguire\j9.8\bin\`
- Details: `jsource/gpu-phase0/DESIGN_GPU_PLATFORMS.md`

GPU nouns are type `GPU` (`3!:0` 524288), not sparse. `G.` constructs them; `g.` names kernels.

---

## 9. Boxing packs (still true)

jllama `docs/conventions.md`: chained `x ; y ; z` re-boxes. For mixed GPU nouns + integers use:

```j
(<x) , (<n_head) , layerbox , (<theta)
```

GPU nouns **may** be boxed (`<Wq` works).

---

## 10. Licenses (for a GitHub push)

This program depends on a **fork of the J runtime**, so **jllamaGPU is
GPL-3.0-only** — the same option this project uses for Jsoftware jsource.

| Piece | License | Where |
|-------|---------|--------|
| This tree (J scripts, docs) | GPL-3.0-only | `LICENSE`, `COPYRIGHT` |
| J engine / jsource (including `jsrc/gpu*.c`) | Jsoftware commercial **or** GPL-3; **this project uses GPL-3** | `jsource/LICENSE`, `LICENSE-GPL3` |
| ggml | MIT (GPL-3 compatible) | `jsource/third_party/ggml.LICENSE` |

Details: `THIRD_PARTY.md`. A release that ships `j.dll` / `libj` must include
GPL-3 source for the engine fork as well as this application.
