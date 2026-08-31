NB. jllamaGPU — GNU GPL v3 only. Copyright (C) 2026 Tom McGuire. See LICENSE.
NB. Named GPU kernels for a small LLM on this engine.
NB. Runtime: "C:\Users\tmcguire\j9.8\bin\jconsole.exe"  (GPU libj already installed).
NB.   load 'jgpu.ijs'
NB. Full rewrite notes: GPU_ENGINE.md
NB.
NB. GGUF weights are n_out x n_in (last axis = K, quantized).
NB. Activations in jllama are n_tok x n_in. Linear is:
NB.   y =. |: W +/ .* |: x
NB. F32 + * - % |: , { $ and ,: stay on device.
NB. RMSNorm/SiLU/softmax/RoPE are 'silu' g. / 'softmax' g. / ...
NB. Attention: 1 0 2 |:  (head permute), empty , K  (KV cache),
NB.   scores + mask and scores % %: d  (dense mixed).
NB. Nouns: G. upload, G.^:_1 download, 2 G. type, 4 G. nbytes,natom.

cocurrent 'jgpu'

silu =: 'silu' g.
softmax =: 'softmax' g.
rmsnorm =: 'rmsnorm' g.
rope =: 'rope' g.
add =: +
mul =: *
gguf_load =: 'gguf' g.
quantize =: 'quant' g.

NB. ggml type ids (left of +/ .* or tid quantize y)
F32 =: 0
F16 =: 1
Q4_0 =: 2
Q4_1 =: 3
Q5_0 =: 6
Q5_1 =: 7
Q8_0 =: 8
Q4_K =: 12
Q5_K =: 13
Q6_K =: 14

NB. x linear W  — x is n_tok x n_in, W is n_out x n_in (GGUF / quantized).
NB. Engine +/ .* already maps J axes to ggml; |: here is J shape so W stays
NB. LEFT (last axis = K). Do not |: a packed weight.
linear =: 4 : '|: y +/ .* |: x'

NB. Llama-style SwiGLU. y = x ; Wg ; Wu ; Wd  (W* are n_out x n_in)
swiglu =: 3 : 0
  'xv wg wu wd' =. y
  h =. (silu xv linear wg) * (xv linear wu)
  h linear wd
)

NB. Embeddings and 1-d gains must be F32 for { and rmsnorm. Load-time only.
asf32 =: G. @: (G.^:_1)

cocurrent 'base'
