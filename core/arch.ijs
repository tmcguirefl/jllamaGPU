NB. jllama architecture scripts as nouns (M14)
NB.
NB. Llama3, Qwen35, and Phi4Mini are 0 : 0 character nouns (multiline scripts).
NB. A 0 : 0 noun is not a single J sentence, so ". (sentence-do) cannot
NB. swallow the LFs. Script-do 0!:0 parses the noun into working memory.
NB. The CLI still uses the do verb to build that sentence:
NB.   ". '0!:0 ' , detect_arch 'models/Qwen3.5-2B.gguf'
NB. which is  ". '0!:0 Qwen35'  i.e. do(script-do Qwen35).
NB.
NB. REPL:
NB.   0!:0 Llama3
NB.   0!:0 Qwen35
NB.   0!:0 Phi4Mini
NB.   ". '0!:0 Phi4Mini'
NB.
NB. Detection is from the .gguf *filename* (not GGUF metadata).

cocurrent 'jllamaarch'

NB. ---------------------------------------------------------------
NB. Filename -> architecture noun name
NB. ---------------------------------------------------------------
arch_lc =: 3 : 0
  a =. a. i. , y
  up =. (a >: 65) *. a <: 90
  (a + 32 * up) { a.
)

arch_base =: 3 : 0
  p =. , y
  if. '/' e. p do. p =. (1 + p i: '/') }. p end.
  if. '\' e. p do. p =. (1 + p i: '\') }. p end.
  p
)

NB. y = GGUF path or filename -> 'Llama3' or 'Qwen35' or 'Phi4Mini'
detect_arch =: 3 : 0
  p =. arch_lc arch_base y
  if. +./ 'qwen3.5' E. p do. 'Qwen35' return. end.
  if. +./ 'qwen-3.5' E. p do. 'Qwen35' return. end.
  if. +./ 'qwen3_5' E. p do. 'Qwen35' return. end.
  if. +./ 'qwen35' E. p do. 'Qwen35' return. end.
  if. +./ 'phi-4-mini' E. p do. 'Phi4Mini' return. end.
  if. +./ 'phi4-mini' E. p do. 'Phi4Mini' return. end.
  if. +./ 'phi4mini' E. p do. 'Phi4Mini' return. end.
  if. +./ 'phi_4_mini' E. p do. 'Phi4Mini' return. end.
  if. +./ 'phi-4' E. p do. 'Phi4Mini' return. end.
  if. +./ 'phi4' E. p do. 'Phi4Mini' return. end.
  if. +./ 'phi3' E. p do. 'Phi4Mini' return. end.
  'Llama3'
)

NB. y = 'Llama3' or 'Qwen35'
NB. ". builds the sentence  0!:0 NAME  then do runs it.
apply_arch =: 3 : '". ''0!:0 '' , y'

NB. y = GGUF path: detect from filename and install that architecture.
load_arch_file =: 3 : 0
  apply_arch detect_arch y
  i. 0 0
)

NB. ---------------------------------------------------------------
NB. Llama3 — Llama dense MHA/GQA graph + GGUF loader
NB. Bring in with:  0!:0 Llama3   or   ". '0!:0 Llama3'
NB. ---------------------------------------------------------------
Llama3 =: 0 : 0
jllama_arch_prev_z_ =: 18!:5 ''
model_from_gguf_jllamagguf_ =: model_from_gguf_llama_jllamagguf_
block_full_jllamamodel_ =: block_full_jllamablock_
block_step_jllamamodel_ =: block_step_jllamablock_
block_prefill_cached_jllamamodel_ =: block_prefill_cached_jllamablock_
RMS_EPS_jllamamodel_ =: 1e_5
RMS_EPS_jllamablock_ =: 1e_5
RMS_EPS_jllamaattn_ =: 1e_5
cocurrent jllama_arch_prev_z_
)

NB. ---------------------------------------------------------------
NB. Qwen35 — hybrid gated-attention + Gated DeltaNet + GGUF loader
NB. Bring in with:  0!:0 Qwen35   or   ". '0!:0 Qwen35'
NB. ---------------------------------------------------------------
Qwen35 =: 0 : 0
jllama_arch_prev_z_ =: 18!:5 ''
load ROOT_jllamasys_ , 'core/gdn.ijs'
load ROOT_jllamasys_ , 'core/qwen35.ijs'
model_from_gguf_jllamagguf_ =: model_from_gguf_qwen_jllamagguf_
block_full_jllamamodel_ =: block_full_jllamaqwen_
block_step_jllamamodel_ =: block_step_jllamaqwen_
block_prefill_cached_jllamamodel_ =: block_prefill_cached_jllamaqwen_
cocurrent jllama_arch_prev_z_
)

NB. ---------------------------------------------------------------
NB. Phi4Mini — Phi-4-mini / phi3 dense GQA + NeoX RoPE + fused SwiGLU
NB. Bring in with:  0!:0 Phi4Mini   or   ". '0!:0 Phi4Mini'
NB. ---------------------------------------------------------------
Phi4Mini =: 0 : 0
jllama_arch_prev_z_ =: 18!:5 ''
load ROOT_jllamasys_ , 'core/phi4mini.ijs'
model_from_gguf_jllamagguf_ =: model_from_gguf_phi_jllamagguf_
block_full_jllamamodel_ =: block_full_jllamaphi_
block_step_jllamamodel_ =: block_step_jllamaphi_
block_prefill_cached_jllamamodel_ =: block_prefill_cached_jllamaphi_
cocurrent jllama_arch_prev_z_
)

NB. Publish on z so  0!:0 Llama3  works from any locale.
Llama3_z_ =: Llama3
Qwen35_z_ =: Qwen35
Phi4Mini_z_ =: Phi4Mini
detect_arch_z_ =: detect_arch
apply_arch_z_ =: apply_arch
load_arch_file_z_ =: load_arch_file

cocurrent 'base'
