NB. jllama Phi-4-mini (GGUF arch=phi3) dense decoder
NB.
NB. Locale: jllamaphi
NB. Installed by  0!:0 Phi4Mini  (core/arch.ijs).
NB.
NB. Same residual skeleton as Llama:
NB.   x = x + Attn( rmsnorm(x) )
NB.   x = x + FFN( rmsnorm(x) )
NB. Differences vs Llama3:
NB.   RoPE is NeoX half-rotate (not pair-wise NORMAL)
NB.   FFN up is often fused (n_embd x 2*n_ff) then split to gate/up
NB.   QKV may be a single attn_qkv.weight
NB. Layer pack matches Llama so generate_jllamamodel_ is unchanged:
NB.   <"_ (attn_n ; wq ; wk ; wv ; wo ; ffn_n ; wg ; wu ; wd)

cocurrent 'jllamaphi'

load ROOT_jllamasys_ , 'core/tensor.ijs'
silu =: silu_jgpu_
softmax =: softmax_jgpu_
rmsnorm =: rmsnorm_jgpu_
linear =: linear_jgpu_
rope =: rope_jgpu_
split_heads =: split_heads_jllamaattn_
merge_heads =: merge_heads_jllamaattn_
attention_heads =: attention_heads_jllamaattn_
kv_empty =: kv_empty_jllamaattn_
ffn_swiglu =: ffn_swiglu_jllamablock_
DEFAULT_THETA =: DEFAULT_THETA_jllamarope_

NB. pos or (pos;theta) apply_rope_qk Q;K  — NeoX mode 2, n_rot from GGUF (Phi-4-mini is 96 of 128)
apply_rope_qk =: 4 : 0
  'Q K' =. y
  d =. {: $ Q
  n_rot =. d
  if. 0 = 4!:0 <'N_ROT' do. n_rot =. N_ROT end.
  if. 32 = 3!:0 x do.
    pad =. x
    if. 1 = # pad do. pad =. pad , < DEFAULT_THETA end.
    'pos theta' =. 2 {. pad
  else.
    pos =. x
    theta =. DEFAULT_THETA
  end.
  spec =. (<, pos) , (<theta) , (<n_rot) , (<2)
  (spec rope Q) ; (spec rope K)
)

NB. y = x ; n_head ; wq ; wk ; wv ; wo [; theta]
mha_full =: 3 : 0
  if. 7 = # y do.
    'xv n_head wq wk wv wo theta' =. y
  else.
    'xv n_head wq wk wv wo' =. y
    theta =. DEFAULT_THETA
  end.
  n_embd =. {: $ xv
  d_head =. n_embd % n_head
  n_kv =. ({. $ wk) % d_head
  Q =. n_head split_heads xv linear wq
  K =. n_kv split_heads xv linear wk
  V =. n_kv split_heads xv linear wv
  pos =. i. # xv
  'Q K' =. (pos ; theta) apply_rope_qk Q ; K
  O =. attention_heads Q ; K ; V
  (merge_heads O) linear wo
)

NB. y = x1 ; n_head ; wq ; wk ; wv ; wo ; kc ; vc ; pos [; theta]
mha_step =: 3 : 0
  if. 10 = # y do.
    'xv n_head wq wk wv wo kc vc pos theta' =. y
  else.
    'xv n_head wq wk wv wo kc vc pos' =. y
    theta =. DEFAULT_THETA
  end.
  if. 1 = #$ xv do. xv =. ,: xv end.
  n_embd =. {: $ xv
  d_head =. n_embd % n_head
  n_kv =. ({. $ wk) % d_head
  Q =. n_head split_heads xv linear wq
  K =. n_kv split_heads xv linear wk
  V =. n_kv split_heads xv linear wv
  'Q K' =. ((, pos) ; theta) apply_rope_qk Q ; K
  kc =. kc , K
  vc =. vc , V
  O =. attention_heads Q ; kc ; vc
  out =. (merge_heads O) linear wo
  out ; kc ; vc
)

block_full =: 3 : 0
  if. 4 = # y do.
    'xv n_head layer theta' =. y
  else.
    'xv n_head layer' =. y
    theta =. DEFAULT_THETA
  end.
  'attn_n wq wk wv wo ffn_n wg wu wd' =. layer
  h =. attn_n rmsnorm xv
  h =. xv + mha_full h ; n_head ; wq ; wk ; wv ; wo ; theta
  r =. ffn_n rmsnorm h
  h + ffn_swiglu r ; wg ; wu ; wd
)

block_step =: 3 : 0
  if. 7 = # y do.
    'xv n_head layer kc vc pos theta' =. y
  else.
    'xv n_head layer kc vc pos' =. y
    theta =. DEFAULT_THETA
  end.
  if. 1 = #$ xv do. xv =. ,: xv end.
  'attn_n wq wk wv wo ffn_n wg wu wd' =. layer
  h =. attn_n rmsnorm xv
  'a kc vc' =. mha_step h ; n_head ; wq ; wk ; wv ; wo ; kc ; vc ; pos ; theta
  h =. xv + a
  r =. ffn_n rmsnorm h
  out =. h + ffn_swiglu r ; wg ; wu ; wd
  (<out) , (<kc) , (<vc)
)

NB. Full-sequence MHA that also returns K/V cache (NeoX RoPE).
mha_prefill =: 3 : 0
  if. 7 = # y do.
    'xv n_head wq wk wv wo theta' =. y
  else.
    'xv n_head wq wk wv wo' =. y
    theta =. DEFAULT_THETA
  end.
  if. 1 = #$ xv do. xv =. ,: xv end.
  n_embd =. {: $ xv
  d_head =. n_embd % n_head
  n_kv =. ({. $ wk) % d_head
  Q =. n_head split_heads xv linear wq
  K =. n_kv split_heads xv linear wk
  V =. n_kv split_heads xv linear wv
  pos =. i. # xv
  'Q K' =. (pos ; theta) apply_rope_qk Q ; K
  O =. attention_heads Q ; K ; V
  out =. (merge_heads O) linear wo
  out ; K ; V
)

NB. Prefill one block as a full sequence (not per-token step).
block_prefill_cached =: 3 : 0
  if. 4 = # y do.
    'xv n_head layer theta' =. y
  else.
    'xv n_head layer' =. y
    theta =. DEFAULT_THETA
  end.
  if. 1 = #$ xv do. xv =. ,: xv end.
  'attn_n wq wk wv wo ffn_n wg wu wd' =. layer
  h =. attn_n rmsnorm xv
  'a kc vc' =. mha_prefill h ; n_head ; wq ; wk ; wv ; wo ; theta
  h =. xv + a
  r =. ffn_n rmsnorm h
  out =. h + ffn_swiglu r ; wg ; wu ; wd
  (<out) , (<kc) , (<vc)
)

cocurrent 'base'
