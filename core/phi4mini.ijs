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

load jpath '~temp/jllama/core/tensor.ijs'
split_heads =: split_heads_jllamaattn_
merge_heads =: merge_heads_jllamaattn_
attention_heads =: attention_heads_jllamaattn_
kv_empty =: kv_empty_jllamaattn_
ffn_swiglu =: ffn_swiglu_jllamablock_
rope_neox =: rope_neox_jllamarope_
DEFAULT_THETA =: DEFAULT_THETA_jllamarope_

NB. pos or (pos;theta) apply_rope_qk Q;K
apply_rope_qk =: 4 : 0
  'Q K' =. y
  (x rope_neox Q) ; (x rope_neox K)
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
  n_kv =. ({: $ wk) % d_head
  Q =. n_head split_heads xv +/ . * wq
  K =. n_kv split_heads xv +/ . * wk
  V =. n_kv split_heads xv +/ . * wv
  pos =. i. # xv
  'Q K' =. (pos ; theta) apply_rope_qk Q ; K
  O =. attention_heads Q ; K ; V
  (merge_heads O) +/ . * wo
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
  n_kv =. ({: $ wk) % d_head
  Q =. n_head split_heads xv +/ . * wq
  K =. n_kv split_heads xv +/ . * wk
  V =. n_kv split_heads xv +/ . * wv
  'Q K' =. ((, pos) ; theta) apply_rope_qk Q ; K
  kc =. kc , K
  vc =. vc , V
  O =. attention_heads Q ; kc ; vc
  out =. (merge_heads O) +/ . * wo
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

block_prefill_cached =: 3 : 0
  if. 4 = # y do.
    'xv n_head layer theta' =. y
  else.
    'xv n_head layer' =. y
    theta =. DEFAULT_THETA
  end.
  layerbox =. <"_ layer
  n_embd =. {: $ xv
  d_head =. n_embd % n_head
  wk =. 2 { layer
  n_kv =. ({: $ wk) % d_head
  'kc vc' =. kv_empty n_kv , d_head
  outs =. (0 , n_embd) $ 0
  for_t. i. # xv do.
    x1 =. ,: t { xv
    'o1 kc vc' =. block_step (<x1) , (<n_head) , layerbox , (<kc) , (<vc) , (<t) , (<theta)
    outs =. outs , o1
  end.
  (<outs) , (<kc) , (<vc)
)

cocurrent 'base'
