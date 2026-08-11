NB. jllama transformer block + SwiGLU FFN (M3)
NB.
NB. Llama-style pre-norm block:
NB.   x = x + MHA( rmsnorm(x) )
NB.   x = x + FFN( rmsnorm(x) )
NB.
NB. FFN = SwiGLU:
NB.   (silu(x mp w_gate) * (x mp w_up)) mp w_down
NB.
NB. Layer is ONE scalar box of 9 weights:
NB.   <"_ (attn_norm ; wq ; wk ; wv ; wo ; ffn_norm ; w_gate ; w_up ; w_down)
NB.
NB. Arg packing rule (critical):
NB.   Chained  x ; y ; z  RE-BOXES when the left is already a box list.
NB.   For mixed nested boxes + numerics, pack with catenate of scalar boxes:
NB.     (<x) , (<n_head) , layerbox , (<theta)
NB.   Pure-numeric packs may still use  ;
NB.
NB. Load order: tensor, rope, attention, block

cocurrent 'jllamablock'

mp =: mp_jllamatensor_
silu =: silu_jllamatensor_
rmsnorm =: rmsnorm_jllamatensor_
mha_full =: mha_full_jllamaattn_
mha_step =: mha_step_jllamaattn_
kv_empty =: kv_empty_jllamaattn_
DEFAULT_THETA =: DEFAULT_THETA_jllamarope_

NB. ---------------------------------------------------------------
NB. SwiGLU FFN
NB. y = x ; w_gate ; w_up ; w_down   (all numeric - ; is fine)
NB. ---------------------------------------------------------------
ffn_swiglu =: 3 : 0
  'xv wg wu wd' =. y
  h =. (silu xv mp wg) * (xv mp wu)
  h mp wd
)

NB. ---------------------------------------------------------------
NB. One block, full sequence (no KV)
NB. y = (<x) , (<n_head) , layerbox [ , (<theta) ]
NB. ---------------------------------------------------------------
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

NB. ---------------------------------------------------------------
NB. One block, single token + KV cache
NB. y = (<x1),(<n_head),layerbox,(<kc),(<vc),(<pos)[,(<theta)]
NB. returns (<out1) , (<kc2) , (<vc2)
NB. ---------------------------------------------------------------
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

NB. Prefill one block via successive steps
NB. y = (<x) , (<n_head) , layerbox [ , (<theta) ]
NB. returns (<outs) , (<kc) , (<vc)
block_prefill_cached =: 3 : 0
  if. 4 = # y do.
    'xv n_head layer theta' =. y
  else.
    'xv n_head layer' =. y
    theta =. DEFAULT_THETA
  end.
  NB. layer is open 9-list after assign; re-box as one scalar box
  layerbox =. <"_ layer
  n_embd =. {: $ xv
  d_head =. n_embd % n_head
  'kc vc' =. kv_empty n_head , d_head
  outs =. (0 , n_embd) $ 0
  for_t. i. # xv do.
    x1 =. ,: t { xv
    'o1 kc vc' =. block_step (<x1) , (<n_head) , layerbox , (<kc) , (<vc) , (<t) , (<theta)
    outs =. outs , o1
  end.
  (<outs) , (<kc) , (<vc)
)

cocurrent 'base'
