NB. jllama GPT-OSS (GGUF arch=gpt-oss) MoE decoder with G: scratch
NB.
NB. Locale: jllamagptoss
NB. Installed by  0!:0 GptOss  (core/arch.ijs).
NB.
NB. Per block:
NB.   x = x + Attn(rmsnorm(x))     GQA + NeoX RoPE + attention sinks
NB.                                even layers SWA (window), odd layers full
NB.   x = x + MoE(post_norm(x))    top-k then softmax; clamped SiLU-GLU
NB. Every projection has a bias. Expert weights stay packed (MXFP4).
NB.
NB. Layer box:
NB.   kind; attn_n; wq;bq; wk;bk; wv;bv; wo;bo; sinks;
NB.   post_n; wrout;brout; wge;bge; wue;bue; wde;bde
NB. kind is 'swa' or 'full'.
NB.
NB. GHP (set at GGUF load):
NB.   n_head ; n_kv ; d_head ; n_rot ; theta ; eps ;
NB.   n_exp ; n_used ; n_ff ; window

cocurrent 'jllamagptoss'

load ROOT_jllamasys_ , 'core/tensor.ijs'
silu =: silu_jgpu_
softmax =: softmax_jgpu_
rmsnorm =: rmsnorm_jgpu_
linear =: linear_jgpu_
rope =: rope_jgpu_
split_heads =: split_heads_jllamaattn_
merge_heads =: merge_heads_jllamaattn_
expand_kv =: expand_kv_jllamaattn_
kv_empty =: kv_empty_jllamaattn_

GHP =: 64 ; 8 ; 64 ; 64 ; 150000 ; 1e_5 ; 32 ; 4 ; 2880 ; 128
NT =: 0
NEMBD =: 0
SILU_A =: 1.702
SILU_LIM =: 7

gs =: 3 : 'G: y $ 0'

scratch_fit =: 3 : 0
  nt =. {. y
  ne =. {: y
  if. (nt = NT) *. (ne = NEMBD) *. 0 ~: NT do. i. 0 0 return. end.
  'nH n_kv d_head n_rot th eps n_exp n_used n_ff win' =. GHP
  H =: gs nt , ne
  A =: gs nt , ne
  FFOUT =: gs nt , ne
  RN =: gs nt , ne
  QH =: gs nt , nH , d_head
  KH =: gs nt , n_kv , d_head
  VH =: gs nt , n_kv , d_head
  OH =: gs nt , nH , d_head
  GATE =: gs nt , n_ff
  UP =: gs nt , n_ff
  NT =: nt
  NEMBD =: ne
  smoutput 'jllama: G: scratch ' , (": nt) , ' x ' , ": ne
  i. 0 0
)

addb =: 4 : 0
  if. 0 -: y do. x else. x +"1 y end.
)

NB. x=window (0=full causal); y=nq;nk;pos0 -> nq x nk mask (CPU, then G.)
attn_mask =: 4 : 0
  win =. x
  'nq nk pos0' =. y
  qpos =. pos0 + i. nq
  kpos =. i. nk
  caus =. kpos >"1 0 qpos
  if. win > 0 do.
    caus =. caus +. win <: qpos -"0 1 kpos
  end.
  MASK_VAL * caus
)

NB. x=sinks (n_head); y=scores (n_head x nq x nk). Host softmax; scores stay small.
NB. den is n_head x nq; %"1 0 divides each key-row by that scalar (decode nq=1).
softmax_sinks =: 4 : 0
  sc =. G.^:_1 y
  sk =. , G.^:_1 x
  nq =. 1 { $ sc
  skt =. nq #"0 sk
  mx =. (>./"1 sc) >. skt
  ex =. ^ sc - mx
  es =. ^ skt - mx
  G. ex %"1 0 es + +/"1 ex
)

NB. y = xv; wq;bq; wk;bk; wv;bv; wo;bo; sinks; nH; n_kv; d_head; n_rot; theta; window; pos0 [; kc; vc]
NB. without cache: returns out
NB. with cache:    returns out ; kc ; vc
gptoss_mha =: 3 : 0
  cached =. 18 <: # y
  if. cached do.
    'xv wq bq wk bk wv bv wo bo sinks nH n_kv d_head n_rot theta win pos0 kc vc' =. y
  else.
    'xv wq bq wk bk wv bv wo bo sinks nH n_kv d_head n_rot theta win pos0' =. y
    kc =. 0 $ 0
    vc =. 0 $ 0
  end.
  if. 1 = #$ xv do. xv =. ,: xv end.
  n_tok =. # xv
  scratch_fit n_tok , {: $ xv
  Q =. QH
  Q =. nH split_heads (xv linear wq) addb bq
  K =. KH
  K =. n_kv split_heads (xv linear wk) addb bk
  V =. VH
  V =. n_kv split_heads (xv linear wv) addb bv
  spec =. (<, pos0 + i. n_tok) , (<theta) , (<n_rot) , (<2)
  Q =. spec rope Q
  K =. spec rope K
  if. cached do.
    if. 0 = # , kc do.
      kc =. K
      vc =. V
    else.
      kc =. kc , K
      vc =. vc , V
    end.
    Kuse =. kc
    Vuse =. vc
  else.
    Kuse =. K
    Vuse =. V
  end.
  n_rep =. nH % n_kv
  if. n_rep > 1 do.
    Kuse =. n_rep expand_kv Kuse
    Vuse =. n_rep expand_kv Vuse
  end.
  nq =. # Q
  nk =. # Kuse
  dh =. {: $ Q
  qb =. 1 0 2 |: Q
  kb =. 1 0 2 |: Kuse
  vb =. 1 0 2 |: Vuse
  scores =. (qb +/ . * 0 2 1 |: kb) % %: dh
  if. (nq > 1) +. win > 0 do.
    scores =. scores + G. win attn_mask nq ; nk ; pos0
  elseif. nq = nk do.
    if. nq > 1 do. scores =. scores + G. 0 attn_mask nq ; nk ; pos0 end.
  end.
  O =. OH
  O =. 1 0 2 |: (sinks softmax_sinks scores) +/ . * vb
  out =. A
  out =. ((merge_heads O) linear wo) addb bo
  if. cached do. out ; kc ; vc else. out end.
)

NB. Clamped GPT-OSS GLU: clamp gate (-inf,7], up [-7,7],  gate*sigmoid(1.702*gate)*(up+1)
oai_glu =: 3 : 0
  'ge up' =. y
  ge =. (G. SILU_LIM) <. ge
  up =. (G. SILU_LIM) <. (G. - SILU_LIM) >. up
  (ge * 1 % 1 + ^ - ge * G. SILU_A) * up + 1
)

NB. Per-token top-k MoE. Expert weights stay packed; n=1 GEMV.
gptoss_moe =: 3 : 0
  'r wrout brout wge bge wue bue wde bde n_used' =. y
  if. 1 = #$ r do. r =. ,: r end.
  nt =. # r
  ne =. {: $ r
  lg =. G.^:_1 (r linear wrout) addb brout
  acc =. (0 , ne) $ 0
  for_t. i. nt do.
    row =. t { lg
    sel =. n_used {. \: row
    sv =. sel { row
    smx =. >./ sv
    sw =. (^ sv - smx) % +/ ^ sv - smx
    xt =. ,: t { r
    ye =. 0
    for_j. i. n_used do.
      e =. j { sel
      w =. j { sw
      ge =. GATE
      ge =. (xt linear (e { wge)) addb (e { bge)
      up =. UP
      up =. (xt linear (e { wue)) addb (e { bue)
      h =. oai_glu ge ; up
      y1 =. (h linear (e { wde)) addb (e { bde)
      if. 0 -: ye do. ye =. y1 * w else. ye =. ye + y1 * w end.
    end.
    acc =. acc , ye
  end.
  acc
)

block_full =: 3 : 0
  if. 4 = # y do.
    'xv n_head layer theta' =. y
  else.
    'xv n_head layer' =. y
  end.
  if. 1 = #$ xv do. xv =. ,: xv end.
  scratch_fit (# xv) , {: $ xv
  'nH n_kv d_head n_rot th eps n_exp n_used n_ff win' =. GHP
  'kind attn_n wq bq wk bk wv bv wo bo sinks post_n wrout brout wge bge wue bue wde bde' =. layer
  if. 32 = 3!:0 kind do. kind =. > kind end.
  wuse =. win
  if. -. kind -: 'swa' do. wuse =. 0 end.
  h =. H
  h =. attn_n rmsnorm xv
  a =. gptoss_mha h ; wq ; bq ; wk ; bk ; wv ; bv ; wo ; bo ; sinks ; nH ; n_kv ; d_head ; n_rot ; th ; wuse ; 0
  h2 =. xv + a
  r =. RN
  r =. post_n rmsnorm h2
  out =. FFOUT
  out =. h2 + gptoss_moe r ; wrout ; brout ; wge ; bge ; wue ; bue ; wde ; bde ; n_used
  out
)

block_step =: 3 : 0
  if. 7 = # y do.
    'xv n_head layer kc vc pos theta' =. y
  else.
    'xv n_head layer kc vc pos' =. y
  end.
  if. 1 = #$ xv do. xv =. ,: xv end.
  scratch_fit (# xv) , {: $ xv
  'nH n_kv d_head n_rot th eps n_exp n_used n_ff win' =. GHP
  'kind attn_n wq bq wk bk wv bv wo bo sinks post_n wrout brout wge bge wue bue wde bde' =. layer
  if. 32 = 3!:0 kind do. kind =. > kind end.
  wuse =. win
  if. -. kind -: 'swa' do. wuse =. 0 end.
  h =. H
  h =. attn_n rmsnorm xv
  'a kc vc' =. gptoss_mha h ; wq ; bq ; wk ; bk ; wv ; bv ; wo ; bo ; sinks ; nH ; n_kv ; d_head ; n_rot ; th ; wuse ; pos ; kc ; vc
  h2 =. xv + a
  r =. RN
  r =. post_n rmsnorm h2
  out =. FFOUT
  out =. h2 + gptoss_moe r ; wrout ; brout ; wge ; bge ; wue ; bue ; wde ; bde ; n_used
  (<out) , (<kc) , (<vc)
)

block_prefill_cached =: 3 : 0
  if. 4 = # y do.
    'xv n_head layer theta' =. y
  else.
    'xv n_head layer' =. y
  end.
  if. 1 = #$ xv do. xv =. ,: xv end.
  scratch_fit (# xv) , {: $ xv
  'nH n_kv d_head n_rot th eps n_exp n_used n_ff win' =. GHP
  'kind attn_n wq bq wk bk wv bv wo bo sinks post_n wrout brout wge bge wue bue wde bde' =. layer
  if. 32 = 3!:0 kind do. kind =. > kind end.
  wuse =. win
  if. -. kind -: 'swa' do. wuse =. 0 end.
  h =. H
  h =. attn_n rmsnorm xv
  'a kc vc' =. gptoss_mha h ; wq ; bq ; wk ; bk ; wv ; bv ; wo ; bo ; sinks ; nH ; n_kv ; d_head ; n_rot ; th ; wuse ; 0 ; (0 $ 0) ; (0 $ 0)
  h2 =. xv + a
  r =. RN
  r =. post_n rmsnorm h2
  out =. FFOUT
  out =. h2 + gptoss_moe r ; wrout ; brout ; wge ; bge ; wue ; bue ; wde ; bde ; n_used
  (<out) , (<kc) , (<vc)
)

cocurrent 'base'
