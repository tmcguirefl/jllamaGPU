NB. jllama Qwen3.5 dense hybrid graph (gated attention + Gated DeltaNet)
NB.
NB. Locale: jllamaqwen
NB.
NB. Installed into the live engine by `". Qwen35` (see core/arch.ijs).
NB. Text-only path: IMRoPE with identical section positions == NeoX on the
NB. first n_rot dims (Qwen3.5-2B: d_head=256, n_rot=64).
NB.
NB. Layer box (open list, first item is a char kind):
NB.   attn: 'attn'; attn_n; wq; wk; wv; wo; qn; kn; post_n; wg; wu; wd
NB.   gdn:  'gdn' ; attn_n; wqkv; wz; wconv; dt; sa; wbeta; walpha; snorm; wout; post_n; wg; wu; wd
NB.
NB. QHP (set by model_from_gguf_qwen):
NB.   n_head ; n_head_kv ; d_head ; n_rot ; theta ; eps ; d_conv ; d_state ; n_k ; n_v

cocurrent 'jllamaqwen'

load jpath '~temp/jllama/core/tensor.ijs'
load jpath '~temp/jllama/core/gdn.ijs'

split_heads =: split_heads_jllamaattn_
merge_heads =: merge_heads_jllamaattn_
attention_heads =: attention_heads_jllamaattn_
kv_empty =: kv_empty_jllamaattn_
ffn_swiglu =: ffn_swiglu_jllamablock_
rope_neox =: rope_neox_jllamarope_
sigmoid =: sigmoid_jllamagdn_
softplus =: softplus_jllamagdn_
l2norm =: l2norm_jllamagdn_
causal_conv1d =: causal_conv1d_jllamagdn_
conv_step =: conv_step_jllamagdn_
conv_empty =: conv_empty_jllamagdn_
gdn_seq =: gdn_seq_jllamagdn_
gdn_empty =: gdn_empty_jllamagdn_

NB. Default 2B-class hparams; overwritten at GGUF load.
QHP =: 8 ; 2 ; 256 ; 64 ; 10000 ; 1e_6 ; 4 ; 128 ; 16 ; 16

qhp_open =: 3 : 0
  if. 0 = # y do. QHP else. y end.
)

NB. (w;eps) rmsnorm_vec1 vector  — one last-axis cell
rmsnorm_vec1 =: 4 : 0
  'w eps' =. x
  ms =. (+/ % #) *: y
  w * y % %: ms + eps
)

NB. weight rmsnorm_e (acts ; eps)  — last-axis RMSNorm with explicit eps
rmsnorm_e =: 4 : 0
  w =. x
  'xv eps' =. y
  (w ; eps) rmsnorm_vec1"1 xv
)

layer_kind =: 3 : 0
  a =. 0 { y
  if. 32 = 3!:0 a do. a =. > a end.
  if. 2 = 3!:0 a do. a else. 'llama' end.
)

NB. ---------------------------------------------------------------
NB. Gated attention (full-attn Qwen3.5 layers)
NB. Q projection is 2 * n_head * d_head (query + sigmoid gate, interleaved
NB. per head as [q | gate]). Q/K RMSNorm then partial NeoX RoPE.
NB. ---------------------------------------------------------------

NB. y = x ; n_head ; n_kv ; d_head ; n_rot ; theta ; eps ; wq ; wk ; wv ; wo ; qn ; kn
qwen_mha_full =: 3 : 0
  'xv n_head n_kv d_head n_rot theta eps wq wk wv wo qn kn' =. y
  if. 1 = #$ xv do. xv =. ,: xv end.
  n_tok =. # xv
  Qf =. xv +/ . * wq
  Qf =. (n_tok , n_head , +: d_head) $ , Qf
  Q =. d_head {."1 Qf
  gate =. d_head }."1 Qf
  Q =. qn rmsnorm_e Q ; eps
  K =. n_kv split_heads xv +/ . * wk
  V =. n_kv split_heads xv +/ . * wv
  K =. kn rmsnorm_e K ; eps
  pos =. i. n_tok
  Q =. (pos ; theta ; n_rot) rope_neox Q
  K =. (pos ; theta ; n_rot) rope_neox K
  O =. attention_heads Q ; K ; V
  O =. O * sigmoid gate
  (merge_heads O) +/ . * wo
)

NB. y = x1 ; n_head ; n_kv ; d_head ; n_rot ; theta ; eps ; wq ; wk ; wv ; wo ; qn ; kn ; kc ; vc ; pos
NB. returns out1 ; kc2 ; vc2
qwen_mha_step =: 3 : 0
  'xv n_head n_kv d_head n_rot theta eps wq wk wv wo qn kn kc vc pos' =. y
  if. 1 = #$ xv do. xv =. ,: xv end.
  Qf =. xv +/ . * wq
  Qf =. (1 , n_head , +: d_head) $ , Qf
  Q =. d_head {."1 Qf
  gate =. d_head }."1 Qf
  Q =. qn rmsnorm_e Q ; eps
  K =. n_kv split_heads xv +/ . * wk
  V =. n_kv split_heads xv +/ . * wv
  K =. kn rmsnorm_e K ; eps
  Q =. ((, pos) ; theta ; n_rot) rope_neox Q
  K =. ((, pos) ; theta ; n_rot) rope_neox K
  kc =. kc , K
  vc =. vc , V
  O =. attention_heads Q ; kc ; vc
  O =. O * sigmoid gate
  out =. (merge_heads O) +/ . * wo
  out ; kc ; vc
)

NB. ---------------------------------------------------------------
NB. Linear attention (Gated DeltaNet)
NB. y = x ; QHP extras ; gdn weights ; [conv_cache ; ssm_state]
NB. ---------------------------------------------------------------

NB. Shared projection + conv + GDN core.
NB. y = xv ; wqkv ; wz ; wconv ; dt ; sa ; wbeta ; walpha ; snorm ; wout ; eps ; d_state ; n_k ; n_v ; d_conv [; ccache ; sstate]
NB. If caches omitted: full sequence, empty initial state.
NB. returns out ; ccache2 ; sstate2
qwen_gdn_run =: 3 : 0
  has_cache =. 17 = # y
  if. has_cache do.
    'xv wqkv wz wconv dt sa wbeta walpha snorm wout eps d_state n_k n_v d_conv ccache sstate' =. y
  else.
    'xv wqkv wz wconv dt sa wbeta walpha snorm wout eps d_state n_k n_v d_conv' =. y
    ccache =. 0 $ 0
    sstate =. 0 $ 0
  end.
  if. 1 = #$ xv do. xv =. ,: xv end.
  n_tok =. # xv
  qkv =. xv +/ . * wqkv
  z =. xv +/ . * wz
  C =. {: $ qkv
  if. (0 = # , ccache) *. n_tok > 1 do.
    mix =. silu wconv causal_conv1d qkv
    if. (d_conv - 1) > n_tok do.
      ccache2 =. (((d_conv - 1) - n_tok) , C) $ 0
      ccache2 =. ccache2 , qkv
    else.
      ccache2 =. (1 - d_conv) {. qkv
    end.
  elseif. 0 = # , ccache do.
    'mix ccache2' =. conv_step wconv ; qkv ; conv_empty d_conv , C
    mix =. silu mix
  elseif. do.
    NB. decode (or short prefill) with existing conv cache, step tokens
    mix =. (0 , C) $ 0
    ccache2 =. ccache
    for_t. i. n_tok do.
      'm1 ccache2' =. conv_step wconv ; (,: t { qkv) ; ccache2
      mix =. mix , m1
    end.
    mix =. silu mix
  end.
  key_dim =. n_k * d_state
  val_dim =. n_v * d_state
  q =. n_k split_heads key_dim {."1 mix
  k =. n_k split_heads key_dim {."1 key_dim }."1 mix
  v =. n_v split_heads val_dim {."1 (+: key_dim) }."1 mix
  q =. eps l2norm q
  k =. eps l2norm k
  beta =. sigmoid xv +/ . * wbeta
  alpha =. xv +/ . * walpha
  g =. sa *"1 softplus alpha +"1 dt
  'core sstate2' =. gdn_seq q ; k ; v ; g ; beta ; sstate
  NB. gated RMSNorm: rmsnorm(core, snorm) * silu(z)
  z3 =. (n_tok , n_v , d_state) $ , z
  core =. (snorm rmsnorm_e core ; eps) * silu z3
  out =. (n_tok , val_dim) $ , core
  out =. out +/ . * wout
  out ; ccache2 ; sstate2
)

NB. ---------------------------------------------------------------
NB. Block
NB. Same residual skeleton as Llama:
NB.   x = x + AttnOrGDN( rmsnorm(x) )
NB.   x = x + FFN( post_norm(x) )
NB. ---------------------------------------------------------------

qwen_ffn =: 3 : 0
  'h post_n wg wu wd eps' =. y
  r =. post_n rmsnorm_e h ; eps
  h + ffn_swiglu r ; wg ; wu ; wd
)

NB. y = (<x) , (<n_head) , layerbox [ , (<theta) ]
NB. n_head/theta from the call are ignored; QHP + layer kind decide.
block_full =: 3 : 0
  if. 4 = # y do.
    'xv n_head layer theta' =. y
  else.
    'xv n_head layer' =. y
  end.
  'nH n_kv d_head n_rot th eps dc dstate nk nv' =. QHP
  kind =. layer_kind layer
  if. kind -: 'attn' do.
    'knd attn_n wq wk wv wo qn kn post_n wg wu wd' =. layer
    h =. attn_n rmsnorm_e xv ; eps
    a =. qwen_mha_full h ; nH ; n_kv ; d_head ; n_rot ; th ; eps ; wq ; wk ; wv ; wo ; qn ; kn
    qwen_ffn (xv + a) ; post_n ; wg ; wu ; wd ; eps
  else.
    'knd attn_n wqkv wz wconv dt sa wbeta walpha snorm wout post_n wg wu wd' =. layer
    h =. attn_n rmsnorm_e xv ; eps
    'a cc ss' =. qwen_gdn_run h ; wqkv ; wz ; wconv ; dt ; sa ; wbeta ; walpha ; snorm ; wout ; eps ; dstate ; nk ; nv ; dc
    qwen_ffn (xv + a) ; post_n ; wg ; wu ; wd ; eps
  end.
)

NB. y = (<x1),(<n_head),layerbox,(<c1),(<c2),(<pos)[,(<theta)]
NB. returns (<out1) , (<c1) , (<c2)
block_step =: 3 : 0
  if. 7 = # y do.
    'xv n_head layer c1 c2 pos theta' =. y
  else.
    'xv n_head layer c1 c2 pos' =. y
  end.
  if. 1 = #$ xv do. xv =. ,: xv end.
  'nH n_kv d_head n_rot th eps dc dstate nk nv' =. QHP
  kind =. layer_kind layer
  if. kind -: 'attn' do.
    'knd attn_n wq wk wv wo qn kn post_n wg wu wd' =. layer
    h =. attn_n rmsnorm_e xv ; eps
    'a c1 c2' =. qwen_mha_step h ; nH ; n_kv ; d_head ; n_rot ; th ; eps ; wq ; wk ; wv ; wo ; qn ; kn ; c1 ; c2 ; pos
    out =. qwen_ffn (xv + a) ; post_n ; wg ; wu ; wd ; eps
    (<out) , (<c1) , (<c2)
  else.
    'knd attn_n wqkv wz wconv dt sa wbeta walpha snorm wout post_n wg wu wd' =. layer
    h =. attn_n rmsnorm_e xv ; eps
    'a c1 c2' =. qwen_gdn_run h ; wqkv ; wz ; wconv ; dt ; sa ; wbeta ; walpha ; snorm ; wout ; eps ; dstate ; nk ; nv ; dc ; c1 ; c2
    out =. qwen_ffn (xv + a) ; post_n ; wg ; wu ; wd ; eps
    (<out) , (<c1) , (<c2)
  end.
)

NB. Prefill via successive steps (builds per-layer cache)
block_prefill_cached =: 3 : 0
  if. 4 = # y do.
    'xv n_head layer theta' =. y
  else.
    'xv n_head layer' =. y
  end.
  layerbox =. <"_ layer
  n_embd =. {: $ xv
  'nH n_kv d_head n_rot th eps dc dstate nk nv' =. QHP
  kind =. layer_kind layer
  if. kind -: 'attn' do.
    'kc vc' =. kv_empty n_kv , d_head
  else.
    C =. (nk * dstate * 2) + nv * dstate
    kc =. conv_empty dc , C
    vc =. gdn_empty nv , dstate
  end.
  outs =. (0 , n_embd) $ 0
  for_t. i. # xv do.
    x1 =. ,: t { xv
    'o1 kc vc' =. block_step (<x1) , (<nH) , layerbox , (<kc) , (<vc) , (<t) , (<th)
    outs =. outs , o1
  end.
  (<outs) , (<kc) , (<vc)
)

cocurrent 'base'
