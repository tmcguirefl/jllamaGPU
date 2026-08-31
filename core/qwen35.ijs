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

load ROOT_jllamasys_ , 'core/tensor.ijs'
load ROOT_jllamasys_ , 'core/gdn.ijs'
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
sigmoid =: sigmoid_jllamagdn_
NB. sigmoid is 1 % 1 + ^@:-  — GPU F32 ^ and - are engine primitives.
softplus =: softplus_jllamagdn_
l2norm =: l2norm_jllamagdn_
causal_conv1d =: causal_conv1d_jllamagdn_
conv_step =: conv_step_jllamagdn_
conv_empty =: conv_empty_jllamagdn_
gdn_seq =: gdn_seq_jllamagdn_
gdn_empty =: gdn_empty_jllamagdn_

NB. Default 2B-class hparams; overwritten at GGUF load.
QHP =: 8 ; 2 ; 256 ; 64 ; 10000 ; 1e_6 ; 4 ; 128 ; 16 ; 16
NT =: 0
NEMBD =: 0

qhp_open =: 3 : 0
  if. 0 = # y do. QHP else. y end.
)

NB. G: zeros of shape y. Re-fit when n_tok or n_embd change (prefill vs decode).
gs =: 3 : 'G: y $ 0'

scratch_fit =: 3 : 0
  nt =. {. y
  ne =. {: y
  if. (nt = NT) *. (ne = NEMBD) *. 0 ~: NT do. i. 0 0 return. end.
  'nH n_kv d_head n_rot th eps dc dstate nk nv' =. QHP
  qkvd =. ((+: nk) + nv) * dstate
  H =: gs nt , ne
  A =: gs nt , ne
  RN =: gs nt , ne
  QKV =: gs nt , qkvd
  ZG =: gs nt , nv * dstate
  MIX =: gs nt , qkvd
  Q =: gs nt , nk , dstate
  K =: gs nt , nk , dstate
  V =: gs nt , nv , dstate
  BETA =: gs nt , nv
  ALPHA =: gs nt , nv
  GG =: gs nt , nv
  CORE =: gs nt , nv , dstate
  QF =: gs nt , nH , +: d_head
  QH =: gs nt , nH , d_head
  GATE =: gs nt , nH , d_head
  KH =: gs nt , n_kv , d_head
  VH =: gs nt , n_kv , d_head
  OH =: gs nt , nH , d_head
  NT =: nt
  NEMBD =: ne
  smoutput 'jllama: G: scratch ' , (": nt) , ' x ' , ": ne
  i. 0 0
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
  scratch_fit n_tok , {: $ xv
  Qf =. QF
  Qf =. xv linear wq
  Qf =. (n_tok , n_head , +: d_head) $ , Qf
  Q =. QH
  Q =. d_head {."1 Qf
  gate =. GATE
  gate =. d_head }."1 Qf
  Q =. qn rmsnorm Q
  K =. KH
  K =. n_kv split_heads xv linear wk
  V =. VH
  V =. n_kv split_heads xv linear wv
  K =. kn rmsnorm K
  pos =. i. n_tok
  spec =. (<pos) , (<theta) , (<n_rot) , (<2)
  Q =. spec rope Q
  K =. spec rope K
  O =. OH
  O =. attention_heads Q ; K ; V
  O =. O * sigmoid gate
  (merge_heads O) linear wo
)

NB. y = x1 ; n_head ; n_kv ; d_head ; n_rot ; theta ; eps ; wq ; wk ; wv ; wo ; qn ; kn ; kc ; vc ; pos
NB. returns out1 ; kc2 ; vc2
qwen_mha_step =: 3 : 0
  'xv n_head n_kv d_head n_rot theta eps wq wk wv wo qn kn kc vc pos' =. y
  if. 1 = #$ xv do. xv =. ,: xv end.
  scratch_fit (# xv) , {: $ xv
  Qf =. QF
  Qf =. xv linear wq
  Qf =. (1 , n_head , +: d_head) $ , Qf
  Q =. QH
  Q =. d_head {."1 Qf
  gate =. GATE
  gate =. d_head }."1 Qf
  Q =. qn rmsnorm Q
  K =. KH
  K =. n_kv split_heads xv linear wk
  V =. VH
  V =. n_kv split_heads xv linear wv
  K =. kn rmsnorm K
  spec =. ((<, pos) , (<theta) , (<n_rot) , (<2))
  Q =. spec rope Q
  K =. spec rope K
  kc =. kc , K
  vc =. vc , V
  O =. attention_heads Q ; kc ; vc
  O =. O * sigmoid gate
  out =. (merge_heads O) linear wo
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
  scratch_fit n_tok , {: $ xv
  qkv =. QKV
  qkv =. xv linear wqkv
  z =. ZG
  z =. xv linear wz
  C =. {: $ qkv
  mix =. MIX
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
  NB. mix is n_tok x (2*n_k + n_v)*d_state. Split via reshape + { (also {."1 on GPU).
  nparts =. (+: n_k) + n_v
  r =. 1 0 2 |: (n_tok , nparts , d_state) $ , mix
  q =. Q
  q =. 1 0 2 |: (i. n_k) { r
  k =. K
  k =. 1 0 2 |: (n_k + i. n_k) { r
  v =. V
  v =. 1 0 2 |: ((+: n_k) + i. n_v) { r
  q =. eps l2norm q
  k =. eps l2norm k
  beta =. BETA
  beta =. sigmoid xv linear wbeta
  alpha =. ALPHA
  alpha =. xv linear walpha
  g =. GG
  g =. sa *"1 softplus alpha +"1 dt
  if. 0 = # , sstate do. sstate =. 0 $ 0 end.
  'core sstate2' =. gdn_seq q ; k ; v ; g ; beta ; sstate
  core =. CORE
  NB. gated RMSNorm: rmsnorm(core, snorm) * silu(z)
  z3 =. (n_tok , n_v , d_state) $ , z
  core =. (snorm rmsnorm core) * silu z3
  out =. A
  out =. ((n_tok , val_dim) $ , core) linear wout
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
  r =. RN
  r =. post_n rmsnorm h
  out =. A
  out =. h + ffn_swiglu r ; wg ; wu ; wd
  out
)

NB. y = (<x) , (<n_head) , layerbox [ , (<theta) ]
NB. n_head/theta from the call are ignored; QHP + layer kind decide.
block_full =: 3 : 0
  if. 4 = # y do.
    'xv n_head layer theta' =. y
  else.
    'xv n_head layer' =. y
  end.
  if. 1 = #$ xv do. xv =. ,: xv end.
  scratch_fit (# xv) , {: $ xv
  'nH n_kv d_head n_rot th eps dc dstate nk nv' =. QHP
  kind =. layer_kind layer
  h =. H
  if. kind -: 'attn' do.
    'knd attn_n wq wk wv wo qn kn post_n wg wu wd' =. layer
    h =. attn_n rmsnorm xv
    a =. qwen_mha_full h ; nH ; n_kv ; d_head ; n_rot ; th ; eps ; wq ; wk ; wv ; wo ; qn ; kn
    qwen_ffn (xv + a) ; post_n ; wg ; wu ; wd ; eps
  else.
    'knd attn_n wqkv wz wconv dt sa wbeta walpha snorm wout post_n wg wu wd' =. layer
    h =. attn_n rmsnorm xv
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
  scratch_fit (# xv) , {: $ xv
  'nH n_kv d_head n_rot th eps dc dstate nk nv' =. QHP
  kind =. layer_kind layer
  h =. H
  if. kind -: 'attn' do.
    'knd attn_n wq wk wv wo qn kn post_n wg wu wd' =. layer
    h =. attn_n rmsnorm xv
    'a c1 c2' =. qwen_mha_step h ; nH ; n_kv ; d_head ; n_rot ; th ; eps ; wq ; wk ; wv ; wo ; qn ; kn ; c1 ; c2 ; pos
    out =. qwen_ffn (xv + a) ; post_n ; wg ; wu ; wd ; eps
    (<out) , (<c1) , (<c2)
  else.
    'knd attn_n wqkv wz wconv dt sa wbeta walpha snorm wout post_n wg wu wd' =. layer
    h =. attn_n rmsnorm xv
    'a c1 c2' =. qwen_gdn_run h ; wqkv ; wz ; wconv ; dt ; sa ; wbeta ; walpha ; snorm ; wout ; eps ; dstate ; nk ; nv ; dc ; c1 ; c2
    out =. qwen_ffn (xv + a) ; post_n ; wg ; wu ; wd ; eps
    (<out) , (<c1) , (<c2)
  end.
)

NB. Full-sequence MHA that also returns K/V cache for decode.
qwen_mha_prefill =: 3 : 0
  'xv n_head n_kv d_head n_rot theta eps wq wk wv wo qn kn' =. y
  if. 1 = #$ xv do. xv =. ,: xv end.
  n_tok =. # xv
  scratch_fit n_tok , {: $ xv
  Qf =. QF
  Qf =. xv linear wq
  Qf =. (n_tok , n_head , +: d_head) $ , Qf
  Q =. QH
  Q =. d_head {."1 Qf
  gate =. GATE
  gate =. d_head }."1 Qf
  Q =. qn rmsnorm Q
  K =. KH
  K =. n_kv split_heads xv linear wk
  V =. VH
  V =. n_kv split_heads xv linear wv
  K =. kn rmsnorm K
  pos =. i. n_tok
  spec =. (<pos) , (<theta) , (<n_rot) , (<2)
  Q =. spec rope Q
  K =. spec rope K
  O =. attention_heads Q ; K ; V
  O =. O * sigmoid gate
  out =. (merge_heads O) linear wo
  out ; K ; V
)

NB. Prefill as one sequence (not per-token step). Returns out ; cache1 ; cache2.
block_prefill_cached =: 3 : 0
  if. 4 = # y do.
    'xv n_head layer theta' =. y
  else.
    'xv n_head layer' =. y
  end.
  if. 1 = #$ xv do. xv =. ,: xv end.
  scratch_fit (# xv) , {: $ xv
  'nH n_kv d_head n_rot th eps dc dstate nk nv' =. QHP
  kind =. layer_kind layer
  h =. H
  if. kind -: 'attn' do.
    'knd attn_n wq wk wv wo qn kn post_n wg wu wd' =. layer
    h =. attn_n rmsnorm xv
    'a kc vc' =. qwen_mha_prefill h ; nH ; n_kv ; d_head ; n_rot ; th ; eps ; wq ; wk ; wv ; wo ; qn ; kn
    out =. qwen_ffn (xv + a) ; post_n ; wg ; wu ; wd ; eps
    (<out) , (<kc) , (<vc)
  else.
    'knd attn_n wqkv wz wconv dt sa wbeta walpha snorm wout post_n wg wu wd' =. layer
    h =. attn_n rmsnorm xv
    'a kc vc' =. qwen_gdn_run h ; wqkv ; wz ; wconv ; dt ; sa ; wbeta ; walpha ; snorm ; wout ; eps ; dstate ; nk ; nv ; dc
    out =. qwen_ffn (xv + a) ; post_n ; wg ; wu ; wd ; eps
    (<out) , (<kc) , (<vc)
  end.
)

cocurrent 'base'
