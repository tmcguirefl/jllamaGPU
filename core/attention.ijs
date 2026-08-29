NB. jllama multi-head attention + KV cache (M2 + M13 GQA)
NB.
NB. Layout (GPU / GGUF):
NB.   x, out:      n_tok x n_embd          (F32)
NB.   Wq, Wo:      n_embd x n_embd         (n_out x n_in, last axis = K)
NB.   Wk, Wv:      (n_head_kv * d_head) x n_embd
NB.   Q heads:     n_tok x n_head x d_head
NB.   K/V heads:   n_tok x n_head_kv x d_head
NB.   K/V cache:   n_past x n_head_kv x d_head
NB. Weight GEMM: x linear_jgpu_ W. Score GEMM stays q +/ . * |: k (F32).
NB.
NB. GQA: n_head_kv may be < n_head (must divide). Each KV head is
NB. repeated n_rep = n_head % n_head_kv times at attention time.
NB. n_head_kv is derived from Wk/Wv width (not a separate arg).
NB.
NB. RoPE on Q and K only. Full prefill uses causal mask.
NB. Decode step with n_q=1 sees full cache (no mask needed).
NB.
NB. Load order: core/tensor.ijs , core/rope.ijs , core/attention.ijs

cocurrent 'jllamaattn'

NB. CPU tensor helpers (causal_mask, allclose) then GPU kernels for the graph.
load ROOT_jllamasys_ , 'core/tensor.ijs'
silu =: silu_jgpu_
softmax =: softmax_jgpu_
rmsnorm =: rmsnorm_jgpu_
linear =: linear_jgpu_
rope =: rope_jgpu_
DEFAULT_THETA =: DEFAULT_THETA_jllamarope_

NB. ---------------------------------------------------------------
NB. Head pack / unpack
NB. ---------------------------------------------------------------

NB. n_head split_heads (n_tok x n_proj) -> n_tok x n_head x d_head
NB. Ravel before $ so items are atoms (otherwise $ keeps row items).
split_heads =: 4 : 0
  'n_tok n_proj' =. $ y
  'split_heads: width not divisible by n_head' assert 0 = x | n_proj
  d_head =. n_proj % x
  (n_tok , x , d_head) $ , y
)

NB. n_tok x n_head x d_head -> n_tok x (n_head * d_head)
merge_heads =: 3 : 0
  'n_tok n_head d_head' =. $ y
  (n_tok , n_head * d_head) $ , y
)

NB. Empty KV cache: y = n_head_kv , d_head  -> k ; v
kv_empty =: 3 : 0
  'n_kv d_head' =. y
  z =. (0 , n_kv , d_head) $ 0
  z ; z
)

NB. GQA: repeat each KV head n_rep times along head axis.
NB. (n_rep) expand_kv (n_tok x n_kv x d) -> n_tok x (n_kv*n_rep) x d
expand_kv =: 4 : 0
  n_rep =. x
  if. 1 = n_rep do. y return. end.
  'expand_kv: n_rep must be >= 1' assert n_rep >: 1
  n_kv =. 1 { $ y
  idx =. n_rep # i. n_kv
  1 0 2 |: idx { 1 0 2 |: y
)

NB. ---------------------------------------------------------------
NB. Core attention
NB. ---------------------------------------------------------------

NB. Single head: q (n_q x d) ; k (n_k x d) ; v (n_k x d)
NB. Causal mask only when n_q=n_k and n_q>1 (full prefill).
attention1 =: 3 : 0
  'q k v' =. y
  d =. {: $ q
  scores =. (q +/ . * |: k) % %: d
  nq =. # q
  nk =. # k
  if. (nq = nk) *. nq > 1 do.
    scores =. scores + causal_mask nq
  end.
  (softmax scores) +/ . * v
)

NB. Q: n_tok x n_head x d_head
NB. K,V: n_tok x n_head_kv x d_head (K,V may be longer on tok axis)
NB. GQA expands K,V to n_head before per-head attention.
NB. -> n_q x n_head x d_head
NB. Heads are an axis: one batched +/ .* , not for_h. GPU rank-3
NB. (b,m,k) +/ .* (b,k,n) is batched GEMM.
attention_heads =: 3 : 0
  'q k v' =. y
  n_head =. 1 { $ q
  n_kv =. 1 { $ k
  'attention_heads: n_head_kv must divide n_head' assert 0 = n_kv | n_head
  n_rep =. n_head % n_kv
  if. n_rep > 1 do.
    k =. n_rep expand_kv k
    v =. n_rep expand_kv v
  end.
  nq =. # q
  nk =. # k
  dh =. {: $ q
  NB. 1 0 2 |:  -> n_head x n_tok x d_head
  qb =. 1 0 2 |: q
  kb =. 1 0 2 |: k
  vb =. 1 0 2 |: v
  scores =. (qb +/ . * 0 2 1 |: kb) % %: dh
  if. (nq = nk) *. nq > 1 do.
    scores =. scores + causal_mask nq
  end.
  1 0 2 |: (softmax scores) +/ . * vb
)

NB. ---------------------------------------------------------------
NB. Project + RoPE
NB. ---------------------------------------------------------------

NB. (n_head) project_qkv x;wq;wk;wv -> Q;K;V (no rope)
NB. n_head_kv derived from {:$ wk using d_head = n_embd % n_head
project_qkv =: 4 : 0
  n_head =. x
  'xv wq wk wv' =. y
  n_embd =. {: $ xv
  'project_qkv: n_embd not divisible by n_head' assert 0 = n_head | n_embd
  d_head =. n_embd % n_head
  'project_qkv: bad Wq out' assert ({. $ wq) = n_head * d_head
  'project_qkv: bad Wk out' assert 0 = d_head | {. $ wk
  n_kv =. ({. $ wk) % d_head
  'project_qkv: bad Wv out' assert ({. $ wv) = n_kv * d_head
  Q =. n_head split_heads xv linear wq
  K =. n_kv split_heads xv linear wk
  V =. n_kv split_heads xv linear wv
  Q ; K ; V
)

NB. pos or (pos;theta) apply_rope_qk Q;K -> Qr;Kr
NB. GPU RoPE: (pos;theta;n_rot;mode) with mode 0 = Llama NORMAL.
apply_rope_qk =: 4 : 0
  'Q K' =. y
  d =. {: $ Q
  if. 32 = 3!:0 x do.
    pad =. x
    if. 1 = # pad do. pad =. pad , < DEFAULT_THETA end.
    'pos theta' =. 2 {. pad
  else.
    pos =. x
    theta =. DEFAULT_THETA
  end.
  spec =. (<, pos) , (<theta) , (<d) , (<0)
  (spec rope Q) ; (spec rope K)
)

NB. ---------------------------------------------------------------
NB. Full-sequence MHA/GQA (reference path, no cache)
NB. y = x ; n_head ; wq ; wk ; wv ; wo
NB. optional trailing theta: x;n_head;wq;wk;wv;wo;theta
NB. ---------------------------------------------------------------
mha_full =: 3 : 0
  if. 7 = # y do.
    'xv n_head wq wk wv wo theta' =. y
  else.
    'xv n_head wq wk wv wo' =. y
    theta =. DEFAULT_THETA
  end.
  'Q K V' =. n_head project_qkv xv ; wq ; wk ; wv
  pos =. i. # xv
  'Q K' =. (pos ; theta) apply_rope_qk Q ; K
  O =. attention_heads Q ; K ; V
  (merge_heads O) linear wo
)

NB. ---------------------------------------------------------------
NB. One decode step with KV cache
NB. y = x1 ; n_head ; wq ; wk ; wv ; wo ; kcache ; vcache ; pos
NB. optional theta as 10th item
NB. x1: 1 x n_embd (or vector n_embd)
NB. returns out1 ; kcache2 ; vcache2
NB. Cache stores n_head_kv heads (unexpanded).
NB. ---------------------------------------------------------------
mha_step =: 3 : 0
  if. 10 = # y do.
    'xv n_head wq wk wv wo kc vc pos theta' =. y
  else.
    'xv n_head wq wk wv wo kc vc pos' =. y
    theta =. DEFAULT_THETA
  end.
  if. 1 = #$ xv do. xv =. ,: xv end.
  'Q K V' =. n_head project_qkv xv ; wq ; wk ; wv
  'Q K' =. ((, pos) ; theta) apply_rope_qk Q ; K
  kc =. kc , K
  vc =. vc , V
  O =. attention_heads Q ; kc ; vc
  out =. (merge_heads O) linear wo
  out ; kc ; vc
)

NB. Full-sequence MHA that also returns K/V cache for decode.
NB. y = x ; n_head ; wq ; wk ; wv ; wo  [; theta]
NB. returns outs ; kcache ; vcache
mha_prefill_cached =: 3 : 0
  if. 7 = # y do.
    'xv n_head wq wk wv wo theta' =. y
  else.
    'xv n_head wq wk wv wo' =. y
    theta =. DEFAULT_THETA
  end.
  if. 1 = #$ xv do. xv =. ,: xv end.
  'Q K V' =. n_head project_qkv xv ; wq ; wk ; wv
  pos =. i. # xv
  'Q K' =. (pos ; theta) apply_rope_qk Q ; K
  O =. attention_heads Q ; K ; V
  out =. (merge_heads O) linear wo
  out ; K ; V
)

cocurrent 'base'
