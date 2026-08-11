NB. jllama multi-head attention + KV cache (M2)
NB.
NB. Layout:
NB.   x, out:      n_tok x n_embd
NB.   Wq Wk Wv Wo: n_embd x n_embd   (MHA; GQA later)
NB.   heads:       n_tok x n_head x d_head
NB.   K/V cache:   n_past x n_head x d_head
NB.
NB. RoPE on Q and K only. Full prefill uses causal mask.
NB. Decode step with n_q=1 sees full cache (no mask needed).
NB.
NB. Load order: core/tensor.ijs , core/rope.ijs , core/attention.ijs

cocurrent 'jllamaattn'

mp =: mp_jllamatensor_
softmax =: softmax_jllamatensor_
causal_mask =: causal_mask_jllamatensor_
rope =: rope_jllamarope_
DEFAULT_THETA =: DEFAULT_THETA_jllamarope_

NB. ---------------------------------------------------------------
NB. Head pack / unpack
NB. ---------------------------------------------------------------

NB. n_head split_heads (n_tok x n_embd) -> n_tok x n_head x d_head
NB. Ravel before $ so items are atoms (otherwise $ keeps row items).
split_heads =: 4 : 0
  'n_tok n_embd' =. $ y
  'split_heads: n_embd not divisible by n_head' assert 0 = x | n_embd
  d_head =. n_embd % x
  (n_tok , x , d_head) $ , y
)

NB. n_tok x n_head x d_head -> n_tok x n_embd
merge_heads =: 3 : 0
  'n_tok n_head d_head' =. $ y
  (n_tok , n_head * d_head) $ , y
)

NB. Empty KV cache: y = n_head , d_head  -> k ; v
kv_empty =: 3 : 0
  'n_head d_head' =. y
  z =. (0 , n_head , d_head) $ 0
  z ; z
)

NB. ---------------------------------------------------------------
NB. Core attention
NB. ---------------------------------------------------------------

NB. Single head: q (n_q x d) ; k (n_k x d) ; v (n_k x d)
NB. Causal mask only when n_q=n_k and n_q>1 (full prefill).
attention1 =: 3 : 0
  'q k v' =. y
  d =. {: $ q
  scores =. (q mp |: k) % %: d
  nq =. # q
  nk =. # k
  if. (nq = nk) *. nq > 1 do.
    scores =. scores + causal_mask nq
  end.
  (softmax scores) mp v
)

NB. Q,K,V each n_tok x n_head x d_head (K,V may be longer on tok axis)
NB. -> n_q x n_head x d_head
attention_heads =: 3 : 0
  'q k v' =. y
  n_head =. 1 { $ q
  nq =. # q
  dh =. {: $ q
  NB. 1 0 2 |:  -> n_head x n_tok x d_head
  qb =. 1 0 2 |: q
  kb =. 1 0 2 |: k
  vb =. 1 0 2 |: v
  heads =. ''
  for_h. i. n_head do.
    heads =. heads , < attention1 (h { qb) ; (h { kb) ; (h { vb)
  end.
  NB. >heads is n_head x n_q x d_head ; reorder to n_q x n_head x d_head
  1 0 2 |: > heads
)

NB. ---------------------------------------------------------------
NB. Project + RoPE
NB. ---------------------------------------------------------------

NB. (n_head) project_qkv x;wq;wk;wv -> Q;K;V (no rope)
project_qkv =: 4 : 0
  n_head =. x
  'xv wq wk wv' =. y
  Q =. n_head split_heads xv mp wq
  K =. n_head split_heads xv mp wk
  V =. n_head split_heads xv mp wv
  Q ; K ; V
)

NB. pos apply_rope_qk Q;K -> Qr;Kr
apply_rope_qk =: 4 : 0
  'Q K' =. y
  (x rope Q) ; (x rope K)
)

NB. ---------------------------------------------------------------
NB. Full-sequence MHA (reference path, no cache)
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
  if. theta ~: DEFAULT_THETA do.
    'Q K' =. (pos ; theta) apply_rope_qk Q ; K
  else.
    'Q K' =. pos apply_rope_qk Q ; K
  end.
  O =. attention_heads Q ; K ; V
  (merge_heads O) mp wo
)

NB. ---------------------------------------------------------------
NB. One decode step with KV cache
NB. y = x1 ; n_head ; wq ; wk ; wv ; wo ; kcache ; vcache ; pos
NB. optional theta as 10th item
NB. x1: 1 x n_embd (or vector n_embd)
NB. returns out1 ; kcache2 ; vcache2
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
  if. theta ~: DEFAULT_THETA do.
    'Q K' =. ((,pos) ; theta) apply_rope_qk Q ; K
  else.
    'Q K' =. (, pos) apply_rope_qk Q ; K
  end.
  kc =. kc , K
  vc =. vc , V
  O =. attention_heads Q ; kc ; vc
  out =. (merge_heads O) mp wo
  out ; kc ; vc
)

NB. Prefill by successive mha_step (builds cache)
NB. y = x ; n_head ; wq ; wk ; wv ; wo  [; theta]
NB. returns outs ; kcache ; vcache
mha_prefill_cached =: 3 : 0
  if. 7 = # y do.
    'xv n_head wq wk wv wo theta' =. y
  else.
    'xv n_head wq wk wv wo' =. y
    theta =. DEFAULT_THETA
  end.
  d_head =. ({: $ xv) % n_head
  'kc vc' =. kv_empty n_head , d_head
  outs =. (0 , {: $ xv) $ 0
  for_t. i. # xv do.
    x1 =. ,: t { xv
    'o1 kc vc' =. mha_step x1 ; n_head ; wq ; wk ; wv ; wo ; kc ; vc ; t ; theta
    outs =. outs , o1
  end.
  outs ; kc ; vc
)

cocurrent 'base'
