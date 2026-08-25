NB. jllama Gated DeltaNet helpers (Qwen3.5 linear-attention layers)
NB.
NB. Locale: jllamagdn
NB.
NB. Recurrence (llama.cpp ggml_gated_delta_net, K=1, scalar gate):
NB.   M is S^T (value x key), shape d x d per head
NB.   M *= exp(g)
NB.   delta = (v - M +/ . * k) * beta
NB.   M += delta */ k
NB.   out = (M +/ . * q) % %: d
NB.
NB. Causal conv1d: per-channel FIR of length d_conv, left-padded with zeros.
NB. Kernel layout in this locale: C x d_conv  (one kernel row per channel).

cocurrent 'jllamagdn'

load jpath '~temp/jllama/core/tensor.ijs'

sigmoid =: 1 % 1 + ^@:-
softplus =: 3 : 0
  NB. log(1+exp(y)); large y ~ y (f64 overflow guard)
  y >. ^. 1 + ^ y
)

NB. L2-normalize last axis. Optional left eps (default 1e_6).
l2norm =: 3 : 0
  1e_6 l2norm y
:
  y % %: x + +/"1 *: y
)

NB. Empty GDN state: y = n_v , d  -> (n_v x d x d) $ 0
gdn_empty =: 3 : 0
  'n_v d' =. y
  (n_v , d , d) $ 0
)

NB. Empty conv cache: y = d_conv , C  -> (d_conv-1) x C
conv_empty =: 3 : 0
  'dc C' =. y
  ((dc - 1) , C) $ 0
)

NB. ---------------------------------------------------------------
NB. Causal conv1d
NB. x = kernel (C x d_conv)
NB. y = n_tok x C
NB. ---------------------------------------------------------------
causal_conv1d =: 4 : 0
  k =. x
  xv =. y
  if. 1 = #$ xv do. xv =. ,: xv end.
  'nt C' =. $ xv
  dc =. {: $ k
  'causal_conv1d: channel mismatch' assert C = # k
  pad =. ((dc - 1) , C) $ 0
  z =. pad , xv
  out =. (0 , C) $ 0
  for_t. i. nt do.
    win =. (t + i. dc) { z
    out =. out , +/"1 k * |: win
  end.
  out
)

NB. One conv step with cache.
NB. y = kernel ; x1 ; cache
NB. kernel: C x d_conv ; x1: C or 1 x C ; cache: (d_conv-1) x C
NB. returns out1 (1 x C) ; cache2
conv_step =: 3 : 0
  'k x1 cache' =. y
  if. 1 = #$ x1 do. x1 =. ,: x1 end.
  win =. cache , x1
  o1 =. ,: +/"1 k * |: win
  cache2 =. }. win
  o1 ; cache2
)

NB. ---------------------------------------------------------------
NB. Gated delta net over a token sequence (all heads).
NB. y = Q ; K ; V ; g ; beta [; state]
NB. Q,K: n_tok x n_k x d
NB. V:   n_tok x n_v x d
NB. g, beta: n_tok x n_v   (scalar gate / beta per head)
NB. state: n_v x d x d  (M = S^T); omitted => zeros
NB. n_v must be a multiple of n_k (repeat Q/K heads if needed).
NB. returns outs (n_tok x n_v x d) ; state2
NB. ---------------------------------------------------------------
gdn_seq =: 3 : 0
  state =. 0
  if. 6 = # y do.
    'Q K V g beta state' =. y
  else.
    'Q K V g beta' =. y
  end.
  n_tok =. # V
  n_v =. 1 { $ V
  d =. {: $ V
  n_k =. 1 { $ Q
  'gdn_seq: n_k must divide n_v' assert 0 = n_k | n_v
  if. n_k < n_v do.
    n_rep =. n_v % n_k
    idx =. n_rep # i. n_k
    Q =. 1 0 2 |: idx { 1 0 2 |: Q
    K =. 1 0 2 |: idx { 1 0 2 |: K
  end.
  if. (0 = # , state) +. 0 = */ $ state do.
    state =. (n_v , d , d) $ 0
  end.
  scale =. % %: d
  Qb =. 1 0 2 |: Q
  Kb =. 1 0 2 |: K
  Vb =. 1 0 2 |: V
  gb =. |: g
  bb =. |: beta
  olist =. (n_v , n_tok , d) $ 0
  st2 =. (n_v , d , d) $ 0
  for_h. i. n_v do.
    M =. h { state
    qh =. h { Qb
    kh =. h { Kb
    vh =. h { Vb
    gh =. h { gb
    bh =. h { bb
    oh =. (0 , d) $ 0
    for_t. i. n_tok do.
      M =. M * ^ t { gh
      kt =. t { kh
      vt =. t { vh
      qt =. t { qh
      delta =. (t { bh) * vt - M +/ . * kt
      M =. M + delta */ kt
      oh =. oh , scale * M +/ . * qt
    end.
    olist =. oh h} olist
    st2 =. M h} st2
  end.
  outs =. 1 0 2 |: olist
  outs ; st2
)

cocurrent 'base'
