NB. jllama RoPE (rotary position embeddings)
NB. Llama-style half-rotation on the last axis:
NB.   rotate_half([x1|x2]) = [-x2|x1]
NB.   out = x * cos + rotate_half(x) * sin
NB.
NB. Last axis is d_head (must be even).
NB. Rank 2: n_tok x d_head
NB. Rank 3: n_tok x n_head x d_head
NB. Positions: integer list length n_tok (usually i. n_tok)
NB.
NB. inv_freq[i] = theta ^ (-(2i)/d) for i = 0 .. d/2-1
NB. e.g. d=4, theta=10000 -> 1 , 0.01
NB.
NB. Load before attention.ijs.

cocurrent 'jllamarope'

DEFAULT_THETA =: 10000

NB. theta rope_inv d_head  -> inv_freq length d_head%2
rope_inv =: 4 : 0
  theta =. x
  d =. y
  'rope: d_head must be even and positive' assert (d > 0) *. 0 = 2 | d
  % theta ^ ((+: i. -: d) % d)
)

NB. pos rope_sincos inv_freq -> cos ; sin  each (#pos) x d_head
rope_sincos =: 4 : 0
  freqs =. x */ y
  c =. 2 o. freqs
  s =. 1 o. freqs
  (c ,. c) ; (s ,. s)
)

NB. rotate_half on last axis
rotate_half1 =: 3 : 0
  h =. -: # y
  (- h }. y) , h {. y
)
rotate_half =: rotate_half1"1

NB. n_head expand_heads (n_tok x d) -> n_tok x n_head x d
NB. J does not broadcast (n_tok x 1 x d) against (n_tok x n_head x d).
NB. Copy each position row across heads (x # y), then reshape.
expand_heads =: 4 : 0
  'n_tok d' =. $ y
  (n_tok , x , d) $ , x # y
)

NB. Monadic: positions i. n_tok , default theta
NB. Dyadic:  pos rope x   or   (pos;theta) rope x
rope =: 3 : 0
  (i. {. $ y) rope y
:
  if. 32 = 3!:0 x do.
    'pos theta' =. 2 {. x , < DEFAULT_THETA
  else.
    pos =. x
    theta =. DEFAULT_THETA
  end.
  rnk =. #$ y
  'rope: expected rank 2 or 3' assert rnk e. 2 3
  d =. {: $ y
  inv =. theta rope_inv d
  'cs ss' =. pos rope_sincos inv
  if. rnk = 3 do.
    nh =. 1 { $ y
    cs =. nh expand_heads cs
    ss =. nh expand_heads ss
  end.
  (y * cs) + (rotate_half y) * ss
)

cocurrent 'base'
