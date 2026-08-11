NB. jllama core tensor / linear-algebra facade
NB. M1: mp, silu, softmax, rmsnorm, linear, causal_mask, allclose
NB.
NB. Locale name must not contain '_' (J uses _ as name_locale_ separator).
NB.
NB. Layout (see docs/conventions.md):
NB.   activations: n_tok x n_embd  (or 1D n_embd)
NB.   linear W:    n_in  x n_out   with  x mp w
NB.   softmax:     over the last axis (1-cells / rows of a matrix)

cocurrent 'jllamatensor'

NB. ---------------------------------------------------------------
NB. Constants
NB. ---------------------------------------------------------------
RMS_EPS =: 1e_5
MASK_VAL =: _1e10           NB. added to forbidden attention scores
ATOL =: 1e_9
RTOL =: 1e_6

NB. ---------------------------------------------------------------
NB. Helpers
NB. ---------------------------------------------------------------

NB. Ensure rank >= 2 (table). Vectors become 1 x n.
make2d =: ,:^:(1 = #@$)

NB. Matrix product: (n,k) mp (k,m) -> (n,m)
mp =: +/ . *

NB. Numeric closeness: 1 iff all atoms match within atol/rtol.
NB. x allclose y
allclose =: 4 : 0
  *./ , (| x - y) <: ATOL + RTOL * | y
)

NB. ---------------------------------------------------------------
NB. Elementwise / reductions
NB. ---------------------------------------------------------------

NB. SiLU / swish: x * sigmoid(x) = x % (1 + exp(-x))
silu =: ] % 1 + ^@:-

NB. Softmax along last axis.
NB. Vector: full softmax. Table: per row (1-cells).
softmax =: 3 : 0
  if. 1 = #$y do.
    z =. y - >./ y
    e =. ^ z
    e % +/ e
  else.
    softmax"1 y
  end.
)

NB. RMSNorm (Llama-style).
NB. Dyadic:  weight rmsnorm x
NB.   weight: length n_embd
NB.   x:      n_embd  or  n_tok x n_embd
NB.   out:    same shape as x
NB. eps = RMS_EPS (1e_5 default; Llama often 1e-5)
rmsnorm =: 4 : 0
  w =. x
  if. 1 = #$y do.
    ms =. (+/ % #) *: y
    w * y % %: ms + RMS_EPS
  else.
    NB. ms is one scalar per row; divide with rank %"1 0
    ms =. (+/ % #)"1 *: y
    w *"1 y %"1 0 %: ms + RMS_EPS
  end.
)

NB. ---------------------------------------------------------------
NB. Linear + mask
NB. ---------------------------------------------------------------

NB. Affine map.
NB.   linear x;w        -> x mp w
NB.   linear x;w;b      -> b +"1 x mp w
NB. x: n_tok x n_in   w: n_in x n_out   b: n_out
linear =: 3 : 0
  n =. # y
  if. n = 2 do.
    'xv wv' =. y
    xv mp wv
  elseif. n = 3 do.
    'xv wv bv' =. y
    bv +"1 xv mp wv
  elseif. do.
    'linear expects x;w or x;w;b' assert 0
  end.
)

NB. Causal attention mask (n x n).
NB. 0 where key pos <= query pos (allowed); MASK_VAL above diagonal.
NB. Usage:  softmax mask + scores
causal_mask =: MASK_VAL * </~@i.

cocurrent 'base'
