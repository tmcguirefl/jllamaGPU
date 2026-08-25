NB. M1 unit tests for core/tensor.ijs
NB. Load via jllama:  jllama_test ''
NB. Or:  load <path to jllama_dev.ijs> then jrequire_jllama_ 'test/test_tensor.ijs'
NB.
NB. tensor.ijs has no locale — load into this test locale.

cocurrent 'jllamatest'

load ROOT_jllamasys_ , 'core/tensor.ijs'

NB. ---------------------------------------------------------------
NB. Individual cases (each returns 1 on success)
NB. ---------------------------------------------------------------

test_mp =: 3 : 0
  a =. 2 3 $ 1 2 3 4 5 6
  b =. 3 2 $ 7 8 9 10 11 12
  c =. a +/ . * b
  expect =. 2 2 $ 58 64 139 154
  assert. expect -: c
  1
)

test_silu =: 3 : 0
  got =. silu 0 1 _1 2
  NB. goldens from numpy x/(1+exp(-x))
  exp =. 0 0.7310585786300049 _0.2689414213699951 1.7615941559557647
  assert. exp allclose got
  1
)

test_softmax_vec =: 3 : 0
  got =. softmax 1 2 3
  exp =. 0.09003057317038046 0.24472847105479764 0.6652409557748218
  assert. exp allclose got
  assert. 1 allclose +/ got
  1
)

test_softmax_mat =: 3 : 0
  got =. softmax 2 3 $ 1 2 3 3 0 0
  exp =. 2 3 $ 0.09003057317038046 0.24472847105479764 0.6652409557748218 0.9094430004804077 0.04527849975979621 0.04527849975979621
  assert. exp allclose got
  assert. (1 1) allclose +/"1 got
  1
)

test_rmsnorm_vec =: 3 : 0
  x =. 1 2 3 4
  w =. 0.5 1 1.5 2
  got =. w rmsnorm x
  exp =. 0.182574057250788 0.7302962290647869 1.6431665153957703 2.9211850162591476
  assert. exp allclose got
  1
)

test_rmsnorm_mat =: 3 : 0
  x =. 2 4 $ 1 2 3 4 4 3 2 1
  w =. 0.5 1 1.5 2
  got =. w rmsnorm x
  exp =. 2 4 $ 0.182574057250788 0.7302962290647869 1.6431665153957703 2.9211850162591476 0.7302962290647869 1.095444343597192 1.095444343597192 0.7302962290647869
  assert. exp allclose got
  1
)

test_linear_bias =: 3 : 0
  x =. 2 3 $ 1 2 3 4 5 6
  w =. 3 2 $ 1 0 0 1 1 1
  b =. 0.1 0.2
  got =. linear x ; w ; b
  exp =. 2 2 $ 4.1 5.2 10.1 11.2
  assert. exp allclose got
  1
)

test_linear_nobias =: 3 : 0
  x =. 2 3 $ 1 2 3 4 5 6
  w =. 3 2 $ 1 0 0 1 1 1
  got =. linear x ; w ; 0
  exp =. 2 2 $ 4 5 10 11
  assert. exp -: got
  1
)

test_causal_mask =: 3 : 0
  m =. causal_mask 4
  assert. 4 4 -: $ m
  NB. diagonal and below are 0 (allowed)
  assert. 0 = (<0 0) { m
  assert. 0 = (<3 0) { m
  assert. 0 = (<3 3) { m
  NB. above diagonal forbidden
  assert. m = MASK_VAL * </~ i. 4
  NB. softmax of zeros+mask: mass only on causal positions
  s =. softmax m + 4 4 $ 0
  assert. 0 allclose (<0 3) { s
  assert. 1 allclose +/"1 s
  1
)

NB. ---------------------------------------------------------------
NB. Runner
NB. ---------------------------------------------------------------
TESTS =: 'test_mp'; 'test_silu'; 'test_softmax_vec'; 'test_softmax_mat'; 'test_rmsnorm_vec'; 'test_rmsnorm_mat'; 'test_linear_bias'; 'test_linear_nobias'; 'test_causal_mask'

run =: 3 : 0
  failed =. 0
  for_t. TESTS do.
    name =. > t
    try.
      r =. (name ~ '')
      if. 1 -: r do.
        smoutput '  pass  ' , name
      else.
        smoutput '  FAIL  ' , name , ' (bad return)'
        failed =. failed + 1
      end.
    catch.
      smoutput '  FAIL  ' , name
      smoutput '        ' , (13!:12 '')
      failed =. failed + 1
    end.
  end.
  if. failed do.
    smoutput 'jllamatest: ' , (": failed) , ' failed'
    0
  else.
    smoutput 'jllamatest: ' , (": # TESTS) , ' passed'
    1
  end.
)

cocurrent 'base'
