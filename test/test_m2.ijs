NB. M2 tests: RoPE, MHA, KV cache parity vs full recompute
NB. Loaded by jllama_test ''

cocurrent 'jllamatestm2'

load ROOT_jllamasys_ , 'core/tensor.ijs'
rope =: rope_jllamarope_
rotate_half =: rotate_half_jllamarope_
rope_inv =: rope_inv_jllamarope_
mha_full =: mha_full_jllamaattn_
mha_prefill_cached =: mha_prefill_cached_jllamaattn_
mha_step =: mha_step_jllamaattn_
kv_empty =: kv_empty_jllamaattn_
split_heads =: split_heads_jllamaattn_
merge_heads =: merge_heads_jllamaattn_
attention1 =: attention1_jllamaattn_

NB. ---------------------------------------------------------------
test_rotate_half =: 3 : 0
  NB. rotate_half is NORMAL pair rotation alias (not NeoX half-rotate)
  got =. rotate_half 1 2 3 4
  assert. (_2 1 _4 3) -: got
  1
)

test_rope_rank2 =: 3 : 0
  x =. 2 4 $ 0 1 2 3 4 5 6 7
  got =. rope x
  exp =. 2 4 $ 0 1 2 3 _2.0461457005669237 6.0673954685722844 5.9297011691608255 7.0596490029216570
  assert. exp allclose got
  NB. position 0 is identity for first row
  assert. (0 { x) allclose 0 { got
  1
)

test_rope_rank3 =: 3 : 0
  x =. 2 2 4 $ i. 16
  got =. rope x
  exp =. 2 2 4 $ 0 1 2 3 4 5 6 7 _3.2508204163259506 11.5944886312764304 9.8895018374908190 11.0994483379249846 _4.4554951320849767 17.1215817949805762 13.8493025068208127 15.1392476729273123
  assert. exp allclose got
  1
)

test_rope_inv =: 3 : 0
  inv =. 10000 rope_inv 4
  NB. i in 0 2 -> theta^(-i/d) => 10000^0 , 10000^(-0.5) => 1 , 0.01
  assert. (1 0.01) allclose inv
  1
)

test_split_merge =: 3 : 0
  x =. 3 4 $ i. 12
  h =. 2 split_heads x
  assert. 3 2 2 -: $ h
  assert. x -: merge_heads h
  1
)

test_attention1_causal =: 3 : 0
  q =. k =. v =. 3 2 $ 1 0 0 1 1 1
  got =. attention1 q ; k ; v
  NB. row0 only sees pos0 -> v0 = 1 0
  assert. (1 0) allclose 0 { got
  NB. rows sum not required; check finite and shape
  assert. 3 2 -: $ got
  1
)

NB. Fixed RNG weights (numpy default_rng(1) rounded) for cache==full
mk_mha_case =: 3 : 0
  n_head =. 2
  X =. 3 4 $ 0.345584 0.821618 0.330437 _1.303157 0.905356 0.446375 _0.536953 0.581118 0.364572 0.294132 0.028422 0.546713
  Wq =. 4 4 $ _0.368227 _0.081455 _0.24106 0.299423 0.019861 _0.146228 _0.390954 _0.128596 0.004071 _0.137801 0.647032 0.503362 _1.355581 _0.944507 _0.087386 _0.211095
  Wk =. 4 4 $ 0.106821 0.108661 1.058919 _0.55601 _0.188803 1.021386 0.323351 0.331532 _0.257003 _0.824038 0.083732 0.054507 _0.613676 _0.341613 _0.036022 _0.472376
  Wv =. 4 4 $ _0.049135 0.047742 0.017793 _0.253146 0.296874 0.445583 0.160424 _0.409115 0.365826 _0.25072 0.43958 _0.535894 0.457234 _0.010032 _0.624374 _0.15695
  Wo =. 4 4 $ 0.027051 0.136396 _0.491094 _0.553687 0.099792 _0.233375 0.117753 0.37976 _0.824394 0.127194 0.612323 _0.148763 _0.405407 0.376122 0.126723 0.447942
  X ; n_head ; Wq ; Wk ; Wv ; Wo
)

test_mha_full_shape =: 3 : 0
  c =. mk_mha_case ''
  out =. mha_full c
  assert. 3 4 -: $ out
  exp =. 3 4 $ _0.71913336681717 _0.1163280473765656 0.7800745807019718 _0.08450540879898068 _0.07316894989646744 _0.15673247827021225 0.16048404533588145 _0.06164647608763159 0.07377329891566273 _0.14922917169513367 0.005437346467964767 _0.10501267221249232
  assert. exp allclose out
  1
)

NB. *** M2 exit criterion: cached prefill matches full recompute ***
test_kv_parity =: 3 : 0
  c =. mk_mha_case ''
  full =. mha_full c
  'cached kc vc' =. mha_prefill_cached c
  assert. full allclose cached
  assert. 3 2 2 -: $ kc
  assert. 3 2 2 -: $ vc
  1
)

NB. Longer random case (no external golden): cache == full only
test_kv_parity_random =: 3 : 0
  n_tok =. 5
  n_head =. 3
  d_head =. 4
  n_embd =. n_head * d_head
  NB. deterministic pseudo-random from integers
  roll =. 3 : '0.1 * <: 20 | y + 3 * i. y'
  X =. (n_tok , n_embd) $ 0.05 * <: 17 | 100 + i. n_tok * n_embd
  Wq =. (n_embd , n_embd) $ 0.02 * <: 19 | 3 + i. *: n_embd
  Wk =. (n_embd , n_embd) $ 0.02 * <: 19 | 7 + i. *: n_embd
  Wv =. (n_embd , n_embd) $ 0.02 * <: 19 | 11 + i. *: n_embd
  Wo =. (n_embd , n_embd) $ 0.02 * <: 19 | 13 + i. *: n_embd
  c =. X ; n_head ; Wq ; Wk ; Wv ; Wo
  full =. mha_full c
  'cached kc vc' =. mha_prefill_cached c
  assert. full allclose cached
  1
)

NB. After prefill, one more step equals full on prefix+new
test_step_extends =: 3 : 0
  'X n_head Wq Wk Wv Wo' =. mk_mha_case ''
  pref =. 2 {. X
  'o_pref kc vc' =. mha_prefill_cached pref ; n_head ; Wq ; Wk ; Wv ; Wo
  x_new =. ,: 2 { X
  'o_new kc2 vc2' =. mha_step x_new ; n_head ; Wq ; Wk ; Wv ; Wo ; kc ; vc ; 2
  full =. mha_full X ; n_head ; Wq ; Wk ; Wv ; Wo
  got =. o_pref , o_new
  assert. full allclose got
  1
)

NB. ---------------------------------------------------------------
TESTS =: 'test_rotate_half'; 'test_rope_inv'; 'test_rope_rank2'; 'test_rope_rank3'; 'test_split_merge'; 'test_attention1_causal'; 'test_mha_full_shape'; 'test_kv_parity'; 'test_kv_parity_random'; 'test_step_extends'

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
    smoutput 'jllamatestm2: ' , (": failed) , ' failed'
    0
  else.
    smoutput 'jllamatestm2: ' , (": # TESTS) , ' passed'
    1
  end.
)

cocurrent 'base'
