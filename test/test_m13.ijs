NB. M13 tests: GQA (n_head_kv < n_head) attention + stack + generate

cocurrent 'jllamatestm13'

load ROOT_jllama_ , 'core/tensor.ijs'
mha_full =: mha_full_jllamaattn_
mha_prefill_cached =: mha_prefill_cached_jllamaattn_
expand_kv =: expand_kv_jllamaattn_
make_synthetic =: make_synthetic_jllamamodel_
make_layer =: make_layer_jllamamodel_
block_full =: block_full_jllamablock_
block_prefill_cached =: block_prefill_cached_jllamablock_
forward_full =: forward_full_jllamamodel_
forward_prefill =: forward_prefill_jllamamodel_
forward_step =: forward_step_jllamamodel_
generate =: generate_jllamamodel_
generate_fullrecompute =: generate_fullrecompute_jllamamodel_
DEFAULT_THETA =: DEFAULT_THETA_jllamarope_

NB. ---------------------------------------------------------------
test_expand_kv =: 3 : 0
  NB. n_tok=2, n_kv=2, d=3 ; n_rep=2 -> 2 x 4 x 3
  kv =. 2 2 3 $ i. 12
  got =. 2 expand_kv kv
  assert. 2 4 3 -: $ got
  NB. heads 0,1,2,3 = kv0,kv0,kv1,kv1
  assert. (0 { 1 0 2 |: kv) -: 0 { 1 0 2 |: got
  assert. (0 { 1 0 2 |: kv) -: 1 { 1 0 2 |: got
  assert. (1 { 1 0 2 |: kv) -: 2 { 1 0 2 |: got
  assert. (1 { 1 0 2 |: kv) -: 3 { 1 0 2 |: got
  assert. kv -: 1 expand_kv kv
  1
)

NB. Tiny GQA MHA weights: n_embd=8, n_head=4, n_kv=2, d_head=2
mk_gqa_case =: 3 : 0
  n_head =. 4
  n_embd =. 8
  n_kv_dim =. 4
  X =. (3 , n_embd) $ 0.05 * <: 17 | 100 + i. 3 * n_embd
  Wq =. (n_embd , n_embd) $ 0.02 * <: 19 | 3 + i. *: n_embd
  Wk =. (n_embd , n_kv_dim) $ 0.02 * <: 19 | 7 + i. n_embd * n_kv_dim
  Wv =. (n_embd , n_kv_dim) $ 0.02 * <: 19 | 11 + i. n_embd * n_kv_dim
  Wo =. (n_embd , n_embd) $ 0.02 * <: 19 | 13 + i. *: n_embd
  X ; n_head ; Wq ; Wk ; Wv ; Wo
)

test_gqa_mha_kv_parity =: 3 : 0
  c =. mk_gqa_case ''
  full =. mha_full c
  'cached kc vc' =. mha_prefill_cached c
  assert. full allclose cached
  assert. 3 2 2 -: $ kc
  assert. 3 2 2 -: $ vc
  assert. 3 8 -: $ full
  1
)

test_gqa_block_kv_parity =: 3 : 0
  n_embd =. 8
  n_head =. 4
  n_ff =. 16
  n_kv =. 2
  layer =. make_layer n_embd ; n_head ; n_ff ; 7 ; n_kv
  x =. (5 , n_embd) $ 0.03 * <: 19 | 50 + i. 5 * n_embd
  full =. block_full (<x) , (<n_head) , layer
  'cached kc vc' =. block_prefill_cached (<x) , (<n_head) , layer
  assert. full allclose cached
  assert. 5 2 2 -: $ kc
  assert. 5 8 -: $ full
  1
)

test_gqa_stack_kv_parity =: 3 : 0
  NB. n_vocab;n_embd;n_head;n_layer;n_ff;theta;seed;n_head_kv
  m =. make_synthetic 16 ; 8 ; 4 ; 2 ; 16 ; DEFAULT_THETA ; 0 ; 2
  'hp wte layers ln_f lm_head' =. > m
  'n_vocab n_embd n_head n_layer n_ff theta n_head_kv' =. hp
  assert. 4 2 -: n_head , n_head_kv
  L =. > 0 { layers
  'attn_n wq wk wv wo ffn_n wg wu wd' =. L
  assert. 8 8 -: $ wq
  assert. 8 4 -: $ wk
  assert. 8 4 -: $ wv
  ids =. 1 2 3 5 8
  full =. m forward_full ids
  'cached caches' =. m forward_prefill ids
  assert. full allclose cached
  assert. 2 = # caches
  'kc vc' =. > 0 { caches
  assert. 5 2 2 -: $ kc
  1
)

test_gqa_step_matches_full =: 3 : 0
  m =. make_synthetic 16 ; 8 ; 4 ; 2 ; 16 ; DEFAULT_THETA ; 1 ; 2
  ids =. 3 1 4 1
  full =. m forward_full ids
  'h0 c0' =. m forward_prefill }: ids
  'h1 c1' =. m forward_step (<{: ids) , (<c0) , (<3)
  assert. ({: full) allclose , h1
  1
)

test_gqa_generate_matches_recompute =: 3 : 0
  m =. make_synthetic 32 ; 8 ; 4 ; 2 ; 16 ; DEFAULT_THETA ; 3 ; 2
  prompt =. 1 4 2
  n_new =. 5
  a =. m generate prompt ; n_new
  b =. m generate_fullrecompute prompt ; n_new
  assert. a -: b
  assert. (n_new + # prompt) = # a
  assert. *./ (0 <: a) *. a < 32
  1
)

test_mha_still_works =: 3 : 0
  NB. default synthetic remains MHA (n_head_kv = n_head)
  m =. make_synthetic 16 ; 8 ; 2 ; 1 ; 16
  'hp wte layers ln_f lm_head' =. > m
  'n_vocab n_embd n_head n_layer n_ff theta n_head_kv' =. hp
  assert. n_head = n_head_kv
  ids =. m generate (1 2) ; 3
  assert. 5 = # ids
  1
)

run =: 3 : 0
  smoutput 'M13 tests...'
  assert. test_expand_kv ''
  smoutput '  expand_kv ok'
  assert. test_gqa_mha_kv_parity ''
  smoutput '  gqa_mha_kv_parity ok'
  assert. test_gqa_block_kv_parity ''
  smoutput '  gqa_block_kv_parity ok'
  assert. test_gqa_stack_kv_parity ''
  smoutput '  gqa_stack_kv_parity ok'
  assert. test_gqa_step_matches_full ''
  smoutput '  gqa_step_matches_full ok'
  assert. test_gqa_generate_matches_recompute ''
  smoutput '  gqa_generate_matches_recompute ok'
  assert. test_mha_still_works ''
  smoutput '  mha_still_works ok'
  smoutput 'M13 tests passed (7)'
  1
)

cocurrent 'base'
