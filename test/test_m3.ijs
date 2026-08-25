NB. M3 tests: SwiGLU block, stack, greedy generate, cache==full

cocurrent 'jllamatestm3'

load ROOT_jllamasys_ , 'core/tensor.ijs'
ffn_swiglu =: ffn_swiglu_jllamablock_
block_full =: block_full_jllamablock_
block_prefill_cached =: block_prefill_cached_jllamablock_
make_synthetic =: make_synthetic_jllamamodel_
make_layer =: make_layer_jllamamodel_
forward_full =: forward_full_jllamamodel_
forward_prefill =: forward_prefill_jllamamodel_
forward_step =: forward_step_jllamamodel_
generate =: generate_jllamamodel_
generate_fullrecompute =: generate_fullrecompute_jllamamodel_
logits_last =: logits_last_jllamamodel_
embed =: embed_jllamamodel_

NB. ---------------------------------------------------------------
test_ffn_swiglu =: 3 : 0
  x =. 2 3 $ 0.1 0.2 _0.1 0.3 _0.2 0.4
  wg =. 3 4 $ 0.05 * i. 12
  wu =. 3 4 $ 0.04 * 1 + i. 12
  wd =. 4 3 $ 0.03 * i. 12
  got =. ffn_swiglu x ; wg ; wu ; wd
  gate =. silu x +/ . * wg
  up =. x +/ . * wu
  exp =. (gate * up) +/ . * wd
  assert. exp allclose got
  assert. 2 3 -: $ got
  1
)

test_block_kv_parity =: 3 : 0
  n_embd =. 8
  n_head =. 2
  n_ff =. 16
  layer =. make_layer n_embd ; n_head ; n_ff ; 7
  x =. (5 , n_embd) $ 0.03 * <: 19 | 50 + i. 5 * n_embd
  full =. block_full (<x) , (<n_head) , layer
  'cached kc vc' =. block_prefill_cached (<x) , (<n_head) , layer
  assert. full allclose cached
  assert. 5 2 4 -: $ kc
  1
)

test_stack_kv_parity =: 3 : 0
  m =. make_synthetic 16 ; 8 ; 2 ; 2 ; 16
  ids =. 1 2 3 5 8
  full =. m forward_full ids
  'cached caches' =. m forward_prefill ids
  assert. full allclose cached
  assert. 2 = # caches
  1
)

test_step_matches_full =: 3 : 0
  m =. make_synthetic 16 ; 8 ; 2 ; 2 ; 16
  ids =. 3 1 4 1
  full =. m forward_full ids
  'h0 c0' =. m forward_prefill }: ids
  'h1 c1' =. m forward_step (<{: ids) , (<c0) , (<3)
  assert. ({: full) allclose , h1
  1
)

test_generate_matches_recompute =: 3 : 0
  m =. make_synthetic 32 ; 8 ; 2 ; 2 ; 16
  prompt =. 1 4 2
  n_new =. 5
  a =. m generate prompt ; n_new
  b =. m generate_fullrecompute prompt ; n_new
  assert. a -: b
  assert. (n_new + # prompt) = # a
  assert. *./ (0 <: a) *. a < 32
  1
)

test_generate_deterministic =: 3 : 0
  m =. make_synthetic 24 ; 8 ; 2 ; 3 ; 12 ; 10000 ; 3
  prompt =. 0 1 2
  a =. m generate prompt ; 4
  b =. m generate prompt ; 4
  assert. a -: b
  1
)

test_logits_shape =: 3 : 0
  m =. make_synthetic 20 ; 8 ; 2 ; 1 ; 16
  h =. m forward_full 1 2 3
  z =. m logits_last h
  assert. 1 = #$ z
  assert. 20 = # z
  1
)

NB. ---------------------------------------------------------------
TESTS =: 'test_ffn_swiglu'; 'test_block_kv_parity'; 'test_stack_kv_parity'; 'test_step_matches_full'; 'test_generate_matches_recompute'; 'test_generate_deterministic'; 'test_logits_shape'

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
    smoutput 'jllamatestm3: ' , (": failed) , ' failed'
    0
  else.
    smoutput 'jllamatestm3: ' , (": # TESTS) , ' passed'
    1
  end.
)

cocurrent 'base'
