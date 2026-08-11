NB. M7 tests: temperature / top-k / top-p sampling + EOS stop

cocurrent 'jllamatestm7'

softmax =: softmax_jllamatensor_
sample_greedy =: sample_greedy_jllamasample_
top_k_filter =: top_k_filter_jllamasample_
top_p_filter =: top_p_filter_jllamasample_
sample_from_probs =: sample_from_probs_jllamasample_
sample_next =: sample_next_jllamasample_
sample_cfg_pack =: sample_cfg_pack_jllamasample_
default_cfg =: default_cfg_jllamasample_
rng_u01 =: rng_u01_jllamasample_
make_synthetic =: make_synthetic_jllamamodel_
generate =: generate_jllamamodel_
generate_sample =: generate_sample_jllamamodel_
logits_last =: logits_last_jllamamodel_
forward_full =: forward_full_jllamamodel_

test_cfg_defaults =: 3 : 0
  c =. sample_cfg_pack ''
  assert. 6 = # c
  assert. 0 = 0 { c
  c2 =. sample_cfg_pack 0.8
  assert. 0.8 = 0 { c2
  assert. 0 = 1 { c2
  1
)

test_greedy_sample_next =: 3 : 0
  z =. 0.1 2.5 _1 0.3
  'tok seed' =. sample_next (<0 0 1 0 _1 1) , (<z)
  assert. 1 = tok
  assert. 0 = seed
  assert. 1 = sample_greedy z
  1
)

test_top_k =: 3 : 0
  z =. 1 5 2 4 3
  z2 =. 2 top_k_filter z
  assert. 5 = 1 { z2
  assert. 4 = 3 { z2
  assert. *./ MASK_jllamasample_ = 0 2 4 { z2
  z3 =. 0 top_k_filter z
  assert. z -: z3
  1
)

test_top_p =: 3 : 0
  pr =. 0.1 0.5 0.2 0.15 0.05
  pr =. pr % +/ pr
  p =. 0.6 top_p_filter pr
  NB. nucleus should keep largest mass first: idx1 (0.5) then idx2 (0.2) => cum 0.7
  assert. 0 = 0 { p
  assert. 0 < 1 { p
  assert. 0 < 2 { p
  assert. 1e_9 > | 1 - +/ p
  p1 =. 1 top_p_filter pr
  assert. pr allclose_jllamatensor_ p1
  1
)

test_sample_from_probs =: 3 : 0
  pr =. 0 0 1 0
  assert. 2 = 0.0 sample_from_probs pr
  assert. 2 = 0.5 sample_from_probs pr
  assert. 2 = 0.999 sample_from_probs pr
  pr2 =. 0.25 0.25 0.25 0.25
  assert. 0 = 0.1 sample_from_probs pr2
  assert. 1 = 0.3 sample_from_probs pr2
  assert. 3 = 0.9 sample_from_probs pr2
  1
)

test_rng_deterministic =: 3 : 0
  'u1 s1' =. rng_u01 123
  'u2 s2' =. rng_u01 123
  assert. u1 = u2
  assert. s1 = s2
  'u3 s3' =. rng_u01 s1
  assert. u3 ~: u1
  1
)

test_generate_greedy_compat =: 3 : 0
  m =. make_synthetic 24 ; 8 ; 2 ; 2 ; 16
  a =. m generate 1 2 3 ; 4
  cfg =. 0 ; 0 ; 1 ; 0 ; _1 ; 1
  b =. m generate_sample (<1 2 3) , (<4) , (<cfg)
  assert. a -: b
  assert. 7 = # a
  1
)

test_generate_sample_seeded =: 3 : 0
  m =. make_synthetic 32 ; 8 ; 2 ; 2 ; 16
  cfg =. 0.9 ; 8 ; 0.95 ; 42 ; _1 ; 1
  a =. m generate_sample (<1 2 3) , (<5) , (<cfg)
  b =. m generate_sample (<1 2 3) , (<5) , (<cfg)
  assert. a -: b
  assert. 8 = # a
  assert. *./ (0 <: a) *. a < 32
  NB. different seed can differ (usually)
  cfg2 =. 0.9 ; 8 ; 0.95 ; 99 ; _1 ; 1
  c =. m generate_sample (<1 2 3) , (<5) , (<cfg2)
  assert. 8 = # c
  1
)

test_eos_stop =: 3 : 0
  m =. make_synthetic 16 ; 8 ; 2 ; 1 ; 16
  NB. force eos by using temp=0 after constructing logits path:
  NB. pick a prompt and greedy; then use eos_id = first generated token with n_new large
  g =. m generate 1 2 ; 1
  eos =. {: g
  cfg =. 0 ; 0 ; 1 ; 0 ; eos ; 1
  out =. m generate_sample (<1 2) , (<10) , (<cfg)
  NB. should stop right after emitting eos (length prompt+1)
  assert. 3 = # out
  assert. eos = {: out
  NB. stop_on_eos=0 continues
  cfg2 =. 0 ; 0 ; 1 ; 0 ; eos ; 0
  out2 =. m generate_sample (<1 2) , (<4) , (<cfg2)
  assert. 6 = # out2
  1
)

test_temp_changes_dist =: 3 : 0
  NB. unit-level: very peaky logits + high temp still valid token
  z =. 10 0 0 0 0
  't1 s1' =. sample_next (<0.01 0 1 1 _1 1) , (<z)
  assert. 0 = t1
  't2 s2' =. sample_next (<5 0 1 1 _1 1) , (<z)
  assert. (t2 >: 0) *. t2 < 5
  1
)

TESTS =: 'test_cfg_defaults'; 'test_greedy_sample_next'; 'test_top_k'; 'test_top_p'; 'test_sample_from_probs'; 'test_rng_deterministic'; 'test_generate_greedy_compat'; 'test_generate_sample_seeded'; 'test_eos_stop'; 'test_temp_changes_dist'

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
    smoutput 'jllamatestm7: ' , (": failed) , ' failed'
    0
  else.
    smoutput 'jllamatestm7: ' , (": # TESTS) , ' passed'
    1
  end.
)

cocurrent 'base'
