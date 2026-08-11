NB. Inspect stories15M GGUF
3 : 0 ''
  loadcore_jllama_ ''
  p =. (jllama_root '') , 'models/stories15M.F16.gguf'
  g =. gguf_load_jllamagguf_ p
  smoutput 'model ' , g gguf_meta_jllamagguf_ 'tokenizer.ggml.model'
  toks =. g gguf_meta_jllamagguf_ 'tokenizer.ggml.tokens'
  smoutput 'ntoks ' , ": # toks
  i =. 0
  while. i < 20 do.
    smoutput (": i) , ' [' , (> i { toks) , ']'
    i =. i + 1
  end.
  smoutput 'bos ' , ": g gguf_meta_default_jllamagguf_ 'tokenizer.ggml.bos_token_id' ; _1
  smoutput 'eos ' , ": g gguf_meta_default_jllamagguf_ 'tokenizer.ggml.eos_token_id' ; _1
  smoutput 'unk ' , ": g gguf_meta_default_jllamagguf_ 'tokenizer.ggml.unknown_token_id' ; _1
  smoutput 'add_bos ' , ": g gguf_meta_default_jllamagguf_ 'tokenizer.ggml.add_bos_token' ; _1
  smoutput 'add_eos ' , ": g gguf_meta_default_jllamagguf_ 'tokenizer.ggml.add_eos_token' ; _1
  scores =. g gguf_meta_default_jllamagguf_ 'tokenizer.ggml.scores' ; (0 $ 0)
  smoutput 'nscores ' , ": # scores
  smoutput 'scores0..8 ' , ": 9 {. scores
  tt =. g gguf_meta_default_jllamagguf_ 'tokenizer.ggml.token_type' ; (0 $ 0)
  smoutput 'tt0..20 ' , ": 21 {. tt
  want =. 'Once' ; 'upon' ; 'time' ; '▁Once' ; '▁upon' ; '▁time' ; '▁' ; 'a' ; '▁a' ; 'Once upon a time'
  for_w. want do.
    w =. > w
    found =. _1
    i =. 0
    while. i < # toks do.
      if. w -: > i { toks do. found =. i break. end.
      i =. i + 1
    end.
    smoutput '''' , w , ''' -> ' , ": found
  end.
  i =. 0
  nb =. 0
  while. i < # toks do.
    t =. > i { toks
    if. (7 = # t) *. ('<0x' -: 3 {. t) *. ('>' = {: t) do. nb =. nb + 1 end.
    i =. i + 1
  end.
  smoutput 'byte_pieces ' , ": nb
  2!:55 ] 0
)
