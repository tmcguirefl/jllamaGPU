#!/Users/tomdevel/j9.8/bin/jconsole
NB. jllamaGPU — GNU GPL v3 only. Copyright (C) 2026 Tom McGuire. See LICENSE.
NB. jllama_cli - standalone shell entry
NB.
NB. From a clone of this tree:
NB.   ./jllama_cli.ijs -m MODEL.gguf -p PROMPT ...
NB.
NB. Loads sibling scripts from this directory (ROOT). No ~temp publish step.
NB. Later: manifest.ijs will install the same tree as a J addon.

cocurrent 'base'

NB. This script is already in 4!:3; use it to find the clone root.
3 : 0 ''
  p =. > {: 4!:3 ''
  load (((p i: '/') {. p) , '/') , 'sysutils.ijs'
  setroot_jllamasys_ 'jllama_cli.ijs'
  i. 0 0
)

load ROOT_jllamasys_ , 'jgpu.ijs'
load ROOT_jllamasys_ , 'core/rope.ijs'
load ROOT_jllamasys_ , 'core/attention.ijs'
load ROOT_jllamasys_ , 'core/block.ijs'
load ROOT_jllamasys_ , 'core/sample.ijs'
load ROOT_jllamasys_ , 'core/model.ijs'
load ROOT_jllamasys_ , 'io/gguf.ijs'
load ROOT_jllamasys_ , 'io/vocab.ijs'
load ROOT_jllamasys_ , 'core/arch.ijs'
". '0!:0 Llama3'
load ROOT_jllamasys_ , 'cli/cli.ijs'

main_jllamacli_ ''
