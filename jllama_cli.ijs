NB. jllama_cli - shell entry for M8
NB. Prefer:
NB.   bin/jllama_cli -m MODEL.gguf -p PROMPT ...
NB. or:
NB.   /Applications/j9.8/bin/jconsole /path/to/jllama_cli.ijs -m ...
NB.
NB. Loads jllama.ijs then runs main_jllamacli_ and exits.

cocurrent 'base'

NB. Resolve this script's directory via 4!:3, load, run main.
3 : 0 ''
  scripts =. 4!:3 ''
  hit =. scripts #~ +./@('jllama_cli.ijs'&E.)@> scripts
  if. #hit do.
    p =. > {: hit
    ROOTCLI =. ((p i: '/') {. p) , '/'
  else.
    ROOTCLI =. (1!:43 '') , '/'
  end.
  load ROOTCLI , 'jllama.ijs'
  loadcore_jllama_ ''
  jrequire_jllama_ 'cli/cli.ijs'
  main_jllamacli_ ''
)
