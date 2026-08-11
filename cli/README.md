# cli/

Shell front-end for jllama (M8).

| File | Role |
|------|------|
| `cli.ijs` | locale `jllamacli` — parse args, load model, generate, print |
| `../jllama_cli.ijs` | jconsole entry (load + `main` + exit) |
| `../bin/jllama_cli` | bash wrapper |

## Usage

```sh
bin/jllama_cli -m test/fixtures/tiny_parity_f16.gguf -p ab -n 3
bin/jllama_cli -m test/fixtures/tiny_parity_f16.gguf -p ab -n 3 --tokens
bin/jllama_cli --help
```

Override jconsole:

```sh
JCONSOLE=/path/to/jconsole bin/jllama_cli -m MODEL.gguf -p hi -n 8
```

## Locale `jllamacli`

| Verb | Role |
|------|------|
| `parse_args` | boxed argv → opts list |
| `run_opts` | opts → exit code (prints text) |
| `run` | argv → exit code (try/catch) |
| `main` | `cli_argv` + `2!:55` |
| `cli_argv` | strip jconsole + `*.ijs` from `ARGV_z_` |

Sampling flags map to M7 cfg: `temp ; top_k ; top_p ; seed ; eos_id ; stop_on_eos`.
