# cli/

Shell front-end for jllama (M8).

| File | Role |
|------|------|
| `cli.ijs` | locale `jllamacli` — parse args, load model, generate, print |
| `../jllama_cli.ijs` | jconsole entry (load + `main` + exit) |
| `../bin/jllama_cli` | bash wrapper (macOS/Linux) |
| `../bin/jllama_cli.cmd` | cmd wrapper (Windows) |

The `#!` line in `jllama_cli.ijs` is for Unix only. J skips it when loading the script, so it stays for macOS; Windows must invoke `jconsole.exe` (or the `.cmd` wrapper).

## Usage

```sh
bin/jllama_cli -m test/fixtures/tiny_parity_f16.gguf -p ab -n 3
bin/jllama_cli -m test/fixtures/tiny_parity_f16.gguf -p ab -n 3 --tokens
bin/jllama_cli --help
```

Windows:

```bat
bin\jllama_cli.cmd -m test\fixtures\tiny_parity_f16.gguf -p ab -n 3
"C:\Users\tmcguire\j9.8\bin\jconsole.exe" C:\Users\tmcguire\jdev\jllamaGPU\jllama_cli.ijs --help
```

Override jconsole:

```sh
JCONSOLE=/path/to/jconsole bin/jllama_cli -m MODEL.gguf -p hi -n 8
```

```bat
set JCONSOLE=C:\Users\tmcguire\j9.8\bin\jconsole.exe
bin\jllama_cli.cmd -m MODEL.gguf -p hi -n 8
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
