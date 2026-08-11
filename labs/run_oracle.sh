#!/usr/bin/env bash
# Run libllama greedy oracle; print only the PROMPT/GEN/FULL result line.
# Usage: run_oracle.sh MODEL.gguf PROMPT N_NEW [--ids ID...]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ORACLE="${JLLAMA_ORACLE:-$ROOT/tools/oracle_greedy}"
if [[ ! -x "$ORACLE" ]]; then
  echo "oracle missing: $ORACLE (build labs/oracle_greedy.c)" >&2
  exit 1
fi
# stderr is noisy (metal/backends); keep stdout result only
"$ORACLE" "$@" 2>/dev/null | awk '/^PROMPT /{print; found=1} END{exit found?0:1}'
