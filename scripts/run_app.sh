#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$project_dir/src${PYTHONPATH:+:$PYTHONPATH}"
native_lib="$project_dir/.native/lib/usr/lib/x86_64-linux-gnu"
if [[ -d "$native_lib" ]]; then
  export LD_LIBRARY_PATH="$native_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export MPLCONFIGDIR="${MPLCONFIGDIR:-$project_dir/.matplotlib}"
mkdir -p "$MPLCONFIGDIR"
exec "$project_dir/.venv/bin/python" -m ui.app "$@"

