#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
runtime_root="${VLFM_RUNTIME_ROOT:-/home/woom/vlfm}"
python_bin="${VLFM_PYTHON:-/home/woom/miniconda3/envs/vlfm/bin/python}"
cd "$runtime_root"
exec "$python_bin" "$script_dir/run_person_search_managed.py" "$@"
