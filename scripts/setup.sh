#!/usr/bin/env bash
set -euo pipefail

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
python_bin="${PYTHON_BIN:-python3}"
venv_dir="${LILAC_VENV_DIR:-$repo_root/.venv}"

if [[ ! -x "$venv_dir/bin/python" ]]; then
  "$python_bin" -m venv "$venv_dir"
fi
"$venv_dir/bin/python" -m pip install --upgrade pip
"$venv_dir/bin/python" -m pip install -r "$repo_root/requirements.txt"

echo "Environment ready: $venv_dir"
echo "Fetching the small reference source (not the 300+ MB weights)."
PATH="$venv_dir/bin:$PATH" "$repo_root/scripts/fetch_reference.sh"
echo
echo "To include the released checkpoint weights:"
echo "  PATH=$venv_dir/bin:\$PATH $repo_root/scripts/fetch_reference.sh --weights"
