#!/usr/bin/env bash
set -euo pipefail

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
reference_dir="${LILAC_REFERENCE_DIR:-$repo_root/.cache/reference/amx-reasoning-v1-instruct}"
revision="${LILAC_REFERENCE_REVISION:-b144ee0138929f0181b9219177f98fc7c8d259c9}"
with_weights=0

if [[ "${1:-}" == "--weights" ]]; then
  with_weights=1
  shift
fi
if [[ $# -ne 0 ]]; then
  echo "usage: $0 [--weights]" >&2
  exit 2
fi
if ! command -v hf >/dev/null 2>&1; then
  echo "hf CLI is missing; run scripts/setup.sh first" >&2
  exit 1
fi

mkdir -p "$reference_dir"
args=(
  download gdiamos/amx-reasoning-v1-instruct
  --repo-type model
  --revision "$revision"
  --local-dir "$reference_dir"
  --exclude paper.pdf
  --exclude LICENSE
)
if [[ $with_weights -eq 0 ]]; then
  args+=(--exclude model.safetensors)
fi

echo "Fetching reference source at $revision"
hf "${args[@]}"
echo "Reference cache: $reference_dir"
if [[ $with_weights -eq 0 ]]; then
  echo "Weights omitted. Re-run with --weights when you want local inference."
fi
