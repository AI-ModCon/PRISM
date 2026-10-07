#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  tools/ci/local_quality.sh [changed|full] [--install] [--python <path>]

Modes:
  changed   Run ruff/mypy on changed Python files only (default).
  full      Run ruff/mypy on the full configured scopes.

Options:
  --install       Install CI quality dependencies from requirements/ci.txt first.
  --python <path> Python executable to use (default: $PRISM_CI_PYTHON or python3).
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODE="changed"
INSTALL_DEPS=0
PY_BIN="${PRISM_CI_PYTHON:-python3}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    changed|full)
      MODE="$1"
      shift
      ;;
    --install)
      INSTALL_DEPS=1
      shift
      ;;
    --python)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --python" >&2
        usage
        exit 2
      fi
      PY_BIN="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

cd "$REPO_ROOT"

if [[ "$INSTALL_DEPS" -eq 1 ]]; then
  "$PY_BIN" -m pip install -r "$REPO_ROOT/requirements/ci.txt"
fi

if [[ "$MODE" == "full" ]]; then
  echo "[quality] Running full ruff check..."
  "$PY_BIN" -m ruff check src tests tools scripts examples

  echo "[quality] Running full mypy check..."
  "$PY_BIN" -m mypy src

  echo "[quality] Full checks passed."
  exit 0
fi

CHANGED_PY_FILES=()
while IFS= read -r line; do
  if [[ -n "$line" ]]; then
    CHANGED_PY_FILES+=("$line")
  fi
done < <(
  {
    git diff --name-only -- '*.py'
    git diff --name-only --cached -- '*.py'
    git ls-files --others --exclude-standard '*.py'
  } | sed '/^$/d' | sort -u
)

if [[ "${#CHANGED_PY_FILES[@]}" -eq 0 ]]; then
  echo "[quality] No changed Python files detected."
  exit 0
fi

echo "[quality] Running ruff on changed files (${#CHANGED_PY_FILES[@]})..."
"$PY_BIN" -m ruff check --fix "${CHANGED_PY_FILES[@]}"
"$PY_BIN" -m ruff format "${CHANGED_PY_FILES[@]}"

SRC_CHANGED=()
for file in "${CHANGED_PY_FILES[@]}"; do
  if [[ "$file" == src/*.py ]]; then
    SRC_CHANGED+=("$file")
  fi
done

if [[ "${#SRC_CHANGED[@]}" -gt 0 ]]; then
  echo "[quality] Running mypy on changed src files (${#SRC_CHANGED[@]})..."
  "$PY_BIN" -m mypy "${SRC_CHANGED[@]}"
else
  echo "[quality] No changed src/*.py files; skipping mypy."
fi

echo "[quality] Changed-file checks passed."
