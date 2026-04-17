#!/usr/bin/env bash
# Stop tracking large generated files that should not live in git (Phase 7).
# Run from repo root after review: bash scripts/git-untrack-artifacts.sh
# Then: git status && git commit -m "chore: stop tracking experiment artifacts"

set -euo pipefail
cd "$(dirname "$0")/.."

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Not a git repository." >&2
  exit 1
fi

git rm -r --cached experiments/ndtictactoe/models-and-data 2>/dev/null || true

git ls-files -- '*.model' | while IFS= read -r f; do
  [ -z "$f" ] && continue
  git rm --cached --ignore-unmatch "$f" || true
done

echo "Done. Review with: git status"
echo "Commit when ready; blobs remain in old commits until history is rewritten."
