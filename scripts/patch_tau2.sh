#!/usr/bin/env bash
# Apply patches/*.patch to the tau2-bench clone. Idempotent: already-applied patches are skipped.
#
# 0001: Environment.set_state syncs tools.db and user_tools.db only when they are the same type.
#       Telecom keeps a separate TelecomUserDB, which upstream overwrites with the agent-side
#       TelecomDB whenever a task carries initialization_data.agent_data, breaking every user tool.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=${TAU2_ROOT:-upstream/tau2-bench}
if [ ! -d "$ROOT/.git" ]; then
  echo "no tau2-bench clone at $ROOT (set TAU2_ROOT, or see README.md)" >&2; exit 1
fi
for p in patches/*.patch; do
  abs="$(cd "$(dirname "$p")" && pwd)/$(basename "$p")"
  if git -C "$ROOT" apply --check "$abs" 2>/dev/null; then
    git -C "$ROOT" apply "$abs" && echo "applied $p"
  elif git -C "$ROOT" apply --reverse --check "$abs" 2>/dev/null; then
    echo "already applied $p"
  else
    echo "cannot apply $p (upstream changed?)" >&2; exit 1
  fi
done
