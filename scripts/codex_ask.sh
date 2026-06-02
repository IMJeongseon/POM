#!/usr/bin/env bash
# codex_ask.sh — invoke Codex CLI non-interactively on a task spec.
#
# Usage:
#   ./scripts/codex_ask.sh "<inline prompt>"
#   ./scripts/codex_ask.sh <path/to/task_file.md>
#
# The wrapper:
#   - locates the codex binary bundled with the VSCode ChatGPT extension
#   - runs with workspace-write sandbox rooted at the MLLAB/FM repo
#   - bypasses git-repo check (we may not always be inside a git worktree)
#   - writes the last agent message to a timestamped file under scripts/codex_runs/

set -euo pipefail

# Auto-locate the codex binary bundled with the VSCode ChatGPT extension
# (the version-stamped directory changes on extension updates).
CODEX_BIN="$(ls -1 /home/jeongseon43/.vscode-server/extensions/openai.chatgpt-*-linux-x64/bin/linux-x86_64/codex 2>/dev/null | sort -V | tail -1)"
REPO_DIR="/home/jeongseon43/MLLAB/FM"
LOG_DIR="$REPO_DIR/scripts/codex_runs"

if [[ ! -x "$CODEX_BIN" ]]; then
    echo "codex binary not found at $CODEX_BIN" >&2
    exit 1
fi

if [[ $# -lt 1 ]]; then
    echo "usage: $(basename "$0") '<prompt>' | <path/to/task.md>" >&2
    exit 1
fi

# Resolve prompt: file → cat contents; else treat as inline string
if [[ -f "$1" ]]; then
    PROMPT="$(cat "$1")"
    SRC_LABEL="file:$1"
else
    PROMPT="$1"
    SRC_LABEL="inline"
fi

mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
LAST_MSG_FILE="$LOG_DIR/codex_${TS}.last_message.txt"

echo "[codex_ask] running codex exec (source=$SRC_LABEL)"
echo "[codex_ask] last-message → $LAST_MSG_FILE"
echo "─────────────────────────────────────────────"

"$CODEX_BIN" exec \
    --cd "$REPO_DIR" \
    --sandbox workspace-write \
    --skip-git-repo-check \
    --color never \
    -o "$LAST_MSG_FILE" \
    "$PROMPT"

echo "─────────────────────────────────────────────"
echo "[codex_ask] done. last message saved to: $LAST_MSG_FILE"
