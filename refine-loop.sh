#!/usr/bin/env bash
set -euo pipefail

SESSIONS=5
START_ROUND="${1:-1}"
WORKDIR="/home/calmingstorm/Desktop/heimdall-build"
LOG_FILE="$WORKDIR/refine_log.txt"
BRANCH="refine/response-behavior"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

trap 'log "=== Refinement loop terminated ==="' EXIT

# --- Preflight ---
log "=== Preflight ==="
command -v claude >/dev/null 2>&1 || { log "FATAL: claude CLI not found"; exit 1; }
[[ -f "$WORKDIR/BUILD_STATUS.md" ]] || { log "FATAL: BUILD_STATUS.md missing"; exit 1; }
[[ -f "$WORKDIR/src/llm/system_prompt.py" ]] || { log "FATAL: system_prompt.py missing"; exit 1; }
[[ -f "$WORKDIR/data/context/architecture.md" ]] || { log "FATAL: architecture.md missing"; exit 1; }
log "Preflight: passed"

# --- Branch setup ---
cd "$WORKDIR"
if git rev-parse --verify "$BRANCH" >/dev/null 2>&1; then
    git checkout "$BRANCH" 2>/dev/null
else
    git checkout master 2>/dev/null
    git checkout -b "$BRANCH" 2>/dev/null
fi

read -r -d '' BUILD_PROMPT << 'PROMPTEOF' || true
You are round ROUND_NUM of 5 in a focused prompt refinement loop.

Read BUILD_STATUS.md — find "Round ROUND_NUM" and do EXACTLY what it says.

CRITICAL CONSTRAINTS:
- ONLY modify src/llm/system_prompt.py and data/context/architecture.md
- ONLY modify the SYSTEM_PROMPT_TEMPLATE string — do NOT touch CHAT_SYSTEM_PROMPT_TEMPLATE, Python logic, imports, or functions
- Do NOT create, modify, or run tests
- Do NOT modify any other files
- SYSTEM_PROMPT_TEMPLATE must stay under 4000 chars
- architecture.md must stay under 8500 chars
- Zero capability loss — refine and clarify, do not remove functionality
- Commit: git add src/llm/system_prompt.py data/context/architecture.md && git commit -m "[Round ROUND_NUM]: description"

If this is Round 2+, read the git diff from the previous round first:
  git diff HEAD~1 -- src/llm/system_prompt.py data/context/architecture.md

This shows you what the last round changed, so you can review before making your own changes.
PROMPTEOF

log "=== Prompt Refinement Loop Starting (rounds $START_ROUND-$SESSIONS) ==="
PASS_COUNT=0
FAIL_COUNT=0

for i in $(seq "$START_ROUND" "$SESSIONS"); do
    log "--- Round $i/$SESSIONS ---"
    PROMPT=$(echo "$BUILD_PROMPT" | sed "s/ROUND_NUM/$i/g")
    SESSION_START=$(date +%s)

    if cd "$WORKDIR" && echo "$PROMPT" | claude --print --dangerously-skip-permissions --output-format text --no-session-persistence 2>&1 | tee -a "$LOG_FILE"; then
        CLAUDE_STATUS="ok"
    else
        CLAUDE_STATUS="failed"
    fi

    ELAPSED=$(( $(date +%s) - SESSION_START ))
    VALID=true

    # Validate system prompt size
    PROMPT_SIZE=$(cd "$WORKDIR" && python3 -c "
from src.llm.system_prompt import SYSTEM_PROMPT_TEMPLATE
print(len(SYSTEM_PROMPT_TEMPLATE))
" 2>/dev/null || echo "0")
    if [[ "$PROMPT_SIZE" -gt 4200 ]]; then
        log "VALIDATION FAIL: system prompt is $PROMPT_SIZE chars (max 4200)"
        VALID=false
    else
        log "VALIDATION OK: system prompt $PROMPT_SIZE chars"
    fi

    # Validate architecture.md size
    ARCH_SIZE=$(wc -c < "$WORKDIR/data/context/architecture.md" 2>/dev/null || echo "0")
    if [[ "$ARCH_SIZE" -gt 8500 ]]; then
        log "VALIDATION FAIL: architecture.md is $ARCH_SIZE bytes (max 8500)"
        VALID=false
    else
        log "VALIDATION OK: architecture.md $ARCH_SIZE bytes"
    fi

    # Validate no other files were modified
    OTHER_CHANGES=$(git diff --name-only HEAD 2>/dev/null | grep -v 'src/llm/system_prompt.py' | grep -v 'data/context/architecture.md' | grep -v 'BUILD_STATUS.md' || true)
    if [[ -n "$OTHER_CHANGES" ]]; then
        log "VALIDATION FAIL: unexpected files modified: $OTHER_CHANGES"
        VALID=false
    fi

    # Validate detection functions still referenced in system prompt or architecture
    for keyword in "fabrication" "hedging" "premature"; do
        if ! grep -qi "$keyword" "$WORKDIR/data/context/architecture.md" 2>/dev/null; then
            log "VALIDATION FAIL: '$keyword' missing from architecture.md"
            VALID=false
        fi
    done

    if [[ "$VALID" == "true" ]]; then
        log "Round $i PASSED ($CLAUDE_STATUS) in ${ELAPSED}s — prompt: ${PROMPT_SIZE}c, arch: ${ARCH_SIZE}b"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        log "Round $i FAILED VALIDATION in ${ELAPSED}s"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi

    [[ $i -lt $SESSIONS ]] && { log "Pausing 5s..."; sleep 5; }
done

log "=== Refinement Loop Finished ==="
log "Results: $PASS_COUNT passed, $FAIL_COUNT failed out of $SESSIONS rounds"
log "Branch: $BRANCH"
log "Latest commit: $(git log --oneline -1)"
log ""
log "=== Final file sizes ==="
log "System prompt: $(python3 -c 'from src.llm.system_prompt import SYSTEM_PROMPT_TEMPLATE; print(len(SYSTEM_PROMPT_TEMPLATE))' 2>/dev/null) chars"
log "Architecture: $(wc -c < data/context/architecture.md) bytes"
log ""
log "Review the branch, then merge: git checkout master && git merge $BRANCH"
