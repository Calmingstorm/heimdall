#!/usr/bin/env bash
set -euo pipefail

SESSIONS=4
START_ROUND="${1:-1}"
WORKDIR="/home/calmingstorm/Desktop/heimdall-build"
LOG_FILE="$WORKDIR/channel_log_loop.txt"
BRANCH="feature/channel-logger"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

trap 'log "=== Channel logger loop terminated ==="' EXIT

# --- Preflight ---
log "=== Preflight ==="
command -v claude >/dev/null 2>&1 || { log "FATAL: claude CLI not found"; exit 1; }
command -v python3 >/dev/null 2>&1 || { log "FATAL: python3 not found"; exit 1; }
python3 -m pytest --version >/dev/null 2>&1 || { log "FATAL: pytest not found"; exit 1; }
[[ -f "$WORKDIR/BUILD_STATUS.md" ]] || { log "FATAL: BUILD_STATUS.md missing"; exit 1; }
[[ -f "$WORKDIR/src/discord/client.py" ]] || { log "FATAL: client.py missing"; exit 1; }
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
You are round ROUND_NUM of 4 in a feature implementation loop.

Read BUILD_STATUS.md — find "Round ROUND_NUM" and do EXACTLY what it says.

CRITICAL RULES:
- Rounds 1-3: Implement code. Run tests after changes. Fix failures your changes cause.
- Round 4: Tests ONLY. Do NOT modify src/ code. Fix tests + write new tests.
- Do NOT modify system_prompt.py or architecture.md
- The passive listener must use ZERO LLM tokens — pure file I/O only
- Write path must be fast: synchronous file append, no DB operations per message
- FTS indexing must be batched/periodic, NOT per-message
- Commit: git add -A && git commit -m "[Round ROUND_NUM]: description"

If this is Round 2+, read the git diff from the previous round first:
  git diff HEAD~1

KEY FILES:
- src/discord/client.py — on_message handler, health timer, search_history handler
- src/discord/channel_logger.py — NEW file (create in Round 1)
- src/sessions/manager.py — search_history method
- src/search/fts.py — FTS5 index tables and methods
- src/tools/registry.py — search_history tool description
PROMPTEOF

log "=== Channel Logger Loop Starting (rounds $START_ROUND-$SESSIONS) ==="
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

    # Validate tests pass
    TEST_OUTPUT=$(cd "$WORKDIR" && python3 -m pytest tests/ -x -q --tb=no 2>&1 | tail -3 || true)
    if echo "$TEST_OUTPUT" | grep -q "passed"; then
        log "TESTS: $TEST_OUTPUT"
    else
        log "TESTS FAILING: $TEST_OUTPUT"
        VALID=false
    fi

    # Validate channel_logger.py exists (after round 1)
    if [[ $i -ge 2 ]] && [[ ! -f "$WORKDIR/src/discord/channel_logger.py" ]]; then
        log "VALIDATION FAIL: channel_logger.py not created"
        VALID=false
    fi

    # Validate no LLM imports in channel_logger (ignore comments)
    if [[ -f "$WORKDIR/src/discord/channel_logger.py" ]]; then
        if grep -P "^(from|import).*\b(codex|openai|anthropic|chat_with_tools)\b" "$WORKDIR/src/discord/channel_logger.py" 2>/dev/null; then
            log "VALIDATION FAIL: channel_logger.py imports LLM modules"
            VALID=false
        fi
    fi

    if [[ "$VALID" == "true" ]]; then
        log "Round $i PASSED ($CLAUDE_STATUS) in ${ELAPSED}s"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        log "Round $i FAILED VALIDATION in ${ELAPSED}s"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi

    [[ $i -lt $SESSIONS ]] && { log "Pausing 10s..."; sleep 10; }
done

log "=== Channel Logger Loop Finished ==="
log "Results: $PASS_COUNT passed, $FAIL_COUNT failed out of $SESSIONS rounds"
log "Branch: $BRANCH"
log "Latest commit: $(git log --oneline -1)"
log "Test results: $(cd "$WORKDIR" && python3 -m pytest tests/ -q --tb=no 2>&1 | tail -1 || true)"
log ""
log "Review the branch, then merge: git checkout master && git merge $BRANCH"
