#!/usr/bin/env bash
set -euo pipefail

SESSIONS=6
START_ROUND="${1:-1}"
WORKDIR="/home/calmingstorm/Desktop/heimdall-build"
LOG_FILE="$WORKDIR/context_fix_log.txt"
BRANCH="fix/context-retention"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

trap 'log "=== Context fix loop terminated ==="' EXIT

# --- Preflight ---
log "=== Preflight ==="
command -v claude >/dev/null 2>&1 || { log "FATAL: claude CLI not found"; exit 1; }
command -v python3 >/dev/null 2>&1 || { log "FATAL: python3 not found"; exit 1; }
python3 -m pytest --version >/dev/null 2>&1 || { log "FATAL: pytest not found"; exit 1; }
[[ -f "$WORKDIR/BUILD_STATUS.md" ]] || { log "FATAL: BUILD_STATUS.md missing"; exit 1; }
[[ -f "$WORKDIR/src/sessions/manager.py" ]] || { log "FATAL: sessions/manager.py missing"; exit 1; }
[[ -f "$WORKDIR/src/discord/client.py" ]] || { log "FATAL: discord/client.py missing"; exit 1; }
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
You are round ROUND_NUM of 5 in a context retention fix loop.

Read BUILD_STATUS.md — find "Round ROUND_NUM" and do EXACTLY what it says.

CRITICAL RULES:
- Primary files: src/discord/client.py and src/sessions/manager.py
- Rounds 1-5: Do NOT run tests. Do NOT modify test files. Code changes only.
- Round 6: Tests ONLY. Do NOT modify src/ code. Fix tests to match new code + write new tests.
- Do NOT modify system_prompt.py or architecture.md
- Commit: git add -A && git commit -m "[Round ROUND_NUM]: description"

If this is Round 2+, read the git diff from the previous round first:
  git diff HEAD~1

This shows you what the last round changed, so you can review before making your own changes.

KEY CONTEXT:
- OpenClaw (reference at /tmp/openclaw/) saves ALL messages to history and only trims when transmitting to the LLM. Heimdall currently drops text-only responses at storage time, which is irreversible.
- The goal is to make Heimdall remember what he said, even in pure-chat responses.
- Do not over-engineer. Make the minimum change needed for each problem.
PROMPTEOF

log "=== Context Retention Fix Loop Starting (rounds $START_ROUND-$SESSIONS) ==="
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

    # Validate tests pass (only on round 6)
    if [[ $i -eq $SESSIONS ]]; then
        TEST_OUTPUT=$(cd "$WORKDIR" && python3 -m pytest tests/ -x -q --tb=no 2>&1 | tail -3 || true)
        if echo "$TEST_OUTPUT" | grep -q "passed"; then
            log "TESTS: $TEST_OUTPUT"
        else
            log "TESTS FAILING: $TEST_OUTPUT"
            VALID=false
        fi
    else
        log "TESTS: skipped (code round)"
    fi

    # Validate detection functions still present
    for fn in detect_fabrication detect_hedging detect_premature_failure; do
        if ! grep -q "$fn" "$WORKDIR/src/discord/client.py" 2>/dev/null; then
            log "VALIDATION FAIL: $fn was removed from client.py"
            VALID=false
        fi
    done

    # Validate the key fix is in place (text-only responses saved)
    if grep -q "text-only replies pollute history" "$WORKDIR/src/discord/client.py" 2>/dev/null; then
        log "VALIDATION WARN: old 'text-only pollute' comment still present — Problem 1 may not be fixed"
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

log "=== Context Fix Loop Finished ==="
log "Results: $PASS_COUNT passed, $FAIL_COUNT failed out of $SESSIONS rounds"
log "Branch: $BRANCH"
log "Latest commit: $(git log --oneline -1)"
log "Test results: $(cd "$WORKDIR" && python3 -m pytest tests/ -q --tb=no 2>&1 | tail -1 || true)"
log ""
log "Review the branch, then merge: git checkout master && git merge $BRANCH"
