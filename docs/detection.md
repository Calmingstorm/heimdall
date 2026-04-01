# Detection Systems

Heimdall uses a three-tier detection system to catch bad LLM behavior before responses reach Discord.

## Tier 1 — First Response (Regex)

These fire on the first LLM response before any tools have been called. They catch structural problems where the model responds with text instead of calling tools.

| Detector | Catches | Example |
|----------|---------|---------|
| **Fabrication** | Claims tool results without calling tools | "I checked the disk and it's at 45%" (no tool called) |
| **Promise without action** | Says "I'll do X" without calling tools | "I'll check the disk space now." (no tool called) |
| **Tool unavailability** | Claims a tool is disabled without trying | "Image generation is not enabled" (tool exists) |
| **Hedging** | Asks permission instead of executing | "Would you like me to check?" / "Shall I proceed?" |
| **Code-block hedging** | Shows a bash command instead of running it | `` ```bash\ndf -h\n``` `` (should call `run_command`) |

Each fires once per request. On detection, a targeted retry message is injected and the model tries again.

## Tier 2 — Post-Tool (Regex)

Fires after tools have been called, catches premature giving up.

| Detector | Catches | Example |
|----------|---------|---------|
| **Premature failure** | Gives up after one error without trying alternatives | "Connection refused. The host is down." (tried one approach) |

Fires once. Retry message: "Try alternative approaches before reporting failure."

## Tier 3 — Completion Classifier (LLM)

Fires at the exit point, right before the response would be sent to Discord. Uses a lightweight GPT-5.4 call to judge whether the user's full request was actually addressed.

### What It Catches

Unlike regex, the classifier understands semantics:

- **Mid-task bailout**: "I'm finishing this by syncing the files..." → INCOMPLETE
- **Partial completion**: "Done! Built the app." (user asked to build AND deploy) → INCOMPLETE
- **Promise of future work**: "Next I'll verify the endpoints." → INCOMPLETE

### What It Doesn't Catch (By Design)

- **Genuine failure reports**: "Tried 3 approaches, all failed. The API is down." → COMPLETE
- **Chat responses**: No tools used → classifier doesn't fire

### How It Works

1. Model responds with text (no tool calls) after having used tools
2. Classifier receives: original user task + tool names called + response text
3. Returns `COMPLETE` or `INCOMPLETE: reason`
4. If INCOMPLETE: injects targeted continuation — "You are not done. {reason}. Continue with tool calls now."
5. Up to 3 continuations per request
6. Fail-open: any error/timeout → treat as COMPLETE (don't block responses)

### Why Not Regex?

The old regex system (`_CHECKPOINT_PATTERNS`, `_PROMISE_PATTERNS`) had fundamental problems:

- Novel phrasings slipped through constantly
- Couldn't catch partial completion (model genuinely thinks it's done)
- Two overlapping pattern lists trying to answer the same question
- Chat exemptions grew endlessly to prevent false positives
- Confirmed bugs where matching patterns were still sent as final responses
