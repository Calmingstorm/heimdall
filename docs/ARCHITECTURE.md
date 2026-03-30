# Heimdall Architecture Reference

Technical architecture document for developers working on or integrating with Heimdall.

## Core Design

Single execution path: every Discord message goes to Codex (ChatGPT) with full tool
access. No classifier, no routing, no approval prompts. Codex decides what to do.

```
Discord message → _process_with_tools()
  → Build system prompt (personality + capabilities + context)
  → Inject session history + context separator
  → Codex API (streaming, tool_choice="auto")
  → Tool loop (up to 20 iterations)
      → Execute tool calls concurrently (asyncio.gather)
      → Truncate + scrub output → feed back to Codex
  → Final text response → Discord
```

## Execution Tiers

**Tier 1 — Direct tools (Codex)**: 61 built-in tools for system commands, file ops,
knowledge base, scheduling, Discord operations. Fast, single-step actions.

**Tier 2 — Deep reasoning (Claude Code CLI)**: Complex multi-step tasks delegated to
`claude -p` via the `claude_code` tool. Code generation, repo analysis, debugging.

**Tier 3 — Autonomous loops**: LLM-driven recurring tasks. Each iteration gets Codex
with full tool access. Modes: act (do + report), notify (check + report), silent
(report only if notable, using [NOTIFY]/[ALERT] markers).

**Tier 4 — Agents**: Autonomous sub-agents spawned by loops or orchestrator. Each agent
gets its own tool loop and goal. Max 5 per channel, 30 iterations, 1-hour lifetime.

## Tool Dispatch

```
Tool handler → _exec_command(address, cmd, timeout, user)
    ├── is_local_address(address)? → run_local_command() [subprocess]
    └── remote?                    → run_ssh_command()   [asyncssh]
```

- 61 tools always available (no pack system — all tools active)
- Discord-native tools handled directly in client.py
- Executor tools dispatched via _exec_command in executor.py
- User-created skills loaded from data/skills/*.py at runtime

## Session Management

```
Message flow:
  history (last N messages) + "---CONTEXT ABOVE IS HISTORY---" + new message
  → Codex processes with full context
  → Only tool-bearing responses saved to history (anti-poisoning)
  → Compaction at >40 messages (summarize, omit errors, preserve outcomes)
  → Compaction triggers learning reflection
```

Session defense (5 layers): context separator, selective saving, abbreviated task history,
compaction error omission, fabrication/hedging/premature-failure detection.

## Knowledge Pipeline

```
ingest_document(content, source)
  → chunk text (overlapping windows)
  → fastembed (BAAI/bge-small-en-v1.5, 384-dim, ONNX, CPU)
  → Store: sqlite-vec (vectors) + FTS5 (keywords)

search_knowledge(query)
  → Embed query → cosine similarity (sqlite-vec)
  → FTS5 keyword search
  → Reciprocal Rank Fusion merge
  → Return top-k results
```

Falls back to FTS5-only if embeddings unavailable. No external servers needed.

## Agent System

```
AgentManager
  → spawn(channel_id, goal, label, tools, max_iterations)
  → Each agent gets: own tool loop, iteration counter, message history
  → Agents cannot spawn sub-agents (AGENT_BLOCKED_TOOLS)
  → Results collected via wait_for_agents() or get_results()

LoopAgentBridge
  → Autonomous loop iteration → spawn_agents_for_loop()
  → Wait for agents → collect results → feed back to loop
  → Max 3 agents per iteration, 10 per loop lifetime
```

Limits: 5 concurrent per channel, 30 iterations, 1-hour lifetime.

## Web UI

```
http://host:3939/ui/  → Vue 3 SPA (CDN, no build step)
/api/*                → REST API (55 endpoints, Bearer token auth)
/api/ws               → WebSocket (live logs + events + chat)
```

13 pages: dashboard, sessions, tools, skills, knowledge, schedules, loops,
processes, audit, config, logs, memory, chat.

Rate limited: 120 req/60s per IP. Security headers. CSRF protection.

## Background Services

| Service | Purpose | Backend |
|---------|---------|---------|
| Session compaction | Summarize long conversations | Codex chat |
| Learning reflection | Extract lessons from conversations | Codex chat |
| Digest summarization | Create periodic activity summaries | Codex chat |
| Autonomous loops | Recurring LLM-driven tasks | Codex + tools |
| Monitoring watcher | Proactive infrastructure checks | Direct tools |
| Stale cache cleanup | Prune _recent_actions, _channel_locks | Timer (5 min) |

## Caching Strategy

- Tool definitions: cached per message, invalidated on skill CRUD and /reload
- System prompt components: host dict, skills text, user memory (60s TTL), reflector prompt (60s TTL)
- Tool memory hints: 30s TTL, evicts stale entries at >100
- Connection pooling: TCPConnector with keepalive=30s, limit=10
- ZoneInfo cache: timezone objects cached to avoid repeated construction
- Pre-compiled regex: hot-path patterns compiled once at module level
- Tool name→handler mapping: dict lookup instead of linear scan

## Security Model

Secret scrubbing (10 patterns) at 9+ locations. Input validation on tool parameters.
Context separator prevents prompt injection. Role architecture prevents role forgery.
Skill sandbox: resource limits, blocked paths, blocked URLs, safe tool allowlist.
Web API: Bearer token auth, rate limiting, security headers, CSRF protection.

## Data Storage

| Store | Technology | Location |
|-------|-----------|----------|
| Sessions | In-memory + optional disk persist | data/sessions/ |
| Knowledge | SQLite + sqlite-vec + FTS5 | data/search.db |
| Audit log | Append-only JSONL | data/audit/ |
| Memory | SQLite key-value | data/memory.db |
| Skills | Python files | data/skills/ |
| Schedules | In-memory (recreated on start) | — |
| Tool memory | In-memory | — |

## Source Layout

```
src/
├── agents/           # Multi-agent orchestration
│   ├── manager.py    # AgentManager (spawn, manage, kill)
│   └── loop_bridge.py # LoopAgentBridge (loop↔agent integration)
├── audit/            # Append-only JSONL audit logging
├── config/           # Pydantic config models
├── discord/          # Discord bot client (main integration point)
│   ├── client.py     # on_message, tool loop, response delivery
│   ├── background_task.py  # Background task delegation
│   └── voice.py      # Voice channel integration (STT/TTS)
├── health/           # Webhook receiver, health check, web UI serving
├── knowledge/        # SQLite+sqlite-vec RAG knowledge base
├── learning/         # Conversation reflection (lesson extraction)
├── llm/              # LLM client, secret scrubbing, circuit breaker
├── monitoring/       # Proactive infrastructure monitoring
├── scheduler/        # Cron and one-time task scheduler
├── search/           # FTS5, sqlite-vec, hybrid search (RRF)
├── sessions/         # Conversation history with compaction
├── tools/            # 61 tool definitions, executor, skills, browser
│   ├── registry.py   # Tool definitions
│   ├── executor.py   # Tool execution dispatch
│   ├── skill_manager.py  # Runtime skill loading
│   └── skill_context.py  # Skill API surface
└── web/              # REST API, WebSocket, chat backend
    ├── api.py        # 55 REST endpoints
    ├── websocket.py  # WebSocket handler
    └── chat.py       # Chat message processing
```
