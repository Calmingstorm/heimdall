# Heimdall

**Autonomous executor Discord bot powered by GPT-5.4, with 61 tools and the burden of seeing everything.**

Heimdall manages servers, containers, services, and code through natural language in Discord. Every message goes to GPT-5.4 (via the ChatGPT Codex API) with full tool access. No classifier, no approval prompts, no hesitation.

## Key Features

- **61 built-in tools** — SSH, browser automation, scheduling, knowledge base, autonomous loops, skills, agents
- **No per-token billing** — uses the ChatGPT subscription Codex API, not metered API keys
- **Completion classifier** — LLM judge catches mid-task bailouts and partial completions before they reach Discord
- **Three-tier detection** — fabrication, hedging, premature failure, and semantic completion analysis
- **Extensible skills** — create custom tools at runtime via Discord
- **RAG knowledge base** — local embeddings + sqlite-vec + FTS5 hybrid search
- **Multi-agent orchestration** — parallel autonomous agents with lifecycle management
- **Web management UI** — dashboard, chat, sessions, tools, skills, knowledge, logs, and more
- **9000+ tests** — comprehensive test suite, all mocked

## Personality

Heimdall is "The All-Seeing Guardian" — in Norse mythology, he watches everything across all nine realms and is profoundly tired of it. Eternally vigilant, deeply competent, can hear the servers breathing. Professional about it. Not okay.

## Quick Links

- [Getting Started](getting-started.md) — install via .deb, Docker, or bare metal
- [Packaging](packaging.md) — building .deb packages, Docker images, and releases
- [Configuration](configuration.md) — all config sections and environment variables
- [Tools](tools.md) — all 61 built-in tools
- [Skills](skills.md) — create custom tools
- [REST API](api.md) — 55 API endpoints
- [Architecture](architecture.md) — how it all works
