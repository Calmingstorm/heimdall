# Heimdall Overhaul Plan — Post Prompt Optimization

## Context

Three major workstreams identified from comprehensive OpenClaw comparison:

1. **Tool Packs** — Make infrastructure tools optional, reduce default token cost
2. **Embedding Overhaul** — Self-contained by default, no hard Ollama dependency
3. **Feature Gaps** — Address real capability gaps vs OpenClaw

---

## Workstream 1: Tool Packs (2 sessions)

### Problem
Every Codex API call sends 76+ tool definitions (~15-20K tokens). Infrastructure
tools (systemd, Docker, Incus, Ansible, Prometheus) assume a specific Linux stack
and are useless for most public bot deployments. They constrain the LLM by
channeling it toward specialized tools instead of creative problem-solving via
generic execution.

### Design
- **Core tools** (always loaded): run_command, run_script, run_command_multi,
  read_file, write_file, post_file, generate_file, browser_*, web_*, knowledge_*,
  schedule_*, skill_*, claude_code, memory_manage, manage_list, delegate_task,
  list_tasks, cancel_task, search_history, search_audit, set_permission,
  parse_time, purge_messages, broadcast + any new tools from WS3
- **Tool packs** (opt-in via config.yml):
  - `docker`: check_docker, docker_logs, docker_compose_*, docker_stats
  - `systemd`: check_service, restart_service, check_logs
  - `incus`: all 11 incus_* tools
  - `ansible`: run_ansible_playbook
  - `prometheus`: query_prometheus, query_prometheus_range, check_disk, check_memory
  - `git`: all 8 git_* tools
  - `comfyui`: generate_image (from WS3e)

### Config
```yaml
tools:
  # Empty or absent = ALL tools loaded (backward compatible)
  # Specify packs to load only core + selected infrastructure tools
  tool_packs: [docker, git, prometheus]
```

### Files
- `src/config/schema.py` — add `tool_packs: list[str]` field
- `src/tools/registry.py` — add `TOOL_PACKS` dict, modify `get_tool_definitions()`
- `src/discord/client.py` — pass packs to `get_tool_definitions()`
- `config.yml` — add commented example
- `tests/test_tool_packs.py` — new test file

### Key Rule
- Empty/absent `tool_packs` = ALL tools (backward compatible, existing deployments unaffected)
- Skills bypass pack filtering (always available)

---

## Workstream 2: Embedding Overhaul (6-8 sessions)

### Problem
Heimdall requires Ollama (external server with nomic-embed-text model) for ALL
embedding operations. If Ollama is down or unavailable:
- Knowledge base ingestion FAILS completely (BUG: FTS5 not written either)
- Semantic search returns nothing
- Tool memory can't record patterns
- Session archive indexing fails

OpenClaw uses SQLite + sqlite-vec (zero external deps) with local GGUF models
and graceful FTS-only fallback. Heimdall should be fully self-contained too.

### Decision: Remove Ollama Entirely
No external embedding server. All embeddings happen in-process using `fastembed`
(ONNX-based, CPU, auto-downloads model on first use).

**New stack**:
- `fastembed` — in-process ONNX embeddings (BAAI/bge-small-en-v1.5, 67MB model, 384-dim)
- `sqlite-vec` — vector storage + cosine similarity in SQLite
- `FTS5` — keyword search fallback (already implemented)

**Dependency swap**:
- Remove: `chromadb` (brings onnxruntime, tokenizers, etc.)
- Add: `fastembed` (brings onnxruntime, tokenizers — net neutral)
- Add: `sqlite-vec` (tiny)
- Remove: ALL Ollama references (embedder.py, config, client.py)

**Performance**: ~16ms per embed on CPU, 384 dimensions. Model auto-downloads
from HuggingFace on first boot (~67MB). No GPU needed. No server needed.

**Why fastembed over alternatives**:
- sentence-transformers: DISQUALIFIED — pulls in 4.8GB PyTorch
- ChromaDB built-in: Requires keeping ChromaDB as a dependency
- onnxruntime manual: Same thing as fastembed but more work
- model2vec: 56x faster but much worse quality (0.52 vs 0.82 similarity)

### Phase 2a: Fix FTS Bug + Local Embedder (2 sessions)

Session 1: Fix FTS bug
- **Bug**: `knowledge/store.py` line 104 — when `embed()` returns None, `continue`
  skips both ChromaDB upsert AND FTS5 write. FTS has no dependency on embeddings.
- **Fix**: Decouple FTS write from embedding success. Always write to FTS5.
- Files: `src/knowledge/store.py`, `src/search/vectorstore.py`

Session 2: Replace OllamaEmbedder with fastembed
- Delete `src/search/embedder.py` (OllamaEmbedder class)
- New `src/search/embedder.py` — `LocalEmbedder` class wrapping fastembed
  - `__init__`: lazy-load fastembed TextEmbedding model
  - `embed(text) -> list[float]`: synchronous embed, called from async via executor
  - Model: `BAAI/bge-small-en-v1.5` (384-dim, 67MB, auto-downloads)
  - Graceful fallback: if fastembed fails to load, `embed()` returns None → FTS-only
- Remove ALL Ollama config: `ollama_url`, `embed_model` from SearchConfig
- Remove Ollama health checks, connection pooling, etc.
- Update all type hints from `OllamaEmbedder` to `LocalEmbedder`
- Files: `src/search/embedder.py`, `src/config/schema.py`, `src/discord/client.py`,
  all TYPE_CHECKING imports

### Phase 2b: Replace ChromaDB with sqlite-vec (3-4 sessions)
- Replace `chromadb.PersistentClient` with SQLite + sqlite-vec in both stores
- Single `.sqlite` file instead of ChromaDB directory
- Auto-migrate existing ChromaDB data on first boot (read old, write new, rename old dir)
- FTS5 tables in same database (currently separate fts.db)
- Remove `chromadb` pip dependency, add `sqlite-vec` and `fastembed`
- New `src/search/sqlite_vec.py` — helpers for extension loading, vector serialization,
  cosine search queries

Files: `src/knowledge/store.py`, `src/search/vectorstore.py`, new `src/search/sqlite_vec.py`,
`pyproject.toml`

### Phase 2c: FTS-Only Fallback Mode (1-2 sessions)
- Knowledge store works without embedder (keyword search only)
- Session search works without embedder
- Bot initializes knowledge/vector stores even when embedder=None
- Tool memory already has Jaccard fallback (no change needed)
- This is the safety net: if fastembed model download fails (air-gapped, disk full, etc.),
  Heimdall still has keyword search over everything

---

## Workstream 3: Feature Gaps (7-12 sessions)

### 3a: PDF Analysis (1-2 sessions)
- New tool: `analyze_pdf` — extract text from PDF (URL or host path)
- Use PyMuPDF (fitz) for extraction
- Handle PDF Discord attachments in `_process_attachments()`
- New dep: `PyMuPDF>=1.24.0`

### 3b: Rich Discord Messaging (1-2 sessions)
- New tool: `add_reaction` — add emoji reaction to a message by ID
- New tool: `create_poll` — Discord native polls
- Enhance `broadcast` to support embed fields/colors/thumbnails

### 3c: Interactive Process Management (2-3 sessions)
- New tool: `manage_process` — start/poll/write/kill/list background processes
- New file: `src/tools/process_manager.py` — ProcessRegistry class
- Ring buffer for output, auto-cleanup dead processes
- Max concurrent processes, max lifetime limits

### 3d: Proactive Image Analysis (1-2 sessions)
- New tool: `analyze_image` — fetch image from URL or host, send to Codex as vision
- Reuses existing Discord attachment → vision pipeline
- Lets LLM proactively decide to look at images

### 3e: Image Generation (2-3 sessions)
- New tool: `generate_image` — text-to-image via ComfyUI API
- New file: `src/tools/comfyui.py` — ComfyUI HTTP client
- Config: `comfyui.enabled`, `comfyui.url`, `comfyui.default_model`
- Goes in `comfyui` tool pack (opt-in)
- Graceful error if ComfyUI not available

---

## Execution Order

```
Phase 1 (foundation):
  WS1: Tool Packs ........................ 2 sessions
  WS2a: FTS Bug + Local Embedder ......... 2 sessions

Phase 2 (storage migration):
  WS2b: sqlite-vec Migration ............. 3-4 sessions
  WS2c: FTS-Only Fallback ................ 1-2 sessions

Phase 3 (features, parallelizable):
  WS3a: PDF Analysis ..................... 1-2 sessions
  WS3b: Rich Discord Messaging ........... 1-2 sessions
  WS3c: Process Management ............... 2-3 sessions
  WS3d: Proactive Image Analysis ......... 1-2 sessions
  WS3e: Image Generation ................. 2-3 sessions

Total: 15-20 sessions
```

---

## Testing Rules
- Full `python3 -m pytest tests/ -q` before AND after every session
- Baseline: 4209+ tests must not decrease
- New tools must pass `test_every_registry_tool_has_handler_or_client_handler`
- All external services mocked in tests
- Backward compatibility: old config (no new fields) must work unchanged
