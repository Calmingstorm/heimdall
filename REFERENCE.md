# OpenClaw / ACPX Architecture Reference
# Pre-analyzed — DO NOT re-fetch repos unless this file lacks a specific answer.
# If you fetch new information, APPEND it here with a dated section header.

## OpenClaw Core Architecture (https://github.com/openclaw/openclaw)

### Routing: No Classifier
- 7-tier hierarchical scope matching (peer → parent → guild+role → guild → team → account → channel)
- Pure configuration-driven, no ML inference
- WeakMap caching for performance
- Session keys: `agentId:channelId:accountId:peerId[:sessionName]`
- **Takeaway for Loki**: Remove classifier entirely. One path: Codex with tools, always.

### Tool Execution: Three Tiers
- **Sandbox**: Isolated container, restricted PATH, allowlist commands
- **Gateway**: Full local access (single-user assumption), env sanitization
- **Node**: Remote service, structured request/response, security levels
- NO SSH in codebase — local execution is direct subprocess
- Pre-flight script validation: `validateScriptFileForShellBleed()` catches `$VARIABLE` injection
- **Takeaway for Loki**: Use direct subprocess for localhost, keep SSH for remote only.

### No Approval System
- Tools are capabilities, not suggestions
- If tool is available in session, it's implicitly executable
- Constraints are explicit in session config, not soft-requested in prompts
- **Takeaway for Loki**: Delete approval.py entirely. Remove requires_approval from all tools.

### System Prompt Strategy
- Bootstrap files injected at runtime (not hardcoded monolith)
- Budget analysis tracks context file injection limits
- Configuration snapshot embedded in prompt
- High assertiveness: no "I'll try" or "I can attempt"
- Tools presented as available capabilities, not optional features
- **Takeaway for Loki**: System prompt should declare capabilities assertively.
  "You have run_command. You have write_file. You have run_script." Not "you can use".

### Background Process Management
- First-class tool with full lifecycle: list, poll, log, kill, write-input
- Configurable yield windows (~10s default)
- Commands can yield after timeout or execute fully backgrounded
- **Takeaway for Loki**: Background tasks already exist, but consider yield pattern.

### Session/History Management
- Persistent sessions stored in `~/.acpx/`
- Turn history with transcript persistence
- Queue-based prompt serialization (preserves session state)
- 5-second heartbeat maintains process lease
- Recency-focused context — responds to latest message, earlier = history
- **Takeaway for Loki**: Already implemented (session manager + compaction). Keep as-is.

### Message Handling
- Bot messages treated as valid context (no special filtering)
- Thread binding for parallel conversations in same channel
- Provenance metadata separates system data from user text
- Content normalization: arrays → strings safely
- **Takeaway for Loki**: Bot buffer already handles this. Keep combine_bot_messages.

### Anti-Hesitation Patterns
- VISION.md: "OpenClaw is the AI that actually does things."
- No prompting language in tool descriptions
- Scope-driven execution: available = executable
- Direct action language in all system instructions
- **Takeaway for Loki**: Already have detect_hedging + retry. Strengthen system prompt.

## ACPX Extension (https://github.com/openclaw/acpx)

### Session Persistence
- Persistent multi-turn sessions scoped to repositories
- Parallel named sessions in same directory
- Directory-walk session discovery (CWD → git root)
- Cooperative cancellation via ACP session/cancel

### CLI Patterns
- Lazy module loading (heavy modules on-demand)
- Multi-format output (JSON, text, quiet)
- Distinct exit codes: denied vs interrupted vs error
- Global + project-level config with CLI flag precedence

### Queue Management
- SessionQueueOwner handles persistent sessions
- 5-second heartbeat for lease validity
- Multiple prompts queued to single agent process
- Configurable queue depth limits

## Key Differences: OpenClaw vs Loki

| Aspect | OpenClaw | Loki (current) | Loki (target) |
|--------|----------|----------------|---------------|
| Routing | Scope matching | Haiku classifier | None — all → Codex |
| Approval | None | Button-based | None |
| Local exec | Direct subprocess | SSH to localhost | Direct subprocess |
| Hesitation | Never | detect_hedging retry | Never + retry |
| Fabrication | N/A | detect_fabrication retry | Keep retry |
| Personality | Neutral | Generic assistant | Existentially distressed |
| Tools | Capabilities | Suggestions with approval | Capabilities |
| Classifier cost | $0 | ~$0.0001/msg (Haiku) | $0 |

## Loki Target Architecture: Two-Tier Execution

```
Every Discord message
  → Codex (with ALL tools + personality in system prompt)
      ├── CHAT: Codex responds directly with personality (no tools)
      ├── SIMPLE TASK: Codex calls tools directly (run_command, check_disk, web_search, etc.)
      │   Fast, no overhead, handles ~80% of requests
      ├── COMPLEX TASK: Codex delegates to claude -p via claude_code tool
      │   Code generation, multi-step builds, repo analysis, debugging
      │   claude -p runs entire chain in one session (no context loss)
      │   Results return to Codex → Codex delivers to Discord
      └── DISCORD OPS: Always Codex (post_file, browser_screenshot, generate_file, embeds)
          claude -p can't interact with Discord — Codex bridges the gap
```

### Execution Chain Example (complex task):
1. User: "Clone this repo, build it, run tests, post the output"
2. Codex sees the message, recognizes multi-step complexity
3. Codex calls `claude_code` with the full task description + allow_edits=true
4. claude -p clones, builds, runs tests — all in one session, no SSH round-trips between steps
5. claude -p returns: text output + file manifest (FILES ON DISK: ...)
6. Codex reads the results, calls `post_file` to attach output to Discord
7. Codex responds with personality-infused summary

### When Codex should delegate to claude -p (system prompt guidance):
- Multi-file code generation or modification
- Reading and following documentation/instructions from repos
- Complex debugging that requires reading code + running tests iteratively
- Building and deploying entire projects
- Any task that would take 3+ tool calls to accomplish step-by-step

### When Codex should handle directly:
- Single commands (run_command, check_disk, check_memory)
- Web searches, URL fetches
- File reads/writes (single file)
- Scheduling, reminders, list management
- Chat, questions, conversation
- Discord operations (screenshots, file posting, embeds)

### Cost: $0 for everything
- Codex: free via ChatGPT subscription
- claude -p: free via Claude Max subscription
- No Haiku classifier: removed
- No Anthropic API key required at all

### Personality flow:
- System prompt gives Codex the existential crisis personality
- claude -p is a neutral worker — no personality injection
- When claude -p returns results, Codex wraps them in personality before posting
- Personality NEVER interferes with tool execution accuracy

---

## Python Library Reference (2026-03-24)
Tested and verified on real installs. Use these APIs exactly as shown.

### 1. fastembed (v0.8.0) — Local Text Embeddings

**Install**: `pip install fastembed`
**Models download to**: `/tmp/fastembed_cache/` by default, or pass `cache_dir=` to constructor.
**ONNX Runtime**: Uses onnxruntime, no GPU required (CPU inference).

```python
from fastembed import TextEmbedding

# Initialize — auto-downloads model on first use (~50MB for bge-small)
model = TextEmbedding("BAAI/bge-small-en-v1.5")  # 384-dim
# Optional: model = TextEmbedding("BAAI/bge-small-en-v1.5", cache_dir="/path/to/cache")

# embed() returns a GENERATOR — must call list() to materialize
embeddings = list(model.embed(["your text here"]))
# embeddings[0] is numpy.ndarray, shape=(384,), dtype=float32

# Multiple texts
texts = ["hello world", "how are you", "vector search"]
embeddings = list(model.embed(texts))
# len(embeddings) == 3, each is ndarray shape=(384,)

# Convert to bytes for sqlite-vec
vec_bytes = embeddings[0].tobytes()  # 1536 bytes (384 * 4 bytes per float32)
```

**Gotchas**:
- `embed()` returns a **generator**, not a list. Always wrap in `list()`.
- Passing a bare string `embed("hello")` works (treats as single doc), but always use a list for clarity.
- Sync only. No async variant.
- First call downloads model (~50MB). Subsequent calls use cache.
- `query_embed()` and `passage_embed()` exist for asymmetric search (query vs document).

**Available models** (30 total, common ones):
| Model | Dimensions |
|-------|-----------|
| BAAI/bge-small-en-v1.5 | 384 |
| BAAI/bge-base-en-v1.5 | 768 |
| BAAI/bge-large-en-v1.5 | 1024 |

**Other useful methods**:
- `TextEmbedding.list_supported_models()` — returns list of dicts with model info
- `model.get_embedding_size()` — returns int (e.g., 384)

---

### 2. sqlite-vec (v0.1.7) — Vector Search in SQLite

**Install**: `pip install sqlite-vec`
**No external dependencies**. Pure extension loaded into Python's sqlite3.

```python
import sqlite3
import sqlite_vec
import struct
import numpy as np

db = sqlite3.connect(":memory:")  # or "file.db"
db.enable_load_extension(True)
sqlite_vec.load(db)
db.enable_load_extension(False)

# Check version
db.execute("SELECT vec_version()").fetchone()[0]  # "v0.1.7"
```

**Option A: vec0 Virtual Table (indexed KNN — fast for large datasets)**:
```python
# Create with L2 distance (default)
db.execute("""
    CREATE VIRTUAL TABLE vec_items USING vec0(
        item_id INTEGER PRIMARY KEY,
        embedding float[384]
    )
""")

# Create with cosine distance
db.execute("""
    CREATE VIRTUAL TABLE vec_items USING vec0(
        item_id INTEGER PRIMARY KEY,
        embedding float[384] distance_metric=cosine
    )
""")

# Insert — vectors must be bytes (float32 little-endian)
vec = np.random.rand(384).astype(np.float32)
db.execute(
    "INSERT INTO vec_items(item_id, embedding) VALUES (?, ?)",
    [1, vec.tobytes()]
)

# Also accepts JSON string
db.execute(
    "INSERT INTO vec_items(item_id, embedding) VALUES (?, ?)",
    [2, json.dumps([0.1] * 384)]
)

# KNN query — use MATCH + k parameter
query_vec = np.random.rand(384).astype(np.float32)
rows = db.execute("""
    SELECT item_id, distance
    FROM vec_items
    WHERE embedding MATCH ?
    AND k = 10
    ORDER BY distance
""", [query_vec.tobytes()]).fetchall()
# Returns list of (item_id, distance) tuples
```

**Option B: Regular Table + vec functions (brute force — fine for <10k rows)**:
```python
db.execute("CREATE TABLE documents (id INTEGER PRIMARY KEY, content TEXT, embedding BLOB)")

db.execute("INSERT INTO documents VALUES (?, ?, ?)",
           [1, "some text", vec.tobytes()])

# Brute-force cosine search on regular table
rows = db.execute("""
    SELECT id, content, vec_distance_cosine(embedding, ?) as dist
    FROM documents
    ORDER BY dist ASC
    LIMIT 10
""", [query_vec.tobytes()]).fetchall()
```

**Serialization helpers**:
```python
# numpy array to bytes
vec_bytes = vec.astype(np.float32).tobytes()

# Python list to bytes (no numpy)
vec_bytes = struct.pack(f'{len(vec_list)}f', *vec_list)

# bytes back to numpy
vec_back = np.frombuffer(vec_bytes, dtype=np.float32)
```

**Available SQL functions**:
| Function | Purpose |
|----------|---------|
| `vec_distance_cosine(a, b)` | Cosine distance between two vectors |
| `vec_distance_L2(a, b)` | L2 (Euclidean) distance |
| `vec_length(v)` | Number of dimensions |
| `vec_normalize(v)` | Unit-normalize a vector |
| `vec_to_json(v)` | Convert vector bytes to JSON array string |
| `vec_version()` | Extension version |

**Gotchas**:
- Vectors are **bytes** (not lists, not numpy). Always `.tobytes()` before insert/query.
- JSON string arrays also work for inserts (sqlite-vec converts internally).
- vec0 virtual table: use `WHERE embedding MATCH ? AND k = N` syntax for KNN.
- Cosine distance: 0.0 = identical, higher = more different. Not cosine similarity.
- `enable_load_extension(True)` required before `sqlite_vec.load(db)`.

---

### 3. PyMuPDF / fitz (v1.27.2) — PDF Text Extraction

**Install**: `pip install PyMuPDF`
**Import**: `import fitz` (NOT `import PyMuPDF`)

```python
import fitz

# Open from file path
doc = fitz.open("/path/to/file.pdf")

# Open from bytes (e.g., fetched from URL)
doc = fitz.open(stream=pdf_bytes, filetype="pdf")

# Page count
doc.page_count  # int
len(doc)        # same thing

# Extract text from all pages
full_text = "\n".join(page.get_text() for page in doc)

# Extract from specific page (0-indexed)
page = doc[2]       # third page
text = page.get_text()  # returns str

# Extract from page range (e.g., pages 5-10, 0-indexed)
for i in range(4, 10):
    text = doc[i].get_text()

# Metadata
doc.metadata  # dict: format, title, author, subject, creator, etc.

# Always close when done
doc.close()

# Or use context manager
with fitz.open("/path/to/file.pdf") as doc:
    text = "\n".join(page.get_text() for page in doc)
```

**get_text() modes**:
| Mode | Returns | Use case |
|------|---------|----------|
| `"text"` (default) | `str` | Plain text extraction |
| `"blocks"` | `list[tuple]` | Text blocks with position info |
| `"words"` | `list[tuple]` | Individual words with bbox |
| `"dict"` | `dict` | Full structured data (fonts, spans, etc.) |
| `"html"` | `str` | HTML representation |

**Gotchas**:
- Import is `fitz`, not `pymupdf` or `PyMuPDF`.
- `fitz.open()` does NOT fetch URLs. Fetch bytes yourself, then pass `stream=`.
- Pages are 0-indexed. `doc[0]` is the first page.
- `get_text()` returns `str` by default. No need to decode.
- Context manager (`with`) works and auto-closes.
- Works with encrypted PDFs if you pass `password=` to `fitz.open()`.

---

### 4. discord.py Poll API (v2.7.1) — Native Discord Polls

**Requires**: discord.py >= 2.4.0 (Poll class added in 2.4)

```python
import discord
from datetime import timedelta

# Create a poll
poll = discord.Poll(
    question="What's your favorite color?",
    duration=timedelta(hours=24),  # required, timedelta object
    multiple=False,                # allow multiple selections (default False)
)

# Add answers (text required, emoji optional)
poll.add_answer(text="Red")
poll.add_answer(text="Blue")
poll.add_answer(text="Green", emoji="\U0001f49a")  # with emoji

# Send the poll (async)
await channel.send(poll=poll)
await ctx.send(poll=poll)
await interaction.response.send_message(poll=poll)

# Poll is passed via the poll= kwarg on send(), NOT as content
```

**Poll constructor parameters**:
| Param | Type | Default | Notes |
|-------|------|---------|-------|
| `question` | `str` | required | The poll question |
| `duration` | `timedelta` | required | How long the poll runs |
| `multiple` | `bool` | `False` | Allow multiple answer selection |
| `layout_type` | `PollLayoutType` | `default` | Only `default` exists currently |

**Poll object attributes** (available after sending):
- `poll.answers` — list of `PollAnswer` objects
- `poll.total_votes` — int (only populated after fetch)
- `poll.is_finalized` / `poll.is_finalised` — bool
- `poll.expires_at` — datetime (when the poll ends)
- `poll.victor_answer` — winning answer after poll ends
- `poll.message` — the Message object (after sending)

**PollAnswer attributes**:
- `answer.text` — str
- `answer.emoji` — optional emoji
- `answer.id` — int (auto-assigned, starts at 1)

**Gotchas**:
- Discord API enforces max **10 answers** per poll. discord.py does NOT validate locally — you get an API error at send time.
- `duration` must be a `timedelta`, not an int. Use `timedelta(hours=24)`.
- Polls are sent via `poll=` kwarg on `.send()`. Cannot combine with `embed=`.
- Poll results (vote counts) require fetching the message again after votes come in.
- `PollLayoutType` only has `default` (value 1) as of discord.py 2.7.1.
