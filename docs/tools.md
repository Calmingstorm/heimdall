# Tools

Heimdall has 61 built-in tools. All execute immediately when called — no approval prompts, no confirmation.

## Tool Categories

| Category | Count | Tools |
|----------|-------|-------|
| Command Execution | 3 | `run_command`, `run_command_multi`, `run_script` |
| File Operations | 3 | `read_file`, `write_file`, `post_file` |
| Browser | 6 | `browser_screenshot`, `browser_read_page`, `browser_read_table`, `browser_click`, `browser_fill`, `browser_evaluate` |
| Knowledge Base | 4 | `search_knowledge`, `ingest_document`, `list_knowledge`, `delete_knowledge` |
| Scheduling | 3 | `schedule_task`, `list_schedules`, `delete_schedule` |
| Skills | 9 | `create_skill`, `edit_skill`, `delete_skill`, `list_skills`, `enable_skill`, `disable_skill`, `install_skill`, `export_skill`, `skill_status` |
| Agents | 8 | `spawn_agent`, `send_to_agent`, `list_agents`, `kill_agent`, `get_agent_results`, `wait_for_agents`, `spawn_loop_agents`, `collect_loop_agents` |
| Autonomous Loops | 3 | `start_loop`, `stop_loop`, `list_loops` |
| Background Tasks | 3 | `delegate_task`, `list_tasks`, `cancel_task` |
| Web | 2 | `web_search`, `fetch_url` |
| Deep Reasoning | 1 | `claude_code` (hidden when `claude_code_host` is not configured) |
| PDF & Images | 3 | `analyze_pdf`, `analyze_image`, `generate_image` |
| Discord | 4 | `purge_messages`, `read_channel`, `add_reaction`, `create_poll` |
| Process Management | 2 | `manage_process`, `manage_list` |
| Other | 7 | `generate_file`, `parse_time`, `memory_manage`, `search_history`, `search_audit`, `create_digest`, `set_permission` |

## Execution Model

Tools execute through a dispatch layer that routes commands to the right host:

```
Tool call → _exec_command(host, command)
                ├── localhost? → subprocess (direct, no SSH)
                └── remote?    → SSH connection
```

- **Concurrent execution** — multiple tool calls in a single LLM turn run in parallel via `asyncio.gather`
- **Output truncation** — tool output is capped at 12,000 characters before the LLM sees it
- **Secret scrubbing** — credentials and tokens are redacted from tool output
- **Timeout enforcement** — each tool call has a configurable timeout (default 300s)

## Tool Loop

The tool loop runs up to 20 iterations per message:

1. GPT-5.4 receives the message + tool definitions
2. Model returns tool calls (or text response)
3. Tools execute concurrently
4. Results are fed back to the model
5. Repeat until the model responds with text (no more tool calls)
6. Completion classifier judges the response before sending

The completion classifier ensures the model actually finished the full request — not just part of it.
