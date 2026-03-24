from __future__ import annotations

# Tool packs: infrastructure tools that are opt-in via config.
# Empty/absent tool_packs in config = ALL tools loaded (backward compatible).
TOOL_PACKS: dict[str, list[str]] = {
    "docker": [
        "check_docker", "docker_logs", "docker_compose_action",
        "docker_compose_status", "docker_compose_logs", "docker_stats",
    ],
    "systemd": ["check_service", "restart_service", "check_logs"],
    "incus": [
        "incus_list", "incus_info", "incus_exec", "incus_start", "incus_stop",
        "incus_restart", "incus_snapshot_list", "incus_snapshot",
        "incus_launch", "incus_delete", "incus_logs",
    ],
    "ansible": ["run_ansible_playbook"],
    "prometheus": [
        "query_prometheus", "query_prometheus_range",
        "check_disk", "check_memory",
    ],
    "git": [
        "git_status", "git_log", "git_diff", "git_show",
        "git_pull", "git_commit", "git_push", "git_branch",
    ],
}

# All tool names that belong to any pack
_ALL_PACK_TOOLS: set[str] = {
    name for tools in TOOL_PACKS.values() for name in tools
}


def get_pack_tool_names(packs: list[str]) -> set[str]:
    """Return the set of tool names enabled by the given packs."""
    result: set[str] = set()
    for pack in packs:
        if pack in TOOL_PACKS:
            result.update(TOOL_PACKS[pack])
    return result


TOOLS: list[dict] = [
    # --- Host monitoring ---
    {
        "name": "check_service",
        "description": "Returns systemd service status on a managed host. Output includes active state, PID, memory, and recent journal lines. For logs only, use check_logs. To restart, use restart_service.",
        "input_schema": {
            "type": "object",
            "properties": {
                "host": {
                    "type": "string",
                    "description": "Host alias from config (e.g. 'myserver', 'webhost')",
                },
                "service": {
                    "type": "string",
                    "description": "Systemd service name (e.g. 'apache2', 'prometheus')",
                },
            },
            "required": ["host", "service"],
        },
    },
    {
        "name": "check_docker",
        "description": "Lists running Docker containers on a managed host. With container name, returns detailed inspect output for that container. For container logs, use docker_logs. For resource stats, use docker_stats.",
        "input_schema": {
            "type": "object",
            "properties": {
                "host": {
                    "type": "string",
                    "description": "Host alias from config",
                },
                "container": {
                    "type": "string",
                    "description": "Container name to inspect. Omit to list all running containers.",
                },
            },
            "required": ["host"],
        },
    },
    {
        "name": "check_disk",
        "description": "Returns disk usage (df -h) on a managed host. Shows all mounted filesystems with size, used, available, and mount point.",
        "input_schema": {
            "type": "object",
            "properties": {
                "host": {
                    "type": "string",
                    "description": "Host alias from config",
                },
            },
            "required": ["host"],
        },
    },
    {
        "name": "check_memory",
        "description": "Returns memory usage (free -h) on a managed host. Shows total, used, free, shared, buffers/cache, and swap.",
        "input_schema": {
            "type": "object",
            "properties": {
                "host": {
                    "type": "string",
                    "description": "Host alias from config",
                },
            },
            "required": ["host"],
        },
    },
    {
        "name": "check_logs",
        "description": "Returns recent journalctl lines from a systemd service. Max 50 lines. For full service status, use check_service.",
        "input_schema": {
            "type": "object",
            "properties": {
                "host": {
                    "type": "string",
                    "description": "Host alias from config",
                },
                "service": {
                    "type": "string",
                    "description": "Systemd service name",
                },
                "lines": {
                    "type": "integer",
                    "description": "Number of recent lines to fetch (max 50, default 20)",
                },
            },
            "required": ["host", "service"],
        },
    },
    # --- Prometheus ---
    {
        "name": "query_prometheus",
        "description": "Runs a PromQL instant query against Prometheus. Returns current metric values as 'N result(s): metric{labels}: value'. Read-only. For historical trends, use query_prometheus_range.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "PromQL query string",
                },
            },
            "required": ["query"],
        },
    },
    # --- Service management ---
    {
        "name": "restart_service",
        "description": "Restarts a systemd service on a managed host. Runs systemctl restart then returns the new status. To check without restarting, use check_service.",
        "input_schema": {
            "type": "object",
            "properties": {
                "host": {
                    "type": "string",
                    "description": "Host alias from config",
                },
                "service": {
                    "type": "string",
                    "description": "Systemd service name to restart",
                },
            },
            "required": ["host", "service"],
        },
    },
    {
        "name": "run_ansible_playbook",
        "description": "Runs an Ansible playbook from the configured playbook directory. Defaults to check mode (dry run) — set check_mode=false for real execution.",
        "input_schema": {
            "type": "object",
            "properties": {
                "playbook": {
                    "type": "string",
                    "description": "Playbook filename (e.g. 'check-services.yml')",
                },
                "limit": {
                    "type": "string",
                    "description": "Limit to specific host(s)",
                },
                "tags": {
                    "type": "string",
                    "description": "Comma-separated Ansible tags",
                },
                "check_mode": {
                    "type": "boolean",
                    "description": "Run in check/dry-run mode (default true)",
                },
            },
            "required": ["playbook"],
        },
    },
    # --- Shell execution ---
    {
        "name": "run_command",
        "description": "Runs a single shell command on a managed host. Returns stdout/stderr (truncated at 200 lines). On failure: 'Command failed (exit N): output'. For multi-line scripts or code blocks, use run_script. For the same command on multiple hosts, use run_command_multi.",
        "input_schema": {
            "type": "object",
            "properties": {
                "host": {
                    "type": "string",
                    "description": "Host alias from config (e.g. 'myserver', 'webhost')",
                },
                "command": {
                    "type": "string",
                    "description": "Shell command to execute (single line; for multi-line use run_script)",
                },
            },
            "required": ["host", "command"],
        },
    },
    {
        "name": "run_script",
        "description": (
            "Writes a script to a temp file on a managed host and executes it. Handles multi-line scripts, "
            "heredocs, code blocks, and complex commands without quoting issues. Temp file is cleaned up after "
            "execution. Returns stdout/stderr (truncated at 200 lines). On failure: 'Script failed (exit N): output'. "
            "Interpreters: bash (default), python3, python, sh, node, ruby, perl. "
            "For single commands, use run_command instead."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "host": {
                    "type": "string",
                    "description": "Host alias from config (e.g. 'myserver', 'webhost')",
                },
                "script": {
                    "type": "string",
                    "description": "Full script content to execute",
                },
                "interpreter": {
                    "type": "string",
                    "description": "Script interpreter: 'bash' (default), 'python3', 'python', 'sh', 'node', 'ruby', 'perl'",
                },
                "filename": {
                    "type": "string",
                    "description": "Optional filename for the temp file (default: auto-generated based on interpreter)",
                },
            },
            "required": ["host", "script"],
        },
    },
    {
        "name": "run_command_multi",
        "description": "Runs the same shell command on multiple managed hosts in parallel. Returns per-host results as markdown blocks: '### hostname\\n```\\noutput\\n```'. Pass ['all'] for every configured host. For a single host, use run_command.",
        "input_schema": {
            "type": "object",
            "properties": {
                "hosts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of host aliases, or ['all'] for all hosts",
                },
                "command": {
                    "type": "string",
                    "description": "Shell command to execute on each host",
                },
            },
            "required": ["hosts", "command"],
        },
    },
    # --- File operations ---
    {
        "name": "read_file",
        "description": "Returns the contents of a file on a managed host. Default 200 lines, max 1000. To write, use write_file. For multi-file analysis, use claude_code with allow_edits=false.",
        "input_schema": {
            "type": "object",
            "properties": {
                "host": {
                    "type": "string",
                    "description": "Host alias from config",
                },
                "path": {
                    "type": "string",
                    "description": "Absolute path to the file",
                },
                "lines": {
                    "type": "integer",
                    "description": "Max lines to read (default 200, max 1000)",
                },
            },
            "required": ["host", "path"],
        },
    },
    {
        "name": "write_file",
        "description": "Writes content to a file on a managed host. Creates the file if missing, overwrites if it exists. To read first, use read_file. For multi-file edits, use claude_code with allow_edits=true.",
        "input_schema": {
            "type": "object",
            "properties": {
                "host": {
                    "type": "string",
                    "description": "Host alias from config",
                },
                "path": {
                    "type": "string",
                    "description": "Absolute path to the file",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write",
                },
            },
            "required": ["host", "path", "content"],
        },
    },
    # --- Discord operations ---
    {
        "name": "purge_messages",
        "description": "Deletes recent messages in the current Discord channel and resets conversation history. Default 100, max 500.",
        "input_schema": {
            "type": "object",
            "properties": {
                "count": {
                    "type": "integer",
                    "description": "Number of messages to delete (default 100, max 500)",
                },
            },
        },
    },
    {
        "name": "post_file",
        "description": "Fetches a file from a managed host and posts it as a Discord attachment. Supports images (png, jpg, gif, webp) and text files. Max 25MB. For generated content, use generate_file instead.",
        "input_schema": {
            "type": "object",
            "properties": {
                "host": {
                    "type": "string",
                    "description": "Host alias from config (e.g. 'myserver', 'webhost')",
                },
                "path": {
                    "type": "string",
                    "description": "Absolute path to the file on the host",
                },
                "caption": {
                    "type": "string",
                    "description": "Optional message to include with the file",
                },
            },
            "required": ["host", "path"],
        },
    },
    {
        "name": "generate_file",
        "description": "Creates a file and posts it as a Discord attachment. For script, code, CSV, report, config, or any downloadable content. Returns the file as a Discord attachment. For files already on a host, use post_file instead.",
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Filename with extension (e.g. 'containers.csv', 'report.md', 'deploy.sh')",
                },
                "content": {
                    "type": "string",
                    "description": "File content to generate",
                },
                "caption": {
                    "type": "string",
                    "description": "Optional message to include with the file",
                },
            },
            "required": ["filename", "content"],
        },
    },
    # --- Scheduling ---
    {
        "name": "schedule_task",
        "description": (
            "Schedules a recurring, one-time, or webhook-triggered task. "
            "Recurring: provide cron expression. One-time: provide run_at (ISO datetime, use parse_time to convert natural language). "
            "Webhook-triggered: provide trigger object matching incoming webhooks (Gitea push, Grafana alert, etc.). "
            "Actions: 'reminder' posts a message, 'check' runs a monitoring tool, 'digest' runs infrastructure digest, "
            "'workflow' runs a multi-step tool chain with conditions and variable substitution."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "Human-readable description (e.g. 'Daily disk check on server')",
                },
                "cron": {
                    "type": "string",
                    "description": "Cron expression for recurring tasks (e.g. '0 9 * * *' = daily 9am). Omit for one-time.",
                },
                "run_at": {
                    "type": "string",
                    "description": "ISO datetime for one-time tasks (e.g. '2026-03-20T09:00'). Use parse_time to convert natural language. Omit for recurring.",
                },
                "trigger": {
                    "type": "object",
                    "description": (
                        "Webhook trigger conditions (AND logic). "
                        "Example: {\"source\": \"gitea\", \"event\": \"push\", \"repo\": \"myproject\"}. Omit for time-based tasks."
                    ),
                    "properties": {
                        "source": {
                            "type": "string",
                            "enum": ["gitea", "grafana", "generic"],
                            "description": "Webhook source to match",
                        },
                        "event": {
                            "type": "string",
                            "description": "Event type (e.g. 'push', 'pull_request', 'alert')",
                        },
                        "repo": {
                            "type": "string",
                            "description": "Repository name substring (case-insensitive)",
                        },
                        "alert_name": {
                            "type": "string",
                            "description": "Grafana alert name substring (case-insensitive)",
                        },
                    },
                },
                "action": {
                    "type": "string",
                    "enum": ["reminder", "check", "digest", "workflow"],
                    "description": "'reminder' = post message, 'check' = monitoring tool, 'digest' = infrastructure digest, 'workflow' = multi-step tool chain",
                },
                "message": {
                    "type": "string",
                    "description": "For reminders: the message to post",
                },
                "tool_name": {
                    "type": "string",
                    "description": "For checks: monitoring tool to run (check_service, check_docker, check_disk, check_memory, check_logs, query_prometheus)",
                },
                "tool_input": {
                    "type": "object",
                    "description": "For checks: tool input parameters (e.g. {\"host\": \"myserver\"})",
                },
                "steps": {
                    "type": "array",
                    "description": "For workflows: ordered steps to execute sequentially",
                    "items": {
                        "type": "object",
                        "properties": {
                            "tool_name": {
                                "type": "string",
                                "description": "Tool to run",
                            },
                            "tool_input": {
                                "type": "object",
                                "description": "Input parameters",
                            },
                            "description": {
                                "type": "string",
                                "description": "Step description",
                            },
                            "condition": {
                                "type": "string",
                                "description": "Run only if previous output contains this substring. Prefix ! to negate.",
                            },
                            "on_failure": {
                                "type": "string",
                                "enum": ["abort", "continue"],
                                "description": "What to do on failure (default: abort)",
                            },
                        },
                        "required": ["tool_name"],
                    },
                },
            },
            "required": ["description", "action"],
        },
    },
    {
        "name": "list_schedules",
        "description": "Returns all scheduled tasks (recurring, one-time, and webhook-triggered) with their IDs, descriptions, and next run times. To delete, use delete_schedule.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "delete_schedule",
        "description": "Deletes a scheduled task by ID. To list schedules first, use list_schedules.",
        "input_schema": {
            "type": "object",
            "properties": {
                "schedule_id": {
                    "type": "string",
                    "description": "Schedule ID to delete",
                },
            },
            "required": ["schedule_id"],
        },
    },
    {
        "name": "parse_time",
        "description": (
            "Converts a natural language time expression to ISO datetime string. "
            "Handles: 'in 30 minutes', 'in 2 hours', 'tomorrow at 9am', 'next Monday at 3pm', "
            "'at 5pm', 'friday at noon'. Uses the bot's configured timezone. "
            "Use this to get the run_at value for schedule_task."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Natural language time (e.g. 'in 2 hours', 'tomorrow at 9am', 'next Friday at 3pm')",
                },
            },
            "required": ["expression"],
        },
    },
    # --- History and memory ---
    {
        "name": "search_history",
        "description": "Searches past conversation history (current and archived sessions) using keyword and semantic matching. Returns timestamped entries: '[date] (role): content'. For ingested docs, use search_knowledge instead.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results (default 10)",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "memory_manage",
        "description": "Persistent memory notes that survive across conversations. 'save' stores facts/preferences, 'list' shows all notes, 'delete' removes a note. Personal notes are per-user; global notes are shared infrastructure knowledge visible to everyone.",
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["save", "list", "delete"],
                    "description": "'save' a note, 'list' all notes, or 'delete' a note",
                },
                "key": {
                    "type": "string",
                    "description": "Short identifier for the note (required for save/delete)",
                },
                "value": {
                    "type": "string",
                    "description": "Content to remember (required for save)",
                },
                "scope": {
                    "type": "string",
                    "enum": ["personal", "global"],
                    "description": "'personal' (default, this user only) or 'global' (shared, visible to all)",
                },
            },
            "required": ["action"],
        },
    },
    # --- Audit ---
    {
        "name": "search_audit",
        "description": "Searches the audit log of past tool executions. Returns entries: '[date] tool_name by user (status, Nms)' with result summary. Filterable by tool name, user, host, keyword, and date.",
        "input_schema": {
            "type": "object",
            "properties": {
                "tool_name": {
                    "type": "string",
                    "description": "Filter by tool name (e.g. 'check_disk', 'restart_service')",
                },
                "user": {
                    "type": "string",
                    "description": "Filter by user name or ID",
                },
                "host": {
                    "type": "string",
                    "description": "Filter by host alias",
                },
                "keyword": {
                    "type": "string",
                    "description": "Free-text search across all fields",
                },
                "date": {
                    "type": "string",
                    "description": "Filter by date prefix (e.g. '2026-03-12')",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results (default 20)",
                },
            },
        },
    },
    {
        "name": "create_digest",
        "description": "Creates a scheduled daily infrastructure digest. Checks disk, memory, services, and Prometheus alerts across all hosts, posts a summary.",
        "input_schema": {
            "type": "object",
            "properties": {
                "cron": {
                    "type": "string",
                    "description": "Cron expression (default '0 8 * * *' = daily 8am)",
                },
                "description": {
                    "type": "string",
                    "description": "Description (default 'Daily Infrastructure Digest')",
                },
            },
        },
    },
    # --- Skills ---
    {
        "name": "create_skill",
        "description": (
            "Creates a new skill (custom tool) from Python code. Immediately available after creation.\n"
            "Code must define: async def execute(inp: dict, context: SkillContext) -> str\n\n"
            "SkillContext API:\n"
            "- await context.run_on_host(alias, command) — run command on managed host\n"
            "- await context.query_prometheus(query) — PromQL query\n"
            "- await context.read_file(host, path, lines=200) — read remote file\n"
            "- await context.execute_tool(name, input) — call any safe built-in tool\n"
            "- await context.http_get(url) / context.http_post(url, json=) — HTTP requests\n"
            "- await context.post_message(text) / context.post_file(data, filename, caption)\n"
            "- await context.search_knowledge(query) / context.ingest_document(content, source)\n"
            "- await context.search_history(query, limit=10)\n"
            "- context.remember(key, value) / context.recall(key) — persistent memory\n"
            "- context.schedule_task(...) / context.list_schedules() / context.delete_schedule(id)\n"
            "- context.get_hosts() / context.get_services() / context.log(msg)\n"
            "See data/skills/*.template for examples."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Skill name (lowercase, underscores only, e.g. 'check_ssl_expiry')",
                },
                "code": {
                    "type": "string",
                    "description": "Full Python source code",
                },
            },
            "required": ["name", "code"],
        },
    },
    {
        "name": "edit_skill",
        "description": "Replaces the code of an existing skill. Immediately reloaded after edit.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Skill name to edit",
                },
                "code": {
                    "type": "string",
                    "description": "New full Python source code",
                },
            },
            "required": ["name", "code"],
        },
    },
    {
        "name": "delete_skill",
        "description": "Deletes a user-created skill. Immediately removed from available tools.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Skill name to delete",
                },
            },
            "required": ["name"],
        },
    },
    {
        "name": "list_skills",
        "description": "Returns all user-created skills with their descriptions, status, and input schemas.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    # --- Docker tools ---
    {
        "name": "docker_logs",
        "description": "Returns recent logs from a Docker container. Supports --since for time-filtered output. For container status, use check_docker. For resource stats, use docker_stats.",
        "input_schema": {
            "type": "object",
            "properties": {
                "host": {
                    "type": "string",
                    "description": "Host alias from config",
                },
                "container": {
                    "type": "string",
                    "description": "Container name",
                },
                "lines": {
                    "type": "integer",
                    "description": "Log lines to fetch (default 50, max 200)",
                },
                "since": {
                    "type": "string",
                    "description": "Show logs since timestamp or relative (e.g. '1h', '30m', '2024-01-01T00:00:00')",
                },
            },
            "required": ["host", "container"],
        },
    },
    {
        "name": "docker_compose_action",
        "description": "Runs a Docker Compose action: up (-d), down, pull, restart, or build on a compose project directory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "host": {
                    "type": "string",
                    "description": "Host alias from config",
                },
                "project_dir": {
                    "type": "string",
                    "description": "Absolute path to directory containing docker-compose.yml",
                },
                "action": {
                    "type": "string",
                    "enum": ["up", "down", "pull", "restart", "build"],
                    "description": "Compose action to perform",
                },
            },
            "required": ["host", "project_dir", "action"],
        },
    },
    {
        "name": "docker_compose_status",
        "description": "Returns status of services in a Docker Compose project. Shows each service with state (running, exited, restarting).",
        "input_schema": {
            "type": "object",
            "properties": {
                "host": {
                    "type": "string",
                    "description": "Host alias from config",
                },
                "project_dir": {
                    "type": "string",
                    "description": "Absolute path to directory containing docker-compose.yml",
                },
            },
            "required": ["host", "project_dir"],
        },
    },
    {
        "name": "docker_compose_logs",
        "description": "Returns logs from a Docker Compose project. Optionally filtered to a single service.",
        "input_schema": {
            "type": "object",
            "properties": {
                "host": {
                    "type": "string",
                    "description": "Host alias from config",
                },
                "project_dir": {
                    "type": "string",
                    "description": "Absolute path to directory containing docker-compose.yml",
                },
                "service": {
                    "type": "string",
                    "description": "Service name to filter. Omit for all services.",
                },
                "lines": {
                    "type": "integer",
                    "description": "Log lines to fetch (default 50, max 200)",
                },
                "since": {
                    "type": "string",
                    "description": "Show logs since timestamp or relative (e.g. '1h', '30m')",
                },
            },
            "required": ["host", "project_dir"],
        },
    },
    {
        "name": "docker_stats",
        "description": "Returns CPU, memory, and network I/O stats for Docker containers on a managed host. For container logs, use docker_logs.",
        "input_schema": {
            "type": "object",
            "properties": {
                "host": {
                    "type": "string",
                    "description": "Host alias from config",
                },
                "container": {
                    "type": "string",
                    "description": "Container name. Omit for all running containers.",
                },
            },
            "required": ["host"],
        },
    },
    # --- Git tools ---
    {
        "name": "git_status",
        "description": "Returns git status for a repository. Shows modified, staged, and untracked files. For commit history, use git_log. For diffs, use git_diff.",
        "input_schema": {
            "type": "object",
            "properties": {
                "host": {
                    "type": "string",
                    "description": "Host alias from config",
                },
                "repo_path": {
                    "type": "string",
                    "description": "Absolute path to the git repository",
                },
            },
            "required": ["host", "repo_path"],
        },
    },
    {
        "name": "git_log",
        "description": "Returns recent commit history (git log --oneline --graph) from a repository. Max 50 commits. For commit details, use git_show. For diffs, use git_diff. For working tree status, use git_status.",
        "input_schema": {
            "type": "object",
            "properties": {
                "host": {
                    "type": "string",
                    "description": "Host alias from config",
                },
                "repo_path": {
                    "type": "string",
                    "description": "Absolute path to the git repository",
                },
                "count": {
                    "type": "integer",
                    "description": "Number of commits (default 10, max 50)",
                },
            },
            "required": ["host", "repo_path"],
        },
    },
    {
        "name": "git_diff",
        "description": "Returns unified diff output for a repository. Without commit: shows uncommitted changes. With commit: shows that commit's diff. For commit messages, use git_log or git_show. For working tree status, use git_status.",
        "input_schema": {
            "type": "object",
            "properties": {
                "host": {
                    "type": "string",
                    "description": "Host alias from config",
                },
                "repo_path": {
                    "type": "string",
                    "description": "Absolute path to the git repository",
                },
                "commit": {
                    "type": "string",
                    "description": "Commit hash (e.g. 'HEAD~1', 'abc1234'). Omit for working directory diff.",
                },
            },
            "required": ["host", "repo_path"],
        },
    },
    {
        "name": "git_show",
        "description": "Returns full details and diff of a specific git commit. Includes author, date, message, and changed files. For log overview, use git_log. For working directory changes, use git_diff without commit.",
        "input_schema": {
            "type": "object",
            "properties": {
                "host": {
                    "type": "string",
                    "description": "Host alias from config",
                },
                "repo_path": {
                    "type": "string",
                    "description": "Absolute path to the git repository",
                },
                "commit": {
                    "type": "string",
                    "description": "Commit hash or reference (e.g. 'HEAD', 'abc1234', 'HEAD~3')",
                },
            },
            "required": ["host", "repo_path", "commit"],
        },
    },
    {
        "name": "git_pull",
        "description": "Pulls latest changes from remote for a repository on a managed host.",
        "input_schema": {
            "type": "object",
            "properties": {
                "host": {
                    "type": "string",
                    "description": "Host alias from config",
                },
                "repo_path": {
                    "type": "string",
                    "description": "Absolute path to the git repository",
                },
            },
            "required": ["host", "repo_path"],
        },
    },
    {
        "name": "git_commit",
        "description": "Stages files and commits in a repository. Without files list, stages all changed files. To push after committing, use git_push.",
        "input_schema": {
            "type": "object",
            "properties": {
                "host": {
                    "type": "string",
                    "description": "Host alias from config",
                },
                "repo_path": {
                    "type": "string",
                    "description": "Absolute path to the git repository",
                },
                "message": {
                    "type": "string",
                    "description": "Commit message",
                },
                "files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Files to stage (relative to repo root). Omit to stage all changes.",
                },
            },
            "required": ["host", "repo_path", "message"],
        },
    },
    {
        "name": "git_push",
        "description": "Pushes commits to the remote repository. To commit first, use git_commit.",
        "input_schema": {
            "type": "object",
            "properties": {
                "host": {
                    "type": "string",
                    "description": "Host alias from config",
                },
                "repo_path": {
                    "type": "string",
                    "description": "Absolute path to the git repository",
                },
                "remote": {
                    "type": "string",
                    "description": "Remote name (default: origin)",
                },
                "branch": {
                    "type": "string",
                    "description": "Branch to push (default: current branch)",
                },
            },
            "required": ["host", "repo_path"],
        },
    },
    {
        "name": "git_branch",
        "description": "Lists, creates, or switches branches in a repository.",
        "input_schema": {
            "type": "object",
            "properties": {
                "host": {
                    "type": "string",
                    "description": "Host alias from config",
                },
                "repo_path": {
                    "type": "string",
                    "description": "Absolute path to the git repository",
                },
                "action": {
                    "type": "string",
                    "enum": ["list", "create", "switch"],
                    "description": "list, create, or switch branches",
                },
                "branch_name": {
                    "type": "string",
                    "description": "Branch name (required for create/switch)",
                },
            },
            "required": ["host", "repo_path", "action"],
        },
    },
    # --- Prometheus range query ---
    {
        "name": "query_prometheus_range",
        "description": "Runs a PromQL range query over a time window. Returns series as 'metric{labels}: N points [first → last]'. For trend analysis (e.g. CPU over 6 hours). For current values only, use query_prometheus.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "PromQL query string",
                },
                "duration": {
                    "type": "string",
                    "description": "Time window (e.g. '1h', '6h', '24h', '7d'). Default '1h'.",
                },
                "step": {
                    "type": "string",
                    "description": "Resolution step (e.g. '1m', '5m', '15m'). Default '5m'.",
                },
            },
            "required": ["query"],
        },
    },
    # --- Background task delegation ---
    {
        "name": "delegate_task",
        "description": (
            "Runs a multi-step task in the background. Returns immediately with task ID; progress updates post to Discord. "
            "Steps execute sequentially. Support: conditions (substring match on previous output, prefix ! to negate), "
            "on_failure (abort/continue), store_as (save output to named variable, reference via {var.name}), "
            "{prev_output} substitution. Track with list_tasks, stop with cancel_task."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "Task description",
                },
                "steps": {
                    "type": "array",
                    "description": "Ordered tool calls to execute",
                    "items": {
                        "type": "object",
                        "properties": {
                            "tool_name": {"type": "string", "description": "Tool to run"},
                            "tool_input": {"type": "object", "description": "Input parameters"},
                            "description": {"type": "string", "description": "Step description"},
                            "condition": {"type": "string", "description": "Run if previous output contains this (prefix ! to negate)"},
                            "on_failure": {"type": "string", "enum": ["abort", "continue"], "description": "Default: abort"},
                            "store_as": {"type": "string", "description": "Save output as named variable"},
                        },
                        "required": ["tool_name"],
                    },
                },
            },
            "required": ["description", "steps"],
        },
    },
    {
        "name": "list_tasks",
        "description": "Returns background tasks. Without task_id: overview of all (running/completed/failed). With task_id: detailed step-by-step results. See delegate_task to create, cancel_task to stop.",
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "Task ID for detailed results (omit for overview)",
                },
            },
        },
    },
    {
        "name": "cancel_task",
        "description": "Cancels a running background task. Get task IDs from list_tasks.",
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "Task ID to cancel (from list_tasks)",
                },
            },
            "required": ["task_id"],
        },
    },
    # --- Knowledge base ---
    {
        "name": "search_knowledge",
        "description": (
            "Searches the knowledge base of ingested documentation, runbooks, configs, and notes. "
            "Returns ranked chunks: '[source] (score: N) content'. "
            "Search here FIRST for environment-specific questions before falling back to web_search. "
            "To add documents, use ingest_document. To list sources, use list_knowledge."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results (default 5)",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "ingest_document",
        "description": (
            "Ingests a document into the knowledge base. Content is chunked and embedded for semantic search. "
            "Re-ingesting the same source name replaces the previous version. "
            "For files on a host, read_file first then pass the content here. Search with search_knowledge."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "Document identifier (e.g. 'ansible/roles/apache/README.md', 'server-runbook')",
                },
                "content": {
                    "type": "string",
                    "description": "Document text content",
                },
            },
            "required": ["source", "content"],
        },
    },
    {
        "name": "list_knowledge",
        "description": "Returns all documents in the knowledge base with source names and chunk counts. To search, use search_knowledge. To remove, use delete_knowledge.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "delete_knowledge",
        "description": "Removes a document from the knowledge base by source name. To list sources first, use list_knowledge.",
        "input_schema": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "Source name to remove",
                },
            },
            "required": ["source"],
        },
    },
    # --- Browser automation ---
    {
        "name": "browser_screenshot",
        "description": "Navigates to a URL in a headless browser, takes a screenshot, and posts it to Discord. Renders JavaScript — works on dashboards (Grafana, Semaphore, Pi-hole), SPAs, and dynamic pages that fetch_url cannot handle. For text extraction, use browser_read_page.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL to screenshot",
                },
                "full_page": {
                    "type": "boolean",
                    "description": "Capture full scrollable page (default false = viewport only)",
                },
                "wait_seconds": {
                    "type": "integer",
                    "description": "Extra wait after page load for dynamic content (default 0, max 10)",
                },
            },
            "required": ["url"],
        },
    },
    {
        "name": "browser_read_page",
        "description": "Navigates to a URL in a headless browser and returns 'Title (url)\\n\\ntext'. Renders JavaScript first — works on SPAs and dynamic pages unlike fetch_url. Optionally scoped to a CSS selector. For tables, use browser_read_table. For screenshots, use browser_screenshot.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL to read",
                },
                "selector": {
                    "type": "string",
                    "description": "CSS selector to scope extraction (e.g. '#main-content', '.results')",
                },
                "wait_seconds": {
                    "type": "integer",
                    "description": "Extra wait for dynamic content (default 0, max 10)",
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Max characters to return (default 4000, max 8000)",
                },
            },
            "required": ["url"],
        },
    },
    {
        "name": "browser_read_table",
        "description": "Navigates to a URL in a headless browser and extracts an HTML table as markdown (| col | col |). Returns structured tabular data. For general text, use browser_read_page.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL containing the table",
                },
                "table_index": {
                    "type": "integer",
                    "description": "Which table to extract (0-based, default 0 = first table)",
                },
                "wait_seconds": {
                    "type": "integer",
                    "description": "Extra wait for dynamic content (default 0, max 10)",
                },
            },
            "required": ["url"],
        },
    },
    {
        "name": "browser_click",
        "description": "Navigates to a URL and clicks an element by CSS selector. Returns visible page text after clicking. To fill forms, use browser_fill.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL to navigate to",
                },
                "selector": {
                    "type": "string",
                    "description": "CSS selector to click (e.g. '#login-btn', 'button.submit')",
                },
                "wait_seconds": {
                    "type": "integer",
                    "description": "Extra wait before clicking (default 0, max 10)",
                },
            },
            "required": ["url", "selector"],
        },
    },
    {
        "name": "browser_fill",
        "description": "Navigates to a URL and fills a form field by CSS selector. Optionally submits by pressing Enter. To click buttons, use browser_click.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL to navigate to",
                },
                "selector": {
                    "type": "string",
                    "description": "CSS selector of the input (e.g. '#username', 'input[name=password]')",
                },
                "value": {
                    "type": "string",
                    "description": "Text to fill",
                },
                "submit": {
                    "type": "boolean",
                    "description": "Press Enter after filling (default false)",
                },
            },
            "required": ["url", "selector", "value"],
        },
    },
    {
        "name": "browser_evaluate",
        "description": "Navigates to a URL and evaluates a JavaScript expression. Returns the expression result. Escape hatch for custom scraping or interaction.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL to navigate to",
                },
                "expression": {
                    "type": "string",
                    "description": "JavaScript expression (e.g. 'document.title', 'document.querySelectorAll(\"a\").length')",
                },
                "wait_seconds": {
                    "type": "integer",
                    "description": "Extra wait before evaluating (default 0, max 10)",
                },
            },
            "required": ["url", "expression"],
        },
    },
    # --- Web tools ---
    {
        "name": "web_search",
        "description": "Searches the web via DuckDuckGo. Returns numbered results: 'N. title\\nurl\\nsnippet'. Max 10 results. For full page content, follow up with fetch_url or browser_read_page.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max results (default 5, max 10)",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "fetch_url",
        "description": "Fetches a URL and returns content as text (HTML converted to readable text, JSON passed through). Static only — for JavaScript-rendered pages use browser_read_page. To find URLs first, use web_search.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL to fetch",
                },
            },
            "required": ["url"],
        },
    },
    # --- Incus tools ---
    {
        "name": "incus_list",
        "description": "Returns all Incus instances (containers/VMs) as a formatted table: name, status, type, IPv4. For details on one instance, use incus_info.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "incus_info",
        "description": "Returns detailed info for an Incus instance: config, devices, snapshots, and resource usage. For a list of all instances, use incus_list.",
        "input_schema": {
            "type": "object",
            "properties": {
                "instance": {
                    "type": "string",
                    "description": "Instance name",
                },
            },
            "required": ["instance"],
        },
    },
    {
        "name": "incus_exec",
        "description": "Executes a command inside an Incus instance. Returns exit code and output. For host-level commands, use run_command instead. For console logs, use incus_logs.",
        "input_schema": {
            "type": "object",
            "properties": {
                "instance": {
                    "type": "string",
                    "description": "Instance name",
                },
                "command": {
                    "type": "string",
                    "description": "Command to execute inside the instance",
                },
                "user": {
                    "type": "string",
                    "description": "Run as this user (default: root)",
                },
            },
            "required": ["instance", "command"],
        },
    },
    {
        "name": "incus_start",
        "description": "Starts an Incus instance.",
        "input_schema": {
            "type": "object",
            "properties": {
                "instance": {
                    "type": "string",
                    "description": "Instance name",
                },
            },
            "required": ["instance"],
        },
    },
    {
        "name": "incus_stop",
        "description": "Stops an Incus instance. With force=true, kills immediately.",
        "input_schema": {
            "type": "object",
            "properties": {
                "instance": {
                    "type": "string",
                    "description": "Instance name",
                },
                "force": {
                    "type": "boolean",
                    "description": "Force stop (default: false)",
                },
            },
            "required": ["instance"],
        },
    },
    {
        "name": "incus_restart",
        "description": "Restarts an Incus instance.",
        "input_schema": {
            "type": "object",
            "properties": {
                "instance": {
                    "type": "string",
                    "description": "Instance name",
                },
                "force": {
                    "type": "boolean",
                    "description": "Force restart (default: false)",
                },
            },
            "required": ["instance"],
        },
    },
    {
        "name": "incus_snapshot_list",
        "description": "Returns all snapshots for an Incus instance with names and creation times.",
        "input_schema": {
            "type": "object",
            "properties": {
                "instance": {
                    "type": "string",
                    "description": "Instance name",
                },
            },
            "required": ["instance"],
        },
    },
    {
        "name": "incus_snapshot",
        "description": "Creates, restores, or deletes a snapshot for an Incus instance.",
        "input_schema": {
            "type": "object",
            "properties": {
                "instance": {
                    "type": "string",
                    "description": "Instance name",
                },
                "action": {
                    "type": "string",
                    "enum": ["create", "restore", "delete"],
                    "description": "Snapshot action",
                },
                "snapshot": {
                    "type": "string",
                    "description": "Snapshot name (required for restore/delete, auto-generated for create if omitted)",
                },
            },
            "required": ["instance", "action"],
        },
    },
    {
        "name": "incus_launch",
        "description": "Launches a new Incus instance from an image. Supports containers and VMs. To remove, use incus_delete.",
        "input_schema": {
            "type": "object",
            "properties": {
                "image": {
                    "type": "string",
                    "description": "Image (e.g. 'images:ubuntu/24.04', 'images:debian/12')",
                },
                "name": {
                    "type": "string",
                    "description": "Instance name",
                },
                "type": {
                    "type": "string",
                    "enum": ["container", "vm"],
                    "description": "Instance type (default: container)",
                },
                "profile": {
                    "type": "string",
                    "description": "Profile to apply (default: 'default')",
                },
            },
            "required": ["image", "name"],
        },
    },
    {
        "name": "incus_delete",
        "description": "Deletes an Incus instance. With force=true, deletes even if running. To create, use incus_launch.",
        "input_schema": {
            "type": "object",
            "properties": {
                "instance": {
                    "type": "string",
                    "description": "Instance name",
                },
                "force": {
                    "type": "boolean",
                    "description": "Force delete even if running (default: false)",
                },
            },
            "required": ["instance"],
        },
    },
    {
        "name": "incus_logs",
        "description": "Returns console log output from an Incus instance. Max 200 lines. To run commands inside, use incus_exec.",
        "input_schema": {
            "type": "object",
            "properties": {
                "instance": {
                    "type": "string",
                    "description": "Instance name",
                },
                "lines": {
                    "type": "integer",
                    "description": "Lines to show (default 50, max 200)",
                },
            },
            "required": ["instance"],
        },
    },
    # --- Claude Code ---
    {
        "name": "claude_code",
        "description": (
            "Deep reasoning agent for complex multi-step tasks: code generation, repo analysis, debugging, "
            "building/deploying projects, reading docs and following instructions, architecture review — "
            "anything that would take 3+ direct tool calls step-by-step. Runs the entire chain in one session "
            "with no context loss. Results return as text + files on disk. "
            "With allow_edits=true, appends 'FILES ON DISK: ...' manifest listing written files.\n"
            "NOT for: git history (use git_log/git_show/git_diff), reading single files (use read_file), "
            "running single commands (use run_command). For single-file writes, use write_file.\n"
            "For code+deploy: call this first to write code, then use infrastructure tools to deploy."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "host": {
                    "type": "string",
                    "description": "Host alias (defaults to claude_code_host from config)",
                },
                "working_directory": {
                    "type": "string",
                    "description": "Absolute path to the repo/directory (e.g. '/root/project/')",
                },
                "prompt": {
                    "type": "string",
                    "description": "Detailed prompt: specify files, functions, and expected behavior",
                },
                "allow_edits": {
                    "type": "boolean",
                    "description": "true = can edit/write files (non-root). false (default) = read-only analysis.",
                },
                "allowed_tools": {
                    "type": "string",
                    "description": "Restrict tools — space-separated (e.g. 'Read Grep Glob'). Default: all.",
                },
            },
            "required": ["working_directory", "prompt"],
        },
    },
    # --- Permissions ---
    {
        "name": "set_permission",
        "description": (
            "Sets a Discord user's permission tier. Admin-only. "
            "Tiers: 'admin' (full tool access), 'user' (read-only monitoring), "
            "'guest' (chat only, no tools)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "Discord user ID (numeric string, e.g. '123456789012345678')",
                },
                "tier": {
                    "type": "string",
                    "enum": ["admin", "user", "guest"],
                    "description": "Permission tier",
                },
            },
            "required": ["user_id", "tier"],
        },
    },
    # --- PDF analysis ---
    {
        "name": "analyze_pdf",
        "description": "Extracts text from a PDF file. Accepts a URL or host:path. "
                       "Returns markdown-formatted text content. "
                       "For host files, fetches via read_file. For URLs, downloads directly. "
                       "Truncated to 12000 chars. For image-heavy PDFs, use browser_screenshot.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to fetch PDF from"},
                "host": {"type": "string", "description": "Host alias for file-based PDF"},
                "path": {"type": "string", "description": "File path on host"},
                "pages": {"type": "string", "description": "Page range, e.g. '1-5' or '3' (default: all)"},
            },
        },
    },
    # --- List management ---
    {
        "name": "manage_list",
        "description": (
            "Manages named lists (grocery, todo, shopping, hardware, gifts, etc.). "
            "Lists are created on first add. Per-user lists are private; shared lists are visible to everyone. "
            "Todo/task lists support mark_done/mark_undone."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "add", "remove", "show", "clear",
                        "mark_done", "mark_undone", "list_all",
                    ],
                    "description": (
                        "'list_all' shows all lists. "
                        "Other actions operate on a specific list_name."
                    ),
                },
                "list_name": {
                    "type": "string",
                    "description": (
                        "List name (e.g. 'grocery', 'todo', 'hardware store'). "
                        "Required for all actions except list_all."
                    ),
                },
                "items": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Item(s) to add, remove, or mark.",
                },
                "owner": {
                    "type": "string",
                    "enum": ["personal", "shared"],
                    "description": (
                        "'personal' = this user only, 'shared' = everyone (default). "
                        "Only applies on first add (list creation)."
                    ),
                },
            },
            "required": ["action"],
        },
    },
]


def get_tool_definitions(enabled_packs: list[str] | None = None) -> list[dict]:
    """Return tool definitions filtered by enabled packs.

    When enabled_packs is None or empty: returns ALL tools (backward compatible).
    When packs specified: returns core tools (not in any pack) + tools from enabled packs.
    """
    if enabled_packs:
        allowed = get_pack_tool_names(enabled_packs)
        return [
            {
                "name": t["name"],
                "description": t["description"],
                "input_schema": t["input_schema"],
            }
            for t in TOOLS
            if t["name"] not in _ALL_PACK_TOOLS or t["name"] in allowed
        ]
    return [
        {
            "name": t["name"],
            "description": t["description"],
            "input_schema": t["input_schema"],
        }
        for t in TOOLS
    ]
