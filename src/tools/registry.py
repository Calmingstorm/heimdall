from __future__ import annotations

TOOLS: list[dict] = [
    {
        "name": "check_service",
        "description": "Check the status of a systemd service on a managed host. Returns the service status output.",
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
        "requires_approval": False,
    },
    {
        "name": "check_docker",
        "description": "List running Docker containers or inspect a specific container on a managed host.",
        "input_schema": {
            "type": "object",
            "properties": {
                "host": {
                    "type": "string",
                    "description": "Host alias from config",
                },
                "container": {
                    "type": "string",
                    "description": "Optional container name to inspect. Omit to list all.",
                },
            },
            "required": ["host"],
        },
        "requires_approval": False,
    },
    {
        "name": "check_disk",
        "description": "Check disk usage on a managed host. Returns df -h output.",
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
        "requires_approval": False,
    },
    {
        "name": "check_memory",
        "description": "Check memory usage on a managed host. Returns free -h output.",
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
        "requires_approval": False,
    },
    {
        "name": "check_logs",
        "description": "Read recent log lines from a systemd service on a managed host.",
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
        "requires_approval": False,
    },
    {
        "name": "query_prometheus",
        "description": "Run a PromQL query against Prometheus. Read-only. Good for checking metrics, uptime, CPU, memory, disk trends.",
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
        "requires_approval": False,
    },
    {
        "name": "restart_service",
        "description": "Restart a systemd service on a managed host.",
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
        "requires_approval": True,
    },
    {
        "name": "run_ansible_playbook",
        "description": "Run an Ansible playbook. Defaults to check mode (dry run) unless check_mode is explicitly set to false.",
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
        "requires_approval": True,
    },
    {
        "name": "run_command",
        "description": "Run a single-line shell command on a managed host. For multi-line scripts, use run_script instead. Use this for simple commands — installing packages, checking status, Docker commands, etc.",
        "input_schema": {
            "type": "object",
            "properties": {
                "host": {
                    "type": "string",
                    "description": "Host alias from config (e.g. 'myserver', 'webhost')",
                },
                "command": {
                    "type": "string",
                    "description": "The shell command to execute (single line preferred; for multi-line use run_script)",
                },
            },
            "required": ["host", "command"],
        },
        "requires_approval": True,
    },
    {
        "name": "run_script",
        "description": (
            "Write a script to a temp file on a managed host and execute it. Use this instead of run_command "
            "for ANY multi-line script, complex commands, heredocs, or code blocks. The script is written to "
            "a temp file, executed with the chosen interpreter, and the temp file is cleaned up. This avoids "
            "all quoting/heredoc issues with SSH."
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
                    "description": "The full script content to execute",
                },
                "interpreter": {
                    "type": "string",
                    "description": "Script interpreter: 'bash' (default), 'python3', 'python', 'sh', 'node', 'ruby', 'perl'",
                },
                "filename": {
                    "type": "string",
                    "description": "Optional filename (default: auto-generated based on interpreter). Used for the temp file.",
                },
            },
            "required": ["host", "script"],
        },
        "requires_approval": True,
    },
    {
        "name": "read_file",
        "description": "Read the contents of a file on a managed host. Does not require approval.",
        "input_schema": {
            "type": "object",
            "properties": {
                "host": {
                    "type": "string",
                    "description": "Host alias from config",
                },
                "path": {
                    "type": "string",
                    "description": "Absolute path to the file to read",
                },
                "lines": {
                    "type": "integer",
                    "description": "Max number of lines to read (default 200). Use for large files.",
                },
            },
            "required": ["host", "path"],
        },
        "requires_approval": False,
    },
    {
        "name": "write_file",
        "description": "Write content to a file on a managed host. Creates the file if it doesn't exist, overwrites if it does.",
        "input_schema": {
            "type": "object",
            "properties": {
                "host": {
                    "type": "string",
                    "description": "Host alias from config",
                },
                "path": {
                    "type": "string",
                    "description": "Absolute path to the file to write",
                },
                "content": {
                    "type": "string",
                    "description": "The content to write to the file",
                },
            },
            "required": ["host", "path", "content"],
        },
        "requires_approval": True,
    },
    {
        "name": "purge_messages",
        "description": "Delete recent messages in the current Discord channel. Also resets conversation history.",
        "input_schema": {
            "type": "object",
            "properties": {
                "count": {
                    "type": "integer",
                    "description": "Number of messages to delete (default 100, max 500)",
                },
            },
        },
        "requires_approval": True,
    },
    {
        "name": "post_file",
        "description": "Fetch a file from a managed host and post it to the current Discord channel. Use this to share images, logs, configs, or any file. Supports images (png, jpg, gif, webp) and text files.",
        "input_schema": {
            "type": "object",
            "properties": {
                "host": {
                    "type": "string",
                    "description": "Host alias from config (e.g. 'myserver', 'webhost')",
                },
                "path": {
                    "type": "string",
                    "description": "Absolute path to the file on the remote host",
                },
                "caption": {
                    "type": "string",
                    "description": "Optional message to include with the file",
                },
            },
            "required": ["host", "path"],
        },
        "requires_approval": False,
    },
    {
        "name": "schedule_task",
        "description": (
            "Schedule a recurring, one-time, or webhook-triggered task. "
            "For reminders: convert the user's natural language time to an ISO datetime for run_at "
            "(e.g. 'in 2 hours' → calculate from current time, 'tomorrow at 9am' → next day 09:00). "
            "Use parse_time if unsure about the conversion. "
            "Recurring tasks use cron expressions (e.g. 'every day at 9am' → '0 9 * * *'). "
            "Webhook-triggered tasks fire when a matching webhook arrives (Gitea push, Grafana alert, etc.). "
            "Actions: 'reminder' posts a message, 'check' runs a read-only monitoring tool, 'digest' runs a full infrastructure digest, "
            "'workflow' runs a multi-step sequence of tools. Workflows can chain any tool (including write tools) and support conditions."
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
                    "description": "Cron expression for recurring tasks (e.g. '0 9 * * *' = daily 9am, '0 */6 * * *' = every 6h). Omit for one-time tasks.",
                },
                "run_at": {
                    "type": "string",
                    "description": "ISO datetime for one-time tasks (e.g. '2026-03-20T09:00'). Convert natural language times to ISO: 'in 2 hours' → add to current time, 'tomorrow at 9am' → next day 09:00. Use parse_time tool if unsure. Omit for recurring tasks.",
                },
                "trigger": {
                    "type": "object",
                    "description": (
                        "For webhook-triggered tasks: conditions to match against incoming webhooks. "
                        "All specified fields must match (AND logic). Omit for time-based tasks. "
                        "Example: {\"source\": \"gitea\", \"event\": \"push\", \"repo\": \"myproject\"} fires on pushes to any repo containing 'myproject'."
                    ),
                    "properties": {
                        "source": {
                            "type": "string",
                            "enum": ["gitea", "grafana", "generic"],
                            "description": "Webhook source to match",
                        },
                        "event": {
                            "type": "string",
                            "description": "Event type to match (e.g. 'push', 'pull_request', 'alert')",
                        },
                        "repo": {
                            "type": "string",
                            "description": "Substring match against repository name (case-insensitive)",
                        },
                        "alert_name": {
                            "type": "string",
                            "description": "Substring match against Grafana alert name (case-insensitive)",
                        },
                    },
                },
                "action": {
                    "type": "string",
                    "enum": ["reminder", "check", "digest", "workflow"],
                    "description": "'reminder' = post a message, 'check' = run a monitoring tool, 'digest' = full infrastructure digest, 'workflow' = multi-step tool chain",
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
                    "description": "For checks: the tool input parameters (e.g. {\"host\": \"myserver\"})",
                },
                "steps": {
                    "type": "array",
                    "description": "For workflows: ordered list of steps to execute sequentially",
                    "items": {
                        "type": "object",
                        "properties": {
                            "tool_name": {
                                "type": "string",
                                "description": "Tool to run in this step",
                            },
                            "tool_input": {
                                "type": "object",
                                "description": "Input parameters for the tool",
                            },
                            "description": {
                                "type": "string",
                                "description": "Human-readable step description",
                            },
                            "condition": {
                                "type": "string",
                                "description": "Optional: only run this step if previous output contains this substring. Prefix with ! to negate (skip if present).",
                            },
                            "on_failure": {
                                "type": "string",
                                "enum": ["abort", "continue"],
                                "description": "What to do if this step fails (default: abort)",
                            },
                        },
                        "required": ["tool_name"],
                    },
                },
            },
            "required": ["description", "action"],
        },
        "requires_approval": True,
    },
    {
        "name": "list_schedules",
        "description": "List all scheduled tasks (recurring and one-time).",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
        "requires_approval": False,
    },
    {
        "name": "delete_schedule",
        "description": "Delete a scheduled task by its ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "schedule_id": {
                    "type": "string",
                    "description": "The schedule ID to delete",
                },
            },
            "required": ["schedule_id"],
        },
        "requires_approval": True,
    },
    {
        "name": "parse_time",
        "description": (
            "Convert a natural language time expression to an ISO datetime string. "
            "Handles: 'in 30 minutes', 'in 2 hours', 'tomorrow at 9am', 'next Monday at 3pm', "
            "'at 5pm', 'friday at noon'. Times use the bot's configured timezone. "
            "Use this before schedule_task if you are unsure about the time conversion."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Natural language time expression (e.g. 'in 2 hours', 'tomorrow at 9am', 'next Friday at 3pm')",
                },
            },
            "required": ["expression"],
        },
        "requires_approval": False,
    },
    {
        "name": "search_history",
        "description": "Search through past conversation history (current and archived sessions) using keyword and semantic search. Can find conversations by meaning even without exact keyword matches. Use this when the user refers to something discussed previously, or to recall past decisions and actions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query to find in past conversations",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max number of results (default 10)",
                },
            },
            "required": ["query"],
        },
        "requires_approval": False,
    },
    {
        "name": "memory_manage",
        "description": "Manage persistent memory notes that survive across conversations. Notes are scoped per user by default (personal) or shared across all users (global). Use 'save' to remember facts, preferences, or decisions. Use 'list' to see all notes (global + personal). Use 'delete' to remove a note. Personal notes are only visible to the user who saved them. Global notes (infrastructure facts, shared knowledge) are visible to everyone.",
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["save", "list", "delete"],
                    "description": "Action to perform: 'save' a note, 'list' all notes, or 'delete' a note",
                },
                "key": {
                    "type": "string",
                    "description": "Short identifier for the note (required for save/delete)",
                },
                "value": {
                    "type": "string",
                    "description": "The content to remember (required for save)",
                },
                "scope": {
                    "type": "string",
                    "enum": ["personal", "global"],
                    "description": "Where to save: 'personal' (default, only this user) or 'global' (shared infrastructure/facts visible to all users)",
                },
            },
            "required": ["action"],
        },
        "requires_approval": False,
    },
    {
        "name": "search_audit",
        "description": "Search the audit log of past tool executions. Returns recent tool calls with who ran them, what inputs were used, whether they were approved, and the result. Useful for reviewing what actions have been taken.",
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
                    "description": "Filter by host alias used in tool_input",
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
                    "description": "Max results to return (default 20)",
                },
            },
        },
        "requires_approval": False,
    },
    {
        "name": "create_digest",
        "description": "Create a scheduled daily infrastructure digest. Runs disk, memory, and service checks across all hosts, queries Prometheus for alerts, and posts a summary to the channel. Use this when asked to set up a daily digest or morning report.",
        "input_schema": {
            "type": "object",
            "properties": {
                "cron": {
                    "type": "string",
                    "description": "Cron expression (default '0 8 * * *' = daily 8am)",
                },
                "description": {
                    "type": "string",
                    "description": "Human-readable description (default 'Daily Infrastructure Digest')",
                },
            },
        },
        "requires_approval": True,
    },
    {
        "name": "create_skill",
        "description": (
            "Create a new skill (tool) by writing Python code. The skill becomes immediately available as a tool you can call.\n"
            "The code must define:\n"
            "1. SKILL_DEFINITION = dict with keys: name, description, input_schema, requires_approval (bool)\n"
            "2. async def execute(inp: dict, context: SkillContext) -> str\n\n"
            "context provides:\n"
            "- await context.run_on_host(alias, command) — SSH to a managed host\n"
            "- await context.query_prometheus(query) — run PromQL query\n"
            "- await context.read_file(host, path, lines=200) — read a remote file\n"
            "- await context.execute_tool(name, input) — call any safe built-in tool (read-only tools only)\n"
            "- await context.http_get(url, params=None, timeout=15) — HTTP GET\n"
            "- await context.http_post(url, json=None, data=None, timeout=15) — HTTP POST\n"
            "- await context.post_message(text) — send a message to the invoking channel\n"
            "- await context.post_file(data, filename, caption) — send a binary file\n"
            "- await context.search_knowledge(query, limit=5) — search the knowledge base\n"
            "- await context.ingest_document(content, source) — add text to the knowledge base\n"
            "- await context.search_history(query, limit=10) — search conversation history\n"
            "- context.schedule_task(description, action, channel_id, **kwargs) — add a scheduled task\n"
            "- context.list_schedules() — list all scheduled tasks\n"
            "- context.delete_schedule(schedule_id) — delete a scheduled task\n"
            "- context.remember(key, value) — save to persistent memory\n"
            "- context.recall(key) — retrieve from persistent memory (returns str or None)\n"
            "- context.get_hosts() — list available host aliases\n"
            "- context.get_services() — list allowed service names\n"
            "- context.log(msg) — write a log message\n"
            "See data/skills/*.template for example skills."
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
                    "description": "Full Python source code for the skill file",
                },
            },
            "required": ["name", "code"],
        },
        "requires_approval": True,
    },
    {
        "name": "edit_skill",
        "description": "Replace the code of an existing user-created skill. The skill is immediately reloaded.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the skill to edit",
                },
                "code": {
                    "type": "string",
                    "description": "New full Python source code",
                },
            },
            "required": ["name", "code"],
        },
        "requires_approval": False,
    },
    {
        "name": "delete_skill",
        "description": "Delete a user-created skill.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the skill to delete",
                },
            },
            "required": ["name"],
        },
        "requires_approval": True,
    },
    {
        "name": "list_skills",
        "description": "List all user-created skills with their descriptions and status.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
        "requires_approval": False,
    },
    # --- Docker tools ---
    {
        "name": "docker_logs",
        "description": "Fetch recent logs from a Docker container on a managed host.",
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
                    "description": "Number of log lines to fetch (default 50, max 200)",
                },
                "since": {
                    "type": "string",
                    "description": "Show logs since timestamp or relative (e.g. '1h', '30m', '2024-01-01T00:00:00')",
                },
            },
            "required": ["host", "container"],
        },
        "requires_approval": False,
    },
    {
        "name": "docker_compose_action",
        "description": "Run a Docker Compose action (up, down, pull, restart, build) on a compose project.",
        "input_schema": {
            "type": "object",
            "properties": {
                "host": {
                    "type": "string",
                    "description": "Host alias from config",
                },
                "project_dir": {
                    "type": "string",
                    "description": "Absolute path to the directory containing docker-compose.yml",
                },
                "action": {
                    "type": "string",
                    "enum": ["up", "down", "pull", "restart", "build"],
                    "description": "Compose action to perform",
                },
            },
            "required": ["host", "project_dir", "action"],
        },
        "requires_approval": True,
    },
    {
        "name": "docker_compose_status",
        "description": "Show status of services in a Docker Compose project. Lists each service with its state (running, exited, restarting).",
        "input_schema": {
            "type": "object",
            "properties": {
                "host": {
                    "type": "string",
                    "description": "Host alias from config",
                },
                "project_dir": {
                    "type": "string",
                    "description": "Absolute path to the directory containing docker-compose.yml",
                },
            },
            "required": ["host", "project_dir"],
        },
        "requires_approval": False,
    },
    {
        "name": "docker_compose_logs",
        "description": "Fetch logs from a Docker Compose project. Can filter to a specific service.",
        "input_schema": {
            "type": "object",
            "properties": {
                "host": {
                    "type": "string",
                    "description": "Host alias from config",
                },
                "project_dir": {
                    "type": "string",
                    "description": "Absolute path to the directory containing docker-compose.yml",
                },
                "service": {
                    "type": "string",
                    "description": "Optional service name to filter logs. Omit for all services.",
                },
                "lines": {
                    "type": "integer",
                    "description": "Number of log lines to fetch (default 50, max 200)",
                },
                "since": {
                    "type": "string",
                    "description": "Show logs since timestamp or relative (e.g. '1h', '30m')",
                },
            },
            "required": ["host", "project_dir"],
        },
        "requires_approval": False,
    },
    {
        "name": "docker_stats",
        "description": "Get CPU, memory, and network stats for Docker containers on a managed host.",
        "input_schema": {
            "type": "object",
            "properties": {
                "host": {
                    "type": "string",
                    "description": "Host alias from config",
                },
                "container": {
                    "type": "string",
                    "description": "Optional container name. Omit to show all running containers.",
                },
            },
            "required": ["host"],
        },
        "requires_approval": False,
    },
    # --- Git tools ---
    {
        "name": "git_status",
        "description": "Check git status of a repository on a managed host. Shows modified, staged, and untracked files.",
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
        "requires_approval": False,
    },
    {
        "name": "git_log",
        "description": "Show recent git commits from a repository on a managed host.",
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
                    "description": "Number of commits to show (default 10, max 50)",
                },
            },
            "required": ["host", "repo_path"],
        },
        "requires_approval": False,
    },
    {
        "name": "git_diff",
        "description": "Show git diff for a repository on a managed host. Shows uncommitted changes by default, or diff of a specific commit.",
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
                    "description": "Optional commit hash to show (e.g. 'HEAD~1', 'abc1234'). Omit for working directory diff.",
                },
            },
            "required": ["host", "repo_path"],
        },
        "requires_approval": False,
    },
    {
        "name": "git_show",
        "description": "Show the full details and diff of a specific git commit.",
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
                    "description": "Commit hash or reference to show (e.g. 'HEAD', 'abc1234', 'HEAD~3')",
                },
            },
            "required": ["host", "repo_path", "commit"],
        },
        "requires_approval": False,
    },
    {
        "name": "git_pull",
        "description": "Pull latest changes from remote for a repository on a managed host.",
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
        "requires_approval": True,
    },
    {
        "name": "git_commit",
        "description": "Stage files and commit in a repository on a managed host. If no files specified, stages all changed files.",
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
                    "description": "Specific files to stage (relative to repo root). If omitted, stages all changed files.",
                },
            },
            "required": ["host", "repo_path", "message"],
        },
        "requires_approval": True,
    },
    {
        "name": "git_push",
        "description": "Push commits to the remote repository on a managed host. Optionally specify a remote and branch.",
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
        "requires_approval": True,
    },
    {
        "name": "git_branch",
        "description": "Create a new branch, switch branches, or list branches in a repository on a managed host.",
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
                    "description": "Action to perform: list branches, create a new branch, or switch to an existing branch",
                },
                "branch_name": {
                    "type": "string",
                    "description": "Branch name (required for create and switch actions)",
                },
            },
            "required": ["host", "repo_path", "action"],
        },
        "requires_approval": True,
    },
    # --- Multi-host tools ---
    {
        "name": "run_command_multi",
        "description": "Run the same shell command on multiple managed hosts in parallel and return consolidated results. Great for 'check everything' requests.",
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
                    "description": "The shell command to execute on each host",
                },
            },
            "required": ["hosts", "command"],
        },
        "requires_approval": True,
    },
    # --- Prometheus range query ---
    {
        "name": "query_prometheus_range",
        "description": "Run a PromQL range query over a time window. Returns data points over time, useful for trends (e.g. 'CPU over the last 6 hours').",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "PromQL query string",
                },
                "duration": {
                    "type": "string",
                    "description": "Time window to query (e.g. '1h', '6h', '24h', '7d'). Default '1h'.",
                },
                "step": {
                    "type": "string",
                    "description": "Query resolution step (e.g. '1m', '5m', '15m'). Default '5m'.",
                },
            },
            "required": ["query"],
        },
        "requires_approval": False,
    },
    # --- Background task delegation ---
    {
        "name": "delegate_task",
        "description": (
            "Delegate a multi-step task to run in the background. Use this when a task requires many tool calls "
            "(e.g., ingesting 20+ files, checking all hosts, batch operations). The task runs independently — "
            "the conversation continues normally. A progress message is posted and updated as steps complete.\n\n"
            "Each step has: tool_name, tool_input, optional description, optional condition (substring match "
            "on previous step output, prefix with ! to negate), optional on_failure ('abort' or 'continue'), "
            "optional store_as (save output to a named variable for later steps via {var.name}).\n"
            "Use {prev_output} in tool_input values to reference the previous step's output."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "Human-readable description of the overall task",
                },
                "steps": {
                    "type": "array",
                    "description": "Ordered list of tool calls to execute sequentially",
                    "items": {
                        "type": "object",
                        "properties": {
                            "tool_name": {"type": "string", "description": "Tool to run"},
                            "tool_input": {"type": "object", "description": "Input parameters"},
                            "description": {"type": "string", "description": "Step description"},
                            "condition": {"type": "string", "description": "Only run if previous output contains this (prefix ! to negate)"},
                            "on_failure": {"type": "string", "enum": ["abort", "continue"], "description": "Default: abort"},
                            "store_as": {"type": "string", "description": "Save this step's output as a named variable"},
                        },
                        "required": ["tool_name"],
                    },
                },
            },
            "required": ["description", "steps"],
        },
        "requires_approval": True,
    },
    {
        "name": "list_tasks",
        "description": "List background tasks, or get full results for a specific task. Without task_id: shows overview of all tasks. With task_id: shows every step's output for that task — use this to summarize what a completed task did.",
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "Optional: get detailed results for a specific task ID",
                },
            },
        },
        "requires_approval": False,
    },
    {
        "name": "cancel_task",
        "description": "Cancel a running background task by its ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "The task ID to cancel (from list_tasks)",
                },
            },
            "required": ["task_id"],
        },
        "requires_approval": False,
    },
    # --- Knowledge base ---
    {
        "name": "search_knowledge",
        "description": (
            "Search the knowledge base of ingested infrastructure documentation, runbooks, configs, and notes. "
            "Use this FIRST when the user asks about infrastructure, configs, deployment procedures, or anything "
            "specific to this environment. Falls back to web_search only if knowledge base has no relevant results."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query describing what you're looking for",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results to return (default 5)",
                },
            },
            "required": ["query"],
        },
        "requires_approval": False,
    },
    {
        "name": "ingest_document",
        "description": (
            "Ingest a document into the knowledge base for future semantic search. "
            "The document is chunked and embedded for retrieval. Use this when the user "
            "uploads a file and asks to remember/embed/index it, or when indexing docs from a host. "
            "Re-ingesting the same source name replaces the previous version."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "Name/identifier for this document (e.g. 'ansible/roles/apache/README.md', 'server-runbook')",
                },
                "content": {
                    "type": "string",
                    "description": "The document text content to ingest. For files on a host, use read_file first then pass the content here.",
                },
            },
            "required": ["source", "content"],
        },
        "requires_approval": False,
    },
    {
        "name": "list_knowledge",
        "description": "List all documents currently in the knowledge base with their source names and chunk counts.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
        "requires_approval": False,
    },
    {
        "name": "delete_knowledge",
        "description": "Remove a document from the knowledge base by source name.",
        "input_schema": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "Source name of the document to remove",
                },
            },
            "required": ["source"],
        },
        "requires_approval": True,
    },
    # --- Browser automation ---
    {
        "name": "browser_screenshot",
        "description": "Navigate to a URL in a headless browser and take a screenshot, posted as a Discord image. Use this for visual inspection of dashboards (Grafana, Semaphore, Pi-hole), web UIs, or any page that needs to be seen. Handles JavaScript-rendered content that fetch_url cannot.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL to navigate to and screenshot",
                },
                "full_page": {
                    "type": "boolean",
                    "description": "Capture the full scrollable page (default false, captures viewport only)",
                },
                "wait_seconds": {
                    "type": "integer",
                    "description": "Extra seconds to wait after page load for dynamic content (default 0, max 10)",
                },
            },
            "required": ["url"],
        },
        "requires_approval": False,
    },
    {
        "name": "browser_read_page",
        "description": "Navigate to a URL in a headless browser and extract the visible text content. Unlike fetch_url, this renders JavaScript first, so it works on SPAs, dashboards, and dynamic pages. Optionally scope to a CSS selector.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL to navigate to",
                },
                "selector": {
                    "type": "string",
                    "description": "Optional CSS selector to scope text extraction (e.g. '#main-content', '.results')",
                },
                "wait_seconds": {
                    "type": "integer",
                    "description": "Extra seconds to wait for dynamic content (default 0, max 10)",
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Max characters to return (default 4000, max 8000)",
                },
            },
            "required": ["url"],
        },
        "requires_approval": False,
    },
    {
        "name": "browser_read_table",
        "description": "Navigate to a URL in a headless browser and extract an HTML table as a markdown table. Useful for scraping structured data from web pages.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL containing the table",
                },
                "table_index": {
                    "type": "integer",
                    "description": "Which table to extract (0-based index, default 0 = first table)",
                },
                "wait_seconds": {
                    "type": "integer",
                    "description": "Extra seconds to wait for dynamic content (default 0, max 10)",
                },
            },
            "required": ["url"],
        },
        "requires_approval": False,
    },
    {
        "name": "browser_click",
        "description": "Navigate to a URL and click an element by CSS selector.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL to navigate to",
                },
                "selector": {
                    "type": "string",
                    "description": "CSS selector of the element to click (e.g. '#login-btn', 'button.submit')",
                },
                "wait_seconds": {
                    "type": "integer",
                    "description": "Extra seconds to wait before clicking (default 0, max 10)",
                },
            },
            "required": ["url", "selector"],
        },
        "requires_approval": True,
    },
    {
        "name": "browser_fill",
        "description": "Navigate to a URL and fill a form field by CSS selector.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL to navigate to",
                },
                "selector": {
                    "type": "string",
                    "description": "CSS selector of the input field (e.g. '#username', 'input[name=password]')",
                },
                "value": {
                    "type": "string",
                    "description": "Text to fill into the field",
                },
                "submit": {
                    "type": "boolean",
                    "description": "Press Enter after filling to submit the form (default false)",
                },
            },
            "required": ["url", "selector", "value"],
        },
        "requires_approval": True,
    },
    {
        "name": "browser_evaluate",
        "description": "Navigate to a URL and run a JavaScript expression, returning the result. Powerful escape hatch for custom scraping or interaction.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL to navigate to",
                },
                "expression": {
                    "type": "string",
                    "description": "JavaScript expression to evaluate (e.g. 'document.title', 'document.querySelectorAll(\"a\").length')",
                },
                "wait_seconds": {
                    "type": "integer",
                    "description": "Extra seconds to wait before evaluating (default 0, max 10)",
                },
            },
            "required": ["url", "expression"],
        },
        "requires_approval": True,
    },
    # --- File generation ---
    {
        "name": "generate_file",
        "description": "Generate a file and post it as a Discord attachment. Use this when the user asks to write a script, code, data export, CSV, report, config file, or any content better served as a downloadable file. When the user asks for code without specifying a host, use this tool.",
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Filename with extension (e.g. 'containers.csv', 'report.md', 'config.yml')",
                },
                "content": {
                    "type": "string",
                    "description": "The file content to generate",
                },
                "caption": {
                    "type": "string",
                    "description": "Optional message to include with the file attachment",
                },
            },
            "required": ["filename", "content"],
        },
        "requires_approval": False,
    },
    # --- Web tools ---
    {
        "name": "web_search",
        "description": "Search the web using DuckDuckGo. Returns top results with titles, URLs, and snippets. Use this when the user asks about current events, documentation, or anything requiring live data.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results (default 5, max 10)",
                },
            },
            "required": ["query"],
        },
        "requires_approval": False,
    },
    {
        "name": "fetch_url",
        "description": "Fetch a URL and return its content as text. Handles HTML (converted to readable text), JSON, and plain text. Use this to read web pages, API responses, or documentation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch",
                },
            },
            "required": ["url"],
        },
        "requires_approval": False,
    },
    # --- Incus tools ---
    {
        "name": "incus_list",
        "description": "List Incus instances (containers/VMs) with status, type, and IP addresses on the configured Incus host.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
        "requires_approval": False,
    },
    {
        "name": "incus_info",
        "description": "Get detailed information about a specific Incus instance, including config, devices, snapshots, and resource usage.",
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
        "requires_approval": False,
    },
    {
        "name": "incus_exec",
        "description": "Execute a command inside an Incus instance.",
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
        "requires_approval": True,
    },
    {
        "name": "incus_start",
        "description": "Start an Incus instance.",
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
        "requires_approval": True,
    },
    {
        "name": "incus_stop",
        "description": "Stop an Incus instance.",
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
        "requires_approval": True,
    },
    {
        "name": "incus_restart",
        "description": "Restart an Incus instance.",
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
        "requires_approval": True,
    },
    {
        "name": "incus_snapshot_list",
        "description": "List snapshots for an Incus instance.",
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
        "requires_approval": False,
    },
    {
        "name": "incus_snapshot",
        "description": "Create, restore, or delete a snapshot for an Incus instance.",
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
                    "description": "Snapshot name (required for restore/delete, optional for create — auto-generated if omitted)",
                },
            },
            "required": ["instance", "action"],
        },
        "requires_approval": True,
    },
    {
        "name": "incus_launch",
        "description": "Launch a new Incus instance from an image (e.g. 'images:ubuntu/24.04', 'images:debian/12').",
        "input_schema": {
            "type": "object",
            "properties": {
                "image": {
                    "type": "string",
                    "description": "Image to launch from (e.g. 'images:ubuntu/24.04')",
                },
                "name": {
                    "type": "string",
                    "description": "Name for the new instance",
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
        "requires_approval": True,
    },
    {
        "name": "incus_delete",
        "description": "Delete an Incus instance.",
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
        "requires_approval": True,
    },
    {
        "name": "incus_logs",
        "description": "Get console log output from an Incus instance.",
        "input_schema": {
            "type": "object",
            "properties": {
                "instance": {
                    "type": "string",
                    "description": "Instance name",
                },
                "lines": {
                    "type": "integer",
                    "description": "Number of lines to show (default: 50, max: 200)",
                },
            },
            "required": ["instance"],
        },
        "requires_approval": False,
    },
    # --- Claude Code ---
    {
        "name": "claude_code",
        "description": (
            "Spawn an AI coding agent for multi-file code tasks: writing features, refactoring, code review, architecture analysis. "
            "Use this when the task requires reasoning about code structure across multiple files. "
            "Do NOT use for: git history (use git_log/git_show/git_diff), reading single files (use read_file), running commands (use run_command). "
            "For code+deploy requests, call this first to write code, then use infrastructure tools to deploy."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "host": {
                    "type": "string",
                    "description": "Host alias to run on (defaults to claude_code_host from config)",
                },
                "working_directory": {
                    "type": "string",
                    "description": "Absolute path to the repo/directory to work in (e.g. '/root/project/')",
                },
                "prompt": {
                    "type": "string",
                    "description": "Detailed prompt describing what to do. Be specific about files, functions, and expected behavior.",
                },
                "allow_edits": {
                    "type": "boolean",
                    "description": "If true, Claude Code can edit/write files (runs as non-root with full permissions). If false (default), read-only analysis only.",
                },
                "allowed_tools": {
                    "type": "string",
                    "description": "Restrict Claude Code's tools — space-separated list (e.g. 'Read Grep Glob' for read-only). Default: all tools.",
                },
            },
            "required": ["working_directory", "prompt"],
        },
        "requires_approval": False,
    },
    # --- Permissions ---
    {
        "name": "set_permission",
        "description": (
            "Change a Discord user's permission tier. Admin-only. "
            "Tiers: 'admin' (full tool access), 'user' (read-only monitoring tools), "
            "'guest' (chat only, no tools). Provide the Discord user ID."
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
                    "description": "Permission tier to assign",
                },
            },
            "required": ["user_id", "tier"],
        },
        "requires_approval": False,
    },
    {
        "name": "manage_list",
        "description": (
            "Manage named lists (grocery, todo, shopping, hardware, gifts, reading, etc.). "
            "Lists are created on the fly when you add to one that doesn't exist yet. "
            "Per-user lists (e.g. 'my todo') are private; shared lists (e.g. 'grocery') "
            "are visible to everyone. Todo/task lists support marking items done/undone. "
            "Use this whenever the user mentions adding, removing, checking, or clearing "
            "items from any kind of list."
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
                        "Action to perform. 'list_all' shows all available lists. "
                        "All other actions operate on a specific list_name."
                    ),
                },
                "list_name": {
                    "type": "string",
                    "description": (
                        "Name of the list (e.g. 'grocery', 'todo', 'hardware store', "
                        "'gift ideas'). Required for all actions except list_all."
                    ),
                },
                "items": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Item(s) to add, remove, or mark. Not needed for show/clear/list_all.",
                },
                "owner": {
                    "type": "string",
                    "enum": ["personal", "shared"],
                    "description": (
                        "Who owns the list. 'personal' = only the requesting user, "
                        "'shared' = visible to everyone. Defaults to 'shared'. "
                        "Only used when creating a new list (first add)."
                    ),
                },
            },
            "required": ["action"],
        },
        "requires_approval": False,
    },
]


def get_tool_definitions() -> list[dict]:
    """Return tool definitions (without internal fields)."""
    return [
        {
            "name": t["name"],
            "description": t["description"],
            "input_schema": t["input_schema"],
        }
        for t in TOOLS
    ]


def requires_approval(tool_name: str) -> bool:
    for t in TOOLS:
        if t["name"] == tool_name:
            return t.get("requires_approval", False)
    return True  # unknown tools require approval by default
