# Agents

Heimdall supports multi-agent orchestration for parallel autonomous work.

## How It Works

Agents are independent LLM sessions spawned by the main bot. Each agent gets its own conversation context, tool access, and iteration budget. Agents work autonomously toward a goal and return results to the parent.

## Spawning Agents

```
@Heimdall spawn an agent to check disk usage on all hosts while another agent reviews the nginx configs
```

Heimdall can spawn multiple agents in parallel. Each agent executes independently with full tool access (except spawning sub-agents — no nesting).

## Limits

| Limit | Value |
|-------|-------|
| Concurrent agents per channel | 5 |
| Max iterations per agent | 30 |
| Max lifetime | 1 hour |
| Agents per loop iteration | 3 |
| Agents per loop (lifetime) | 10 |

## Agent Tools

| Tool | Description |
|------|-------------|
| `spawn_agent` | Start a new autonomous agent with a goal |
| `send_to_agent` | Send a message to a running agent |
| `list_agents` | List all active agents |
| `kill_agent` | Terminate an agent |
| `get_agent_results` | Get an agent's output |
| `wait_for_agents` | Wait for agents to complete |
| `spawn_loop_agents` | Spawn agents from within an autonomous loop iteration |
| `collect_loop_agents` | Collect results from loop-spawned agents |

## Loop Integration

Autonomous loops can spawn agents via the `LoopAgentBridge` for parallel subtasks. This allows a recurring loop to delegate work to multiple agents each iteration.

## Silent Agents

Agents do not post directly to Discord. The parent bot collects agent results and delivers one cohesive response. Agent output is context for the parent, not output for the user.
