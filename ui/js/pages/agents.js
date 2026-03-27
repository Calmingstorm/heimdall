/**
 * Heimdall Management UI — Agents Page
 * Live agent status cards with real-time updates, tool usage, kill controls
 */
import { api } from '../api.js';

const { ref, computed, onMounted, onUnmounted } = Vue;

export default {
  template: `
    <div class="p-6 page-fade-in">
      <div class="flex items-center justify-between mb-4 flex-wrap gap-2">
        <h1 class="text-xl font-semibold">Agents</h1>
        <div class="flex gap-2 items-center">
          <label class="flex items-center gap-1 text-xs text-gray-400 cursor-pointer">
            <input type="checkbox" v-model="autoRefresh" class="ag-checkbox" />
            Auto-refresh
          </label>
          <button @click="fetchAgents" class="btn btn-ghost text-xs" :disabled="loading">
            {{ loading ? 'Loading...' : 'Refresh' }}
          </button>
        </div>
      </div>

      <!-- Summary stats -->
      <div v-if="agents.length > 0" class="ag-stats-bar">
        <div class="ag-stat">
          <span class="ag-stat-value">{{ agents.length }}</span>
          <span class="ag-stat-label">Total</span>
        </div>
        <div class="ag-stat">
          <span class="ag-stat-value ag-stat-running">{{ runningCount }}</span>
          <span class="ag-stat-label">Running</span>
        </div>
        <div class="ag-stat">
          <span class="ag-stat-value ag-stat-completed">{{ completedCount }}</span>
          <span class="ag-stat-label">Completed</span>
        </div>
        <div class="ag-stat">
          <span class="ag-stat-value ag-stat-failed">{{ failedCount }}</span>
          <span class="ag-stat-label">Failed</span>
        </div>
      </div>

      <!-- Status filter -->
      <div v-if="agents.length > 0" class="ag-filter-bar" role="toolbar" aria-label="Filter agents by status">
        <button v-for="f in statusFilters" :key="f.value"
                class="ag-filter-btn" :class="{ 'ag-filter-active': statusFilter === f.value }"
                @click="statusFilter = f.value"
                :aria-pressed="statusFilter === f.value">
          {{ f.label }}
          <span v-if="f.count > 0" class="ag-filter-count">{{ f.count }}</span>
        </button>
      </div>

      <!-- Loading -->
      <div v-if="loading && agents.length === 0" class="space-y-2">
        <div v-for="n in 3" :key="n" class="skeleton skeleton-row"></div>
      </div>
      <div v-else-if="error" class="hm-card border-red-900 error-state" role="alert">
        <span class="error-icon" aria-hidden="true">\u26A0</span>
        <p class="text-red-400">{{ error }}</p>
        <button @click="fetchAgents" class="btn btn-ghost text-xs">Retry</button>
      </div>
      <div v-else-if="agents.length === 0" class="hm-card empty-state">
        <span class="empty-state-icon">\u{1F916}</span>
        <span class="empty-state-text">No agents</span>
        <span class="empty-state-hint">Agents are spawned via Discord commands or the chat interface</span>
      </div>

      <!-- Agent cards -->
      <div v-else class="ag-card-grid" role="list" aria-label="Agent list">
        <div v-for="agent in filteredAgents" :key="agent.id"
             class="ag-card" :class="'ag-card-' + agent.status" role="listitem">
          <!-- Card header -->
          <div class="ag-card-header">
            <div class="ag-card-title-row">
              <span class="ag-status-dot" :class="'ag-dot-' + agent.status" role="img" :aria-label="'Status: ' + agent.status"></span>
              <span class="ag-card-label">{{ agent.label }}</span>
              <span class="ag-card-id">{{ agent.id }}</span>
            </div>
            <span class="ag-status-badge" :class="'ag-badge-' + agent.status">{{ agent.status }}</span>
          </div>

          <!-- Goal -->
          <div class="ag-card-goal">{{ agent.goal }}</div>

          <!-- Progress bar (running agents) -->
          <div v-if="agent.status === 'running'" class="ag-progress-bar">
            <div class="ag-progress-fill" :style="{ width: progressPercent(agent) + '%' }"></div>
          </div>

          <!-- Stats row -->
          <div class="ag-card-stats">
            <div class="ag-card-stat">
              <span class="ag-card-stat-label">Iterations</span>
              <span class="ag-card-stat-value">{{ agent.iteration_count }}</span>
            </div>
            <div class="ag-card-stat">
              <span class="ag-card-stat-label">Runtime</span>
              <span class="ag-card-stat-value">{{ formatRuntime(agent.runtime_seconds) }}</span>
            </div>
            <div class="ag-card-stat">
              <span class="ag-card-stat-label">Tools</span>
              <span class="ag-card-stat-value">{{ (agent.tools_used || []).length }}</span>
            </div>
          </div>

          <!-- Tools used -->
          <div v-if="agent.tools_used && agent.tools_used.length > 0" class="ag-card-tools">
            <span v-for="tool in agent.tools_used" :key="tool" class="ag-tool-chip">{{ tool }}</span>
          </div>

          <!-- Requester -->
          <div class="ag-card-meta">
            <span v-if="agent.requester_name" class="text-gray-500 text-xs">
              by {{ agent.requester_name }}
            </span>
            <span v-if="agent.created_at" class="text-gray-600 text-xs">
              {{ formatDate(agent.created_at) }}
            </span>
          </div>

          <!-- Result / error (terminal states) -->
          <div v-if="agent.result && agent.status !== 'running'" class="ag-card-result">
            <div class="ag-result-label">Result</div>
            <div class="ag-result-text">{{ agent.result }}</div>
          </div>
          <div v-if="agent.error" class="ag-card-error">
            <div class="ag-result-label">Error</div>
            <div class="ag-result-text text-red-400">{{ agent.error }}</div>
          </div>

          <!-- Kill button (running only) -->
          <div v-if="agent.status === 'running'" class="ag-card-actions">
            <button @click="killAgent(agent.id)" class="btn btn-danger text-xs"
                    :disabled="killing === agent.id">
              {{ killing === agent.id ? 'Killing...' : 'Kill Agent' }}
            </button>
          </div>
        </div>
      </div>
    </div>`,

  setup() {
    const agents = ref([]);
    const loading = ref(true);
    const error = ref(null);
    const killing = ref(null);
    const autoRefresh = ref(true);
    const statusFilter = ref('all');
    let refreshInterval = null;

    const runningCount = computed(() => agents.value.filter(a => a.status === 'running').length);
    const completedCount = computed(() => agents.value.filter(a => a.status === 'completed').length);
    const failedCount = computed(() => agents.value.filter(a => ['failed', 'timeout', 'killed'].includes(a.status)).length);

    const statusFilters = computed(() => [
      { value: 'all', label: 'All', count: agents.value.length },
      { value: 'running', label: 'Running', count: runningCount.value },
      { value: 'completed', label: 'Completed', count: completedCount.value },
      { value: 'failed', label: 'Failed', count: failedCount.value },
    ]);

    const filteredAgents = computed(() => {
      if (statusFilter.value === 'all') return agents.value;
      if (statusFilter.value === 'failed') {
        return agents.value.filter(a => ['failed', 'timeout', 'killed'].includes(a.status));
      }
      return agents.value.filter(a => a.status === statusFilter.value);
    });

    function formatRuntime(seconds) {
      if (!seconds && seconds !== 0) return '—';
      if (seconds < 60) return `${Math.round(seconds)}s`;
      const mins = Math.floor(seconds / 60);
      const secs = Math.round(seconds % 60);
      return `${mins}m ${secs}s`;
    }

    function formatDate(ts) {
      if (!ts) return '';
      try {
        const d = new Date(ts * 1000);
        return d.toLocaleString();
      } catch { return ''; }
    }

    function progressPercent(agent) {
      const max = 30;
      return Math.min(100, Math.round((agent.iteration_count / max) * 100));
    }

    async function fetchAgents() {
      loading.value = true;
      error.value = null;
      try {
        const data = await api.get('/api/agents');
        agents.value = Array.isArray(data) ? data : [];
      } catch (e) {
        error.value = e.message;
      }
      loading.value = false;
    }

    async function killAgent(agentId) {
      killing.value = agentId;
      try {
        await api.del(`/api/agents/${encodeURIComponent(agentId)}`);
        await fetchAgents();
      } catch { /* ignore */ }
      killing.value = null;
    }

    function startAutoRefresh() {
      stopAutoRefresh();
      if (autoRefresh.value) {
        refreshInterval = setInterval(() => {
          if (autoRefresh.value && runningCount.value > 0) {
            fetchAgents();
          }
        }, 5000);
      }
    }

    function stopAutoRefresh() {
      if (refreshInterval) {
        clearInterval(refreshInterval);
        refreshInterval = null;
      }
    }

    onMounted(() => {
      fetchAgents();
      startAutoRefresh();
    });

    onUnmounted(() => {
      stopAutoRefresh();
    });

    return {
      agents, loading, error, killing, autoRefresh, statusFilter,
      runningCount, completedCount, failedCount,
      statusFilters, filteredAgents,
      formatRuntime, formatDate, progressPercent,
      fetchAgents, killAgent, startAutoRefresh, stopAutoRefresh,
    };
  },
};
