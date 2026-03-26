/**
 * Loki Management UI — Processes Page
 * View/kill background processes, poll output, auto-refresh
 */
import { api, ws } from '../api.js';

const { ref, computed, onMounted, onUnmounted, watch } = Vue;

export default {
  template: `
    <div class="p-6">
      <div class="flex items-center justify-between mb-4">
        <h1 class="text-xl font-semibold">Processes</h1>
        <div class="flex items-center gap-3">
          <label class="flex items-center gap-2 text-xs text-gray-400 cursor-pointer">
            <span class="toggle-switch" style="width:28px; height:16px;">
              <input type="checkbox" v-model="autoRefresh" />
              <span class="toggle-slider" style="border-radius:8px;">
                <span style="width:10px; height:10px; left:3px; bottom:3px;"></span>
              </span>
            </span>
            Auto-refresh
            <span v-if="autoRefresh" class="text-green-400">(5s)</span>
          </label>
          <button @click="fetchProcesses" class="btn btn-ghost text-xs" :disabled="loading">
            {{ loading ? 'Loading...' : 'Refresh' }}
          </button>
        </div>
      </div>

      <div v-if="loading && processes.length === 0" class="space-y-2">
        <div v-for="n in 3" :key="n" class="skeleton skeleton-row"></div>
      </div>
      <div v-else-if="error" class="loki-card border-red-900 error-state">
        <p class="text-red-400">{{ error }}</p>
        <button @click="fetchProcesses" class="btn btn-ghost text-xs">Retry</button>
      </div>
      <div v-else-if="processes.length === 0" class="loki-card">
        <p class="text-gray-400">No background processes running.</p>
      </div>
      <div v-else>
        <!-- Summary -->
        <div class="grid grid-cols-2 md:grid-cols-3 gap-3 mb-4">
          <div class="loki-card text-center">
            <div class="text-2xl font-bold">{{ processes.length }}</div>
            <div class="text-gray-400 text-xs">Total</div>
          </div>
          <div class="loki-card text-center">
            <div class="text-2xl font-bold" :class="runningCount > 0 ? 'text-green-400' : ''">{{ runningCount }}</div>
            <div class="text-gray-400 text-xs">Running</div>
          </div>
          <div class="loki-card text-center">
            <div class="text-2xl font-bold">{{ completedCount }}</div>
            <div class="text-gray-400 text-xs">Completed</div>
          </div>
        </div>

        <!-- Process cards -->
        <div class="space-y-3">
          <div v-for="p in processes" :key="p.pid" class="loki-card">
            <div class="flex items-start justify-between mb-2">
              <div class="flex items-center gap-2">
                <span class="loop-status-dot" :class="procStatusDot(p.status)"></span>
                <span class="font-mono text-sm font-semibold">PID {{ p.pid }}</span>
                <span class="badge" :class="statusBadge(p.status)">{{ p.status }}</span>
                <span v-if="p.exit_code !== null && p.exit_code !== undefined"
                      class="text-xs text-gray-500">(exit {{ p.exit_code }})</span>
              </div>
              <div class="flex items-center gap-2">
                <span class="text-xs text-gray-500">{{ formatUptime(p.uptime_seconds) }}</span>
                <button v-if="p.status === 'running'"
                        @click="confirmKill(p.pid)"
                        class="btn btn-danger text-xs">Kill</button>
              </div>
            </div>

            <div class="text-sm font-mono text-gray-300 mb-2" :title="p.command">
              {{ p.command }}
            </div>

            <div class="text-xs text-gray-500 mb-1">
              <span class="text-gray-600">Host:</span> {{ p.host || 'local' }}
            </div>

            <!-- Output preview (last 3 lines) -->
            <div v-if="p.output_preview && p.output_preview.length > 0" class="mt-2">
              <div class="text-xs text-gray-600 mb-1">Recent output:</div>
              <pre class="process-output-preview">{{ p.output_preview.join('\\n') }}</pre>
            </div>
          </div>
        </div>
      </div>

      <!-- Kill confirmation -->
      <div v-if="killTarget !== null" class="modal-overlay" @click.self="killTarget = null">
        <div class="modal-content">
          <h3 class="text-lg font-semibold mb-2">Kill Process</h3>
          <p class="text-gray-400 text-sm mb-4">
            Kill process <span class="font-mono font-semibold text-gray-200">{{ killTarget }}</span>?
          </p>
          <div class="flex gap-2 justify-end">
            <button @click="killTarget = null" class="btn btn-ghost">Cancel</button>
            <button @click="doKill" class="btn btn-danger" :disabled="killing">
              {{ killing ? 'Killing...' : 'Kill' }}
            </button>
          </div>
        </div>
      </div>
    </div>`,

  setup() {
    const processes = ref([]);
    const loading = ref(true);
    const error = ref(null);
    const autoRefresh = ref(true);
    let refreshTimer = null;

    // Kill
    const killTarget = ref(null);
    const killing = ref(false);

    const runningCount = computed(() => processes.value.filter(p => p.status === 'running').length);
    const completedCount = computed(() => processes.value.filter(p => p.status !== 'running').length);

    function procStatusDot(status) {
      if (status === 'running') return 'loop-status-running';
      if (status === 'failed' || status === 'error') return 'loop-status-error';
      return 'loop-status-stopped';
    }

    function statusBadge(status) {
      if (status === 'running') return 'badge-success';
      if (status === 'completed' || status === 'exited') return 'badge-info';
      if (status === 'killed' || status === 'error' || status === 'failed') return 'badge-danger';
      return 'badge-warning';
    }

    function formatUptime(seconds) {
      if (seconds == null) return '-';
      const s = Math.round(seconds);
      if (s < 60) return `${s} seconds`;
      if (s < 3600) {
        const m = Math.floor(s / 60);
        const rem = s % 60;
        return rem > 0 ? `${m} min, ${rem}s` : `${m} min`;
      }
      const h = Math.floor(s / 3600);
      const m = Math.floor((s % 3600) / 60);
      return m > 0 ? `${h} hour${h !== 1 ? 's' : ''}, ${m} min` : `${h} hour${h !== 1 ? 's' : ''}`;
    }

    async function fetchProcesses() {
      loading.value = true;
      error.value = null;
      try {
        processes.value = await api.get('/api/processes');
      } catch (e) {
        error.value = e.message;
      }
      loading.value = false;
    }

    function startAutoRefresh() {
      stopAutoRefresh();
      if (autoRefresh.value) {
        refreshTimer = setInterval(() => {
          if (!loading.value) fetchProcesses();
        }, 5000);
      }
    }

    function stopAutoRefresh() {
      if (refreshTimer) {
        clearInterval(refreshTimer);
        refreshTimer = null;
      }
    }

    // Watch autoRefresh toggle
    watch(autoRefresh, (val) => {
      if (val) startAutoRefresh();
      else stopAutoRefresh();
    });

    function confirmKill(pid) {
      killTarget.value = pid;
    }

    async function doKill() {
      if (killTarget.value === null) return;
      killing.value = true;
      try {
        await api.del(`/api/processes/${killTarget.value}`);
        await fetchProcesses();
      } catch { /* ignore */ }
      killing.value = false;
      killTarget.value = null;
    }

    // WebSocket: refresh on process events
    function onEvent(data) {
      if (data.payload && (data.payload.pid || data.payload.type === 'process')) {
        fetchProcesses();
      }
    }

    onMounted(() => {
      fetchProcesses();
      ws.subscribe('events', onEvent);
      startAutoRefresh();
    });

    onUnmounted(() => {
      ws.unsubscribe('events', onEvent);
      stopAutoRefresh();
    });

    return {
      processes, loading, error, autoRefresh,
      killTarget, killing,
      runningCount, completedCount,
      procStatusDot, statusBadge, formatUptime,
      fetchProcesses, confirmKill, doKill,
    };
  },
};
