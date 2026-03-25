/**
 * Loki Management UI — Processes Page
 * View/kill background processes, poll output
 */
import { api, ws } from '../api.js';

const { ref, computed, onMounted, onUnmounted } = Vue;

export default {
  template: `
    <div class="p-6">
      <div class="flex items-center justify-between mb-4">
        <h1 class="text-xl font-semibold">Processes</h1>
        <button @click="fetchProcesses" class="btn btn-ghost text-xs" :disabled="loading">
          {{ loading ? 'Loading...' : 'Refresh' }}
        </button>
      </div>

      <div v-if="loading && processes.length === 0" class="flex items-center gap-2 text-gray-400">
        <div class="spinner"></div> Loading processes...
      </div>
      <div v-else-if="error" class="loki-card border-red-900">
        <p class="text-red-400">{{ error }}</p>
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

        <!-- Process table -->
        <table class="loki-table">
          <thead>
            <tr>
              <th>PID</th>
              <th>Command</th>
              <th>Host</th>
              <th>Status</th>
              <th>Uptime</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="p in processes" :key="p.pid">
              <td class="font-mono text-sm">{{ p.pid }}</td>
              <td class="text-sm font-mono text-gray-300" :title="p.command">{{ truncate(p.command, 60) }}</td>
              <td class="text-sm text-gray-400">{{ p.host || 'local' }}</td>
              <td>
                <span class="badge" :class="statusBadge(p.status)">{{ p.status }}</span>
                <span v-if="p.exit_code !== null && p.exit_code !== undefined"
                      class="text-xs text-gray-500 ml-1">({{ p.exit_code }})</span>
              </td>
              <td class="text-sm text-gray-400">{{ formatUptime(p.uptime_seconds) }}</td>
              <td class="whitespace-nowrap">
                <button v-if="p.status === 'running'"
                        @click="confirmKill(p.pid)"
                        class="btn btn-danger text-xs">Kill</button>
                <span v-else class="text-gray-600 text-xs">-</span>
              </td>
            </tr>
          </tbody>
        </table>
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

    // Kill
    const killTarget = ref(null);
    const killing = ref(false);

    const runningCount = computed(() => processes.value.filter(p => p.status === 'running').length);
    const completedCount = computed(() => processes.value.filter(p => p.status !== 'running').length);

    function truncate(text, max) {
      if (!text) return '';
      return text.length > max ? text.slice(0, max) + '...' : text;
    }

    function statusBadge(status) {
      if (status === 'running') return 'badge-success';
      if (status === 'completed' || status === 'exited') return 'badge-info';
      if (status === 'killed' || status === 'error') return 'badge-danger';
      return 'badge-warning';
    }

    function formatUptime(seconds) {
      if (seconds == null) return '-';
      const s = Math.round(seconds);
      if (s < 60) return `${s}s`;
      if (s < 3600) return `${Math.floor(s / 60)}m ${s % 60}s`;
      const h = Math.floor(s / 3600);
      const m = Math.floor((s % 3600) / 60);
      return `${h}h ${m}m`;
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
    });

    onUnmounted(() => {
      ws.unsubscribe('events', onEvent);
    });

    return {
      processes, loading, error,
      killTarget, killing,
      runningCount, completedCount,
      truncate, statusBadge, formatUptime,
      fetchProcesses, confirmKill, doKill,
    };
  },
};
