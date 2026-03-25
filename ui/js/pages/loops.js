/**
 * Loki Management UI — Loops Page
 * View/start/stop autonomous loops, view iteration history
 */
import { api, ws } from '../api.js';

const { ref, computed, onMounted, onUnmounted } = Vue;

export default {
  template: `
    <div class="p-6">
      <div class="flex items-center justify-between mb-4">
        <h1 class="text-xl font-semibold">Autonomous Loops</h1>
        <div class="flex gap-2">
          <button @click="showCreate = !showCreate" class="btn btn-primary text-xs">
            {{ showCreate ? 'Cancel' : 'Start Loop' }}
          </button>
          <button @click="fetchLoops" class="btn btn-ghost text-xs" :disabled="loading">
            {{ loading ? 'Loading...' : 'Refresh' }}
          </button>
        </div>
      </div>

      <!-- Create form -->
      <div v-if="showCreate" class="loki-card mb-4">
        <h2 class="text-sm font-medium mb-3">Start New Loop</h2>

        <div class="mb-3">
          <label class="text-gray-400 text-xs block mb-1">Goal</label>
          <textarea v-model="form.goal" class="loki-input" rows="3"
                    placeholder="What should this loop accomplish? e.g. Monitor disk usage and warn if above 80%"></textarea>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-3 gap-3 mb-3">
          <div>
            <label class="text-gray-400 text-xs block mb-1">Interval (seconds)</label>
            <input v-model.number="form.interval_seconds" type="number" class="loki-input"
                   min="10" placeholder="60" />
          </div>
          <div>
            <label class="text-gray-400 text-xs block mb-1">Mode</label>
            <select v-model="form.mode" class="loki-input">
              <option value="notify">Notify (check + report)</option>
              <option value="act">Act (check + take actions + report)</option>
              <option value="silent">Silent (only report if notable)</option>
            </select>
          </div>
          <div>
            <label class="text-gray-400 text-xs block mb-1">Max Iterations</label>
            <input v-model.number="form.max_iterations" type="number" class="loki-input"
                   min="1" placeholder="50" />
          </div>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-3 mb-3">
          <div>
            <label class="text-gray-400 text-xs block mb-1">Stop Condition (optional)</label>
            <input v-model="form.stop_condition" type="text" class="loki-input"
                   placeholder="e.g. when disk is below 50%" />
          </div>
          <div>
            <label class="text-gray-400 text-xs block mb-1">Channel ID</label>
            <input v-model="form.channel_id" type="text" class="loki-input"
                   placeholder="Discord channel ID" />
          </div>
        </div>

        <div v-if="createError" class="mb-3 text-red-400 text-sm">{{ createError }}</div>
        <div v-if="createSuccess" class="mb-3 text-green-400 text-sm">{{ createSuccess }}</div>

        <button @click="doCreate" class="btn btn-primary text-xs" :disabled="creating">
          {{ creating ? 'Starting...' : 'Start Loop' }}
        </button>
      </div>

      <!-- Loop list -->
      <div v-if="loading && loops.length === 0" class="flex items-center gap-2 text-gray-400">
        <div class="spinner"></div> Loading loops...
      </div>
      <div v-else-if="error" class="loki-card border-red-900">
        <p class="text-red-400">{{ error }}</p>
      </div>
      <div v-else-if="loops.length === 0 && !showCreate" class="loki-card">
        <p class="text-gray-400">No active loops. Click "Start Loop" to create one.</p>
      </div>
      <div v-else-if="loops.length > 0">
        <!-- Summary -->
        <div class="grid grid-cols-2 md:grid-cols-3 gap-3 mb-4">
          <div class="loki-card text-center">
            <div class="text-2xl font-bold">{{ loops.length }}</div>
            <div class="text-gray-400 text-xs">Active Loops</div>
          </div>
          <div class="loki-card text-center">
            <div class="text-2xl font-bold">{{ totalIterations }}</div>
            <div class="text-gray-400 text-xs">Total Iterations</div>
          </div>
          <div class="loki-card text-center">
            <div class="text-2xl font-bold">{{ modeBreakdown }}</div>
            <div class="text-gray-400 text-xs">Modes</div>
          </div>
        </div>

        <!-- Loop cards -->
        <div class="space-y-3">
          <div v-for="loop in loops" :key="loop.id" class="loki-card">
            <div class="flex items-start justify-between mb-2">
              <div class="flex items-center gap-2">
                <span class="badge" :class="statusBadge(loop.status)">{{ loop.status || 'running' }}</span>
                <span class="badge" :class="modeBadge(loop.mode)">{{ loop.mode }}</span>
                <span class="font-mono text-xs text-gray-500">{{ loop.id }}</span>
              </div>
              <button @click="confirmStop(loop.id)" class="btn btn-danger text-xs">Stop</button>
            </div>

            <div class="text-sm text-gray-200 mb-2">{{ loop.goal }}</div>

            <div class="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs text-gray-400">
              <div>
                <span class="text-gray-500">Interval:</span>
                {{ formatInterval(loop.interval_seconds) }}
              </div>
              <div>
                <span class="text-gray-500">Iterations:</span>
                {{ loop.iteration_count }} / {{ loop.max_iterations }}
              </div>
              <div>
                <span class="text-gray-500">Last trigger:</span>
                {{ loop.last_trigger ? formatAge(loop.last_trigger) : 'pending' }}
              </div>
              <div>
                <span class="text-gray-500">Created:</span>
                {{ formatAge(loop.created_at) }}
              </div>
            </div>

            <div v-if="loop.stop_condition" class="mt-2 text-xs text-gray-500">
              <span class="text-gray-600">Stop when:</span> {{ loop.stop_condition }}
            </div>

            <div v-if="loop.requester_name" class="mt-1 text-xs text-gray-600">
              Started by {{ loop.requester_name }}
            </div>
          </div>
        </div>
      </div>

      <!-- Stop confirmation -->
      <div v-if="stopTarget" class="modal-overlay" @click.self="stopTarget = null">
        <div class="modal-content">
          <h3 class="text-lg font-semibold mb-2">Stop Loop</h3>
          <p class="text-gray-400 text-sm mb-4">
            Stop loop <span class="font-mono font-semibold text-gray-200">{{ stopTarget }}</span>?
            The current iteration will finish before stopping.
          </p>
          <div class="flex gap-2 justify-end">
            <button @click="stopTarget = null" class="btn btn-ghost">Cancel</button>
            <button @click="doStop" class="btn btn-danger" :disabled="stopping">
              {{ stopping ? 'Stopping...' : 'Stop Loop' }}
            </button>
          </div>
        </div>
      </div>
    </div>`,

  setup() {
    const loops = ref([]);
    const loading = ref(true);
    const error = ref(null);

    // Create form
    const showCreate = ref(false);
    const form = ref({
      goal: '',
      interval_seconds: 60,
      mode: 'notify',
      max_iterations: 50,
      stop_condition: '',
      channel_id: '',
    });
    const creating = ref(false);
    const createError = ref(null);
    const createSuccess = ref(null);

    // Stop
    const stopTarget = ref(null);
    const stopping = ref(false);

    const totalIterations = computed(() =>
      loops.value.reduce((sum, l) => sum + (l.iteration_count || 0), 0)
    );

    const modeBreakdown = computed(() => {
      const modes = {};
      for (const l of loops.value) {
        modes[l.mode] = (modes[l.mode] || 0) + 1;
      }
      return Object.entries(modes).map(([m, c]) => `${c} ${m}`).join(', ') || '-';
    });

    function statusBadge(status) {
      if (status === 'running') return 'badge-success';
      if (status === 'stopped') return 'badge-danger';
      return 'badge-info';
    }

    function modeBadge(mode) {
      if (mode === 'act') return 'badge-warning';
      if (mode === 'silent') return 'badge-info';
      return 'badge-success';
    }

    function formatInterval(seconds) {
      if (!seconds) return '-';
      if (seconds < 60) return `${seconds}s`;
      if (seconds < 3600) return `${Math.floor(seconds / 60)}m`;
      return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
    }

    function formatAge(ts) {
      if (!ts) return '-';
      const now = Date.now() / 1000;
      const t = typeof ts === 'number' ? ts : new Date(ts).getTime() / 1000;
      const diff = now - t;
      if (diff < 60) return 'just now';
      if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
      if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
      return `${Math.floor(diff / 86400)}d ago`;
    }

    async function fetchLoops() {
      loading.value = true;
      error.value = null;
      try {
        loops.value = await api.get('/api/loops');
      } catch (e) {
        error.value = e.message;
      }
      loading.value = false;
    }

    async function doCreate() {
      createError.value = null;
      createSuccess.value = null;
      const f = form.value;
      if (!f.goal.trim()) { createError.value = 'Goal is required'; return; }
      if (!f.channel_id.trim()) { createError.value = 'Channel ID is required'; return; }

      const payload = {
        goal: f.goal.trim(),
        channel_id: f.channel_id.trim(),
        interval_seconds: f.interval_seconds || 60,
        mode: f.mode,
        max_iterations: f.max_iterations || 50,
      };
      if (f.stop_condition.trim()) payload.stop_condition = f.stop_condition.trim();

      creating.value = true;
      try {
        const result = await api.post('/api/loops', payload);
        createSuccess.value = `Loop started: ${result.loop_id}`;
        form.value = {
          goal: '', interval_seconds: 60, mode: 'notify',
          max_iterations: 50, stop_condition: '', channel_id: '',
        };
        await fetchLoops();
        setTimeout(() => { showCreate.value = false; createSuccess.value = null; }, 800);
      } catch (e) {
        createError.value = e.message;
      }
      creating.value = false;
    }

    function confirmStop(id) {
      stopTarget.value = id;
    }

    async function doStop() {
      if (!stopTarget.value) return;
      stopping.value = true;
      try {
        await api.del(`/api/loops/${encodeURIComponent(stopTarget.value)}`);
        await fetchLoops();
      } catch { /* ignore */ }
      stopping.value = false;
      stopTarget.value = null;
    }

    // WebSocket: refresh on loop events
    function onEvent(data) {
      if (data.payload && (data.payload.loop_id || data.payload.type === 'loop')) {
        fetchLoops();
      }
    }

    onMounted(() => {
      fetchLoops();
      ws.subscribe('events', onEvent);
    });

    onUnmounted(() => {
      ws.unsubscribe('events', onEvent);
    });

    return {
      loops, loading, error,
      showCreate, form, creating, createError, createSuccess,
      stopTarget, stopping,
      totalIterations, modeBreakdown,
      statusBadge, modeBadge, formatInterval, formatAge,
      fetchLoops, doCreate, confirmStop, doStop,
    };
  },
};
