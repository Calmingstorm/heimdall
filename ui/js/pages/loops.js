/**
 * Heimdall Management UI — Loops Page
 * View/start/stop autonomous loops, view iteration history
 */
import { api, ws } from '../api.js';

const { ref, computed, onMounted, onUnmounted } = Vue;

export default {
  template: `
    <div class="p-6 page-fade-in">
      <div class="flex items-center justify-between mb-4 flex-wrap gap-2">
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
      <div v-if="showCreate" class="hm-card mb-4">
        <h2 class="text-sm font-medium mb-3">Start New Loop</h2>

        <div class="mb-3">
          <label class="text-gray-400 text-xs block mb-1">Goal</label>
          <textarea v-model="form.goal" class="hm-input" rows="3"
                    placeholder="What should this loop accomplish? e.g. Monitor disk usage and warn if above 80%"></textarea>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-3 gap-3 mb-3">
          <div>
            <label class="text-gray-400 text-xs block mb-1">Interval (seconds)</label>
            <input v-model.number="form.interval_seconds" type="number" class="hm-input"
                   min="10" placeholder="60" />
          </div>
          <div>
            <label class="text-gray-400 text-xs block mb-1">Mode</label>
            <select v-model="form.mode" class="hm-input">
              <option value="notify">Notify (check + report)</option>
              <option value="act">Act (check + take actions + report)</option>
              <option value="silent">Silent (only report if notable)</option>
            </select>
          </div>
          <div>
            <label class="text-gray-400 text-xs block mb-1">Max Iterations</label>
            <input v-model.number="form.max_iterations" type="number" class="hm-input"
                   min="1" placeholder="50" />
          </div>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-3 mb-3">
          <div>
            <label class="text-gray-400 text-xs block mb-1">Stop Condition (optional)</label>
            <input v-model="form.stop_condition" type="text" class="hm-input"
                   placeholder="e.g. when disk is below 50%" />
          </div>
          <div>
            <label class="text-gray-400 text-xs block mb-1">Channel ID</label>
            <input v-model="form.channel_id" type="text" class="hm-input"
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
      <div v-if="loading && loops.length === 0" class="space-y-2">
        <div v-for="n in 3" :key="n" class="skeleton skeleton-row"></div>
      </div>
      <div v-else-if="error" class="hm-card border-red-900 error-state" role="alert">
        <span class="error-icon" aria-hidden="true">\u26A0</span>
        <p class="text-red-400">{{ error }}</p>
        <button @click="fetchLoops" class="btn btn-ghost text-xs">Retry</button>
      </div>
      <div v-else-if="loops.length === 0 && !showCreate" class="hm-card empty-state">
        <span class="empty-state-icon">\u{1F504}</span>
        <span class="empty-state-text">No active loops</span>
        <span class="empty-state-hint">Click "Start Loop" to create an autonomous recurring task</span>
      </div>
      <div v-else-if="loops.length > 0">
        <!-- Summary -->
        <div class="grid grid-cols-2 md:grid-cols-3 gap-3 mb-4">
          <div class="hm-card text-center">
            <div class="text-2xl font-bold">{{ loops.length }}</div>
            <div class="text-gray-400 text-xs">Total Loops</div>
          </div>
          <div class="hm-card text-center">
            <div class="text-2xl font-bold text-green-400">{{ runningCount }}</div>
            <div class="text-gray-400 text-xs">Running</div>
          </div>
          <div class="hm-card text-center">
            <div class="text-2xl font-bold">{{ totalIterations }}</div>
            <div class="text-gray-400 text-xs">Total Iterations</div>
          </div>
        </div>

        <!-- Loop cards -->
        <div class="space-y-3">
          <div v-for="loop in loops" :key="loop.id" class="hm-card">
            <div class="flex items-start justify-between mb-2">
              <div class="flex items-center gap-2">
                <span class="loop-status-dot" :class="statusDotClass(loop.status)"></span>
                <span class="badge" :class="statusBadge(loop.status)">{{ loop.status || 'running' }}</span>
                <span class="badge" :class="modeBadge(loop.mode)">{{ loop.mode }}</span>
                <span class="font-mono text-xs text-gray-500">{{ loop.id }}</span>
              </div>
              <div class="flex gap-2">
                <button @click="doRestart(loop.id)" class="btn btn-ghost text-xs"
                        :disabled="restartingId === loop.id"
                        title="Restart loop with same config">
                  {{ restartingId === loop.id ? 'Restarting...' : 'Restart' }}
                </button>
                <button v-if="loop.status === 'running'"
                        @click="confirmStop(loop.id)" class="btn btn-danger text-xs">Stop</button>
              </div>
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
                <div class="mt-1 w-full bg-gray-800 rounded-full h-1">
                  <div class="bg-indigo-500 h-1 rounded-full transition-all duration-300"
                       :style="{ width: Math.min(100, (loop.iteration_count / loop.max_iterations) * 100) + '%' }"></div>
                </div>
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

            <!-- Iteration history -->
            <div v-if="loop.iteration_history && loop.iteration_history.length > 0" class="mt-3">
              <button @click="toggleHistory(loop.id)" class="text-xs text-gray-500 hover:text-gray-300 flex items-center gap-1 mb-1">
                <span class="tool-expand-icon" :style="{ transform: expandedHistory[loop.id] ? 'rotate(90deg)' : '' }">&#9654;</span>
                Recent iterations ({{ loop.iteration_history.length }})
              </button>
              <div v-if="expandedHistory[loop.id]" class="loop-history">
                <div v-for="(entry, i) in loop.iteration_history" :key="i"
                     class="loop-history-entry">
                  {{ entry }}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Stop confirmation -->
      <div v-if="stopTarget" class="modal-overlay" @click.self="stopTarget = null" role="dialog" aria-modal="true" aria-labelledby="loop-stop-title">
        <div class="modal-content">
          <h3 id="loop-stop-title" class="text-lg font-semibold mb-2">Stop Loop</h3>
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

    // Restart
    const restartingId = ref(null);

    // History expansion
    const expandedHistory = ref({});

    const totalIterations = computed(() =>
      loops.value.reduce((sum, l) => sum + (l.iteration_count || 0), 0)
    );

    const runningCount = computed(() =>
      loops.value.filter(l => l.status === 'running').length
    );

    function statusDotClass(status) {
      if (status === 'running') return 'loop-status-running';
      if (status === 'error') return 'loop-status-error';
      return 'loop-status-stopped';
    }

    function statusBadge(status) {
      if (status === 'running') return 'badge-success';
      if (status === 'error') return 'badge-danger';
      if (status === 'completed') return 'badge-info';
      return 'badge-warning';
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

    function toggleHistory(loopId) {
      expandedHistory.value = {
        ...expandedHistory.value,
        [loopId]: !expandedHistory.value[loopId],
      };
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

    async function doRestart(loopId) {
      restartingId.value = loopId;
      try {
        await api.post(`/api/loops/${encodeURIComponent(loopId)}/restart`);
        await fetchLoops();
      } catch { /* ignore */ }
      restartingId.value = null;
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
      restartingId, expandedHistory,
      totalIterations, runningCount,
      statusDotClass, statusBadge, modeBadge,
      formatInterval, formatAge, toggleHistory,
      fetchLoops, doCreate, confirmStop, doStop, doRestart,
    };
  },
};
