/**
 * Loki Management UI — Dashboard Page
 * Bot status, guilds, quick stats, recent activity feed with live WebSocket updates
 */
import { api, ws } from '../api.js';

const { ref, computed, onMounted, onUnmounted, nextTick } = Vue;

export default {
  template: `
    <div class="p-6">
      <h1 class="text-xl font-semibold mb-4">Dashboard</h1>

      <!-- Skeleton loading -->
      <div v-if="loading" class="space-y-4">
        <div class="loki-card flex items-center gap-3">
          <div class="skeleton" style="width:12px;height:12px;border-radius:50%;flex-shrink:0;"></div>
          <div><div class="skeleton skeleton-text" style="width:120px;"></div><div class="skeleton skeleton-text" style="width:80px;margin-bottom:0;"></div></div>
        </div>
        <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
          <div v-for="n in 6" :key="n" class="loki-card text-center">
            <div class="skeleton skeleton-stat"></div>
            <div class="skeleton skeleton-text" style="width:60%;margin:0.25rem auto 0;"></div>
          </div>
        </div>
      </div>

      <!-- Error state with retry -->
      <div v-else-if="error" class="loki-card border-red-900 error-state">
        <p class="text-red-400">{{ error }}</p>
        <button @click="retry" class="btn btn-ghost text-xs">Retry</button>
      </div>

      <div v-else>
        <!-- Bot status banner -->
        <div class="loki-card mb-4 flex items-center gap-3">
          <span class="status-dot" :class="status.status === 'online' ? 'online' : 'starting'" style="width:12px;height:12px;"></span>
          <div>
            <div class="font-semibold">Loki</div>
            <div class="text-gray-400 text-sm">{{ status.status === 'online' ? 'Online' : 'Starting' }} &middot; {{ uptime }}</div>
          </div>
        </div>

        <!-- Quick stats -->
        <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3 mb-6">
          <div class="loki-card text-center" v-for="s in stats" :key="s.label">
            <div class="text-2xl font-bold" :class="s.color || ''">{{ s.value }}</div>
            <div class="text-gray-400 text-xs mt-0.5">{{ s.label }}</div>
          </div>
        </div>

        <!-- Two-column: Guilds + Recent Activity -->
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <!-- Guilds -->
          <div class="loki-card">
            <div class="text-gray-400 text-sm font-medium mb-2">Connected Guilds</div>
            <div v-if="!status.guilds || status.guilds.length === 0" class="text-gray-500 text-sm">No guilds connected</div>
            <div v-else class="space-y-1.5">
              <div v-for="g in status.guilds" :key="g.id" class="flex items-center gap-2 text-sm">
                <span class="status-dot online" style="width:6px;height:6px;"></span>
                <span>{{ g.name }}</span>
                <span class="text-gray-500 text-xs ml-auto font-mono">{{ g.id }}</span>
              </div>
            </div>
          </div>

          <!-- Recent Activity -->
          <div class="loki-card lg:col-span-2">
            <div class="flex items-center justify-between mb-2">
              <div class="text-gray-400 text-sm font-medium">
                Recent Activity
                <span v-if="newEventCount > 0" class="badge badge-success ml-1" style="font-size:0.625rem;">+{{ newEventCount }}</span>
              </div>
              <button @click="fetchActivity" class="btn btn-ghost text-xs" :disabled="activityLoading">
                {{ activityLoading ? 'Loading...' : 'Refresh' }}
              </button>
            </div>
            <div v-if="activityLoading && activity.length === 0" class="text-gray-500 text-sm">Loading...</div>
            <div v-else-if="activity.length === 0" class="text-gray-500 text-sm">No recent activity</div>
            <div v-else class="table-responsive">
              <table class="loki-table">
                <thead>
                  <tr>
                    <th>Time</th>
                    <th>Tool</th>
                    <th>User</th>
                    <th>Duration</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="(a, i) in activity" :key="a._key || i" :class="{ 'flash-new': a._isNew }">
                    <td class="text-gray-400 text-xs font-mono whitespace-nowrap">{{ formatTime(a.timestamp) }}</td>
                    <td class="font-mono text-sm">{{ a.tool_name }}</td>
                    <td class="text-gray-400 text-sm">{{ a.user_name || a.user_id || '\u2014' }}</td>
                    <td class="text-gray-400 text-xs font-mono">{{ a.execution_time_ms }}ms</td>
                    <td>
                      <span v-if="a.error" class="badge badge-danger">error</span>
                      <span v-else class="badge badge-success">ok</span>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </div>`,

  setup() {
    const status = ref({});
    const loading = ref(true);
    const error = ref(null);
    const activity = ref([]);
    const activityLoading = ref(false);
    const newEventCount = ref(0);
    let eventKeyCounter = 0;

    const uptime = computed(() => {
      const s = status.value.uptime_seconds || 0;
      const d = Math.floor(s / 86400);
      const h = Math.floor((s % 86400) / 3600);
      const m = Math.floor((s % 3600) / 60);
      if (d > 0) return `${d}d ${h}h ${m}m`;
      return `${h}h ${m}m`;
    });

    const stats = computed(() => [
      { label: 'Guilds', value: status.value.guild_count ?? 0 },
      { label: 'Tools', value: status.value.tool_count ?? 0 },
      { label: 'Skills', value: status.value.skill_count ?? 0 },
      { label: 'Sessions', value: status.value.session_count ?? 0 },
      { label: 'Loops', value: status.value.loop_count ?? 0, color: status.value.loop_count > 0 ? 'text-green-400' : '' },
      { label: 'Schedules', value: status.value.schedule_count ?? 0 },
    ]);

    function formatTime(ts) {
      if (!ts) return '\u2014';
      try {
        const d = new Date(ts);
        return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
      } catch { return ts; }
    }

    async function fetchStatus() {
      try {
        status.value = await api.get('/api/status');
        error.value = null;
      } catch (e) {
        error.value = e.message;
      } finally {
        loading.value = false;
      }
    }

    async function fetchActivity() {
      activityLoading.value = true;
      try {
        activity.value = await api.get('/api/audit?limit=10');
        newEventCount.value = 0;
      } catch { /* ignore */ }
      activityLoading.value = false;
    }

    function retry() {
      loading.value = true;
      error.value = null;
      fetchStatus();
      fetchActivity();
    }

    // Auto-refresh status every 15s
    let interval = null;

    function onEvent(data) {
      // Push new events to the top of the activity feed with flash animation
      if (data.payload && data.payload.tool_name) {
        const entry = { ...data.payload, _isNew: true, _key: ++eventKeyCounter };
        activity.value.unshift(entry);
        if (activity.value.length > 10) activity.value.pop();
        newEventCount.value++;
        // Remove flash class after animation
        setTimeout(() => { entry._isNew = false; }, 1500);
        // Reset new event counter after a while
        clearTimeout(onEvent._resetTimer);
        onEvent._resetTimer = setTimeout(() => { newEventCount.value = 0; }, 10000);
      }
    }

    onMounted(async () => {
      await Promise.all([fetchStatus(), fetchActivity()]);
      interval = setInterval(fetchStatus, 15000);
      ws.subscribe('events', onEvent);
    });

    onUnmounted(() => {
      if (interval) clearInterval(interval);
      ws.unsubscribe('events', onEvent);
    });

    return { status, loading, error, uptime, stats, activity, activityLoading, newEventCount, fetchActivity, fetchStatus, formatTime, retry };
  },
};
