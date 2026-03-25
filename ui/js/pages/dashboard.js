/**
 * Loki Management UI — Dashboard Page
 * Bot status, guilds, quick stats, recent activity feed
 */
import { api, ws } from '../api.js';

const { ref, computed, onMounted, onUnmounted } = Vue;

export default {
  template: `
    <div class="p-6">
      <h1 class="text-xl font-semibold mb-4">Dashboard</h1>

      <div v-if="loading" class="flex items-center gap-2 text-gray-400">
        <div class="spinner"></div> Loading...
      </div>
      <div v-else-if="error" class="loki-card border-red-900">
        <p class="text-red-400">{{ error }}</p>
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
              <div class="text-gray-400 text-sm font-medium">Recent Activity</div>
              <button @click="fetchActivity" class="btn btn-ghost text-xs" :disabled="activityLoading">
                {{ activityLoading ? 'Loading...' : 'Refresh' }}
              </button>
            </div>
            <div v-if="activityLoading && activity.length === 0" class="text-gray-500 text-sm">Loading...</div>
            <div v-else-if="activity.length === 0" class="text-gray-500 text-sm">No recent activity</div>
            <table v-else class="loki-table">
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
                <tr v-for="(a, i) in activity" :key="i">
                  <td class="text-gray-400 text-xs font-mono whitespace-nowrap">{{ formatTime(a.timestamp) }}</td>
                  <td class="font-mono text-sm">{{ a.tool_name }}</td>
                  <td class="text-gray-400 text-sm">{{ a.user_name || a.user_id || '—' }}</td>
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
    </div>`,

  setup() {
    const status = ref({});
    const loading = ref(true);
    const error = ref(null);
    const activity = ref([]);
    const activityLoading = ref(false);

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
      if (!ts) return '—';
      try {
        const d = new Date(ts);
        return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
      } catch { return ts; }
    }

    async function fetchStatus() {
      try {
        status.value = await api.get('/api/status');
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
      } catch { /* ignore */ }
      activityLoading.value = false;
    }

    // Auto-refresh status every 15s
    let interval = null;

    function onEvent(data) {
      // Push new events to the top of the activity feed
      if (data.payload && data.payload.tool_name) {
        activity.value.unshift(data.payload);
        if (activity.value.length > 10) activity.value.pop();
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

    return { status, loading, error, uptime, stats, activity, activityLoading, fetchActivity, formatTime };
  },
};
