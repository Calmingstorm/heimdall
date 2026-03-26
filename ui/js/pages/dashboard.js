/**
 * Loki Management UI — Dashboard Page
 * Bot status, guilds, quick stats, recent activity feed with live WebSocket updates
 */
import { api, ws } from '../api.js';

const { ref, computed, onMounted, onUnmounted, nextTick } = Vue;

export default {
  template: `
    <div class="p-6 page-fade-in">
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
        <span class="error-icon">\u26A0</span>
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
        <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3 mb-4">
          <div class="loki-card text-center stat-card" v-for="s in stats" :key="s.label">
            <div class="stat-icon" :class="s.iconColor">{{ s.icon }}</div>
            <div class="text-2xl font-bold" :class="s.color || ''">{{ s.value }}</div>
            <div class="text-gray-400 text-xs mt-0.5">{{ s.label }}</div>
          </div>
        </div>

        <!-- Quick actions -->
        <div class="loki-card mb-4">
          <div class="text-gray-400 text-sm font-medium mb-2">Quick Actions</div>
          <div class="flex flex-wrap gap-2">
            <button @click="reloadConfig" class="btn btn-ghost text-xs" :disabled="actionLoading.reload">
              {{ actionLoading.reload ? 'Reloading...' : '\u21bb Reload Config' }}
            </button>
            <button @click="clearSessions" class="btn btn-ghost text-xs" :disabled="actionLoading.clearSessions">
              {{ actionLoading.clearSessions ? 'Clearing...' : '\u2715 Clear All Sessions' }}
            </button>
            <button @click="stopAllLoops" class="btn btn-ghost text-xs" :disabled="actionLoading.stopLoops || (status.loop_count || 0) === 0">
              {{ actionLoading.stopLoops ? 'Stopping...' : '\u25a0 Stop All Loops' }}
            </button>
          </div>
          <div v-if="actionMessage" class="text-xs mt-2" :class="actionMessage.ok ? 'text-green-400' : 'text-red-400'">{{ actionMessage.text }}</div>
        </div>

        <!-- Three-column: Guilds + Recent Activity + Recent Errors -->
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <!-- Guilds -->
          <div class="loki-card">
            <div class="text-gray-400 text-sm font-medium mb-2">Connected Guilds</div>
            <div v-if="!status.guilds || status.guilds.length === 0" class="text-gray-500 text-sm text-center py-2">No guilds connected</div>
            <div v-else class="space-y-1.5">
              <div v-for="g in status.guilds" :key="g.id" class="flex items-center gap-2 text-sm">
                <span class="status-dot online" style="width:6px;height:6px;"></span>
                <span>{{ g.name }}</span>
                <span v-if="g.member_count" class="text-gray-500 text-xs ml-auto">{{ g.member_count }} members</span>
              </div>
            </div>
          </div>

          <!-- Recent Activity -->
          <div class="loki-card">
            <div class="flex items-center justify-between mb-2">
              <div class="text-gray-400 text-sm font-medium">
                Recent Activity
                <span v-if="newEventCount > 0" class="badge badge-success ml-1" style="font-size:0.625rem;">+{{ newEventCount }}</span>
              </div>
              <button @click="fetchActivity" class="btn btn-ghost text-xs" :disabled="activityLoading">
                {{ activityLoading ? '...' : '\u21bb' }}
              </button>
            </div>
            <div v-if="activityLoading && activity.length === 0" class="text-gray-500 text-sm">Loading...</div>
            <div v-else-if="activity.length === 0" class="text-gray-500 text-sm">No recent activity</div>
            <div v-else class="space-y-1">
              <div v-for="(a, i) in activity" :key="a._key || i"
                   class="flex items-center gap-2 text-xs py-1 border-b border-gray-800 last:border-0"
                   :class="{ 'flash-new': a._isNew }">
                <span v-if="a.error" class="text-red-400">\u2022</span>
                <span v-else class="text-green-400">\u2022</span>
                <span class="font-mono text-gray-300 truncate" style="max-width:40%;">{{ a.tool_name }}</span>
                <span class="text-gray-500 ml-auto whitespace-nowrap">{{ formatTime(a.timestamp) }}</span>
              </div>
            </div>
          </div>

          <!-- Recent Errors -->
          <div class="loki-card">
            <div class="text-gray-400 text-sm font-medium mb-2">Recent Errors</div>
            <div v-if="errorsLoading && errors.length === 0" class="text-gray-500 text-sm">Loading...</div>
            <div v-else-if="errors.length === 0" class="text-gray-500 text-sm">No recent errors</div>
            <div v-else class="space-y-1.5">
              <div v-for="(e, i) in errors" :key="i" class="text-xs py-1 border-b border-gray-800 last:border-0">
                <div class="flex items-center gap-2">
                  <span class="text-red-400">\u26a0</span>
                  <span class="font-mono text-red-300 truncate" style="max-width:40%;">{{ e.tool_name }}</span>
                  <span class="text-gray-500 ml-auto whitespace-nowrap">{{ formatTime(e.timestamp) }}</span>
                </div>
                <div v-if="e.error_message" class="text-gray-500 pl-5 truncate" style="max-width:90%;">{{ e.error_message }}</div>
              </div>
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
    const errors = ref([]);
    const errorsLoading = ref(false);
    const newEventCount = ref(0);
    const actionLoading = ref({ reload: false, clearSessions: false, stopLoops: false });
    const actionMessage = ref(null);
    let eventKeyCounter = 0;

    const uptime = computed(() => {
      const s = status.value.uptime_seconds || 0;
      const d = Math.floor(s / 86400);
      const h = Math.floor((s % 86400) / 3600);
      const m = Math.floor((s % 3600) / 60);
      const parts = [];
      if (d > 0) parts.push(`${d} day${d !== 1 ? 's' : ''}`);
      if (h > 0) parts.push(`${h} hour${h !== 1 ? 's' : ''}`);
      if (parts.length === 0 || (d === 0 && h === 0)) parts.push(`${m} min${m !== 1 ? 's' : ''}`);
      return parts.join(', ');
    });

    const stats = computed(() => [
      { label: 'Guilds', value: status.value.guild_count ?? 0, icon: '\u2302', iconColor: 'text-blue-400' },
      { label: 'Users', value: status.value.user_count ?? 0, icon: '\u263a', iconColor: 'text-cyan-400' },
      { label: 'Tools', value: status.value.tool_count ?? 0, icon: '\u2692', iconColor: 'text-purple-400' },
      { label: 'Sessions', value: status.value.session_count ?? 0, icon: '\u2630', iconColor: 'text-yellow-400' },
      { label: 'Loops', value: status.value.loop_count ?? 0, icon: '\u27f3', iconColor: 'text-green-400', color: status.value.loop_count > 0 ? 'text-green-400' : '' },
      { label: 'Schedules', value: status.value.schedule_count ?? 0, icon: '\u23f0', iconColor: 'text-orange-400' },
    ]);

    function formatTime(ts) {
      if (!ts) return '\u2014';
      try {
        const d = new Date(ts);
        return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
      } catch { return ts; }
    }

    function showAction(text, ok = true) {
      actionMessage.value = { text, ok };
      setTimeout(() => { actionMessage.value = null; }, 4000);
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
        activity.value = await api.get('/api/audit?limit=8');
        newEventCount.value = 0;
      } catch { /* ignore */ }
      activityLoading.value = false;
    }

    async function fetchErrors() {
      errorsLoading.value = true;
      try {
        errors.value = await api.get('/api/audit?error_only=1&limit=5');
      } catch { /* ignore */ }
      errorsLoading.value = false;
    }

    async function reloadConfig() {
      actionLoading.value = { ...actionLoading.value, reload: true };
      try {
        await api.post('/api/reload');
        showAction('Config reloaded');
      } catch (e) {
        showAction(e.message, false);
      }
      actionLoading.value = { ...actionLoading.value, reload: false };
    }

    async function clearSessions() {
      actionLoading.value = { ...actionLoading.value, clearSessions: true };
      try {
        const res = await api.post('/api/sessions/clear-all');
        showAction(`Cleared ${res.count} session${res.count !== 1 ? 's' : ''}`);
        await fetchStatus();
      } catch (e) {
        showAction(e.message, false);
      }
      actionLoading.value = { ...actionLoading.value, clearSessions: false };
    }

    async function stopAllLoops() {
      actionLoading.value = { ...actionLoading.value, stopLoops: true };
      try {
        const res = await api.post('/api/loops/stop-all');
        showAction(res.result);
        await fetchStatus();
      } catch (e) {
        showAction(e.message, false);
      }
      actionLoading.value = { ...actionLoading.value, stopLoops: false };
    }

    function retry() {
      loading.value = true;
      error.value = null;
      fetchStatus();
      fetchActivity();
      fetchErrors();
    }

    // Auto-refresh status every 15s
    let interval = null;

    function onEvent(data) {
      // Push new events to the top of the activity feed with flash animation
      if (data.payload && data.payload.tool_name) {
        const entry = { ...data.payload, _isNew: true, _key: ++eventKeyCounter };
        activity.value.unshift(entry);
        if (activity.value.length > 8) activity.value.pop();
        newEventCount.value++;
        // Also add to errors if it's an error
        if (entry.error) {
          errors.value.unshift(entry);
          if (errors.value.length > 5) errors.value.pop();
        }
        // Remove flash class after animation
        setTimeout(() => { entry._isNew = false; }, 1500);
        // Reset new event counter after a while
        clearTimeout(onEvent._resetTimer);
        onEvent._resetTimer = setTimeout(() => { newEventCount.value = 0; }, 10000);
      }
    }

    onMounted(async () => {
      await Promise.all([fetchStatus(), fetchActivity(), fetchErrors()]);
      interval = setInterval(fetchStatus, 15000);
      ws.subscribe('events', onEvent);
    });

    onUnmounted(() => {
      if (interval) clearInterval(interval);
      ws.unsubscribe('events', onEvent);
    });

    return {
      status, loading, error, uptime, stats,
      activity, activityLoading, newEventCount,
      errors, errorsLoading,
      actionLoading, actionMessage,
      fetchActivity, fetchStatus, formatTime, retry,
      reloadConfig, clearSessions, stopAllLoops,
    };
  },
};
