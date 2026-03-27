/**
 * Heimdall Management UI — Dashboard Page
 * Real-time stats, agent panel, health indicators, activity feed
 */
import { api, ws } from '../api.js';

const { ref, computed, onMounted, onUnmounted, nextTick } = Vue;

export default {
  template: `
    <div class="p-6 page-fade-in" role="region" aria-label="Dashboard">
      <!-- Skeleton loading -->
      <div v-if="loading" class="space-y-4" role="status" aria-label="Loading dashboard">
        <div class="hm-card dash-hero-skeleton">
          <div class="skeleton" style="width:48px;height:48px;border-radius:50%;flex-shrink:0;"></div>
          <div style="flex:1;">
            <div class="skeleton skeleton-text" style="width:140px;"></div>
            <div class="skeleton skeleton-text" style="width:200px;margin-bottom:0;"></div>
          </div>
        </div>
        <div class="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-3">
          <div v-for="n in 10" :key="n" class="hm-card text-center">
            <div class="skeleton skeleton-stat"></div>
            <div class="skeleton skeleton-text" style="width:60%;margin:0.25rem auto 0;"></div>
          </div>
        </div>
      </div>

      <!-- Error state with retry -->
      <div v-else-if="error" class="hm-card border-red-900 error-state" role="alert">
        <span class="error-icon" aria-hidden="true">\u26A0</span>
        <p class="text-red-400">{{ error }}</p>
        <button @click="retry" class="btn btn-ghost text-xs">Retry</button>
      </div>

      <div v-else>
        <!-- Hero status banner -->
        <div class="dash-hero hm-card mb-4">
          <div class="dash-hero-left">
            <div class="dash-hero-ring" :class="status.status === 'online' ? 'ring-online' : 'ring-starting'">
              <svg viewBox="0 0 48 48" class="dash-ring-svg" role="img" :aria-label="'Uptime: ' + uptime">
                <circle cx="24" cy="24" r="20" fill="none" stroke="currentColor" stroke-width="3" opacity="0.15"/>
                <circle cx="24" cy="24" r="20" fill="none" stroke="currentColor" stroke-width="3"
                  stroke-dasharray="125.66" :stroke-dashoffset="uptimeRingOffset"
                  stroke-linecap="round" class="dash-ring-progress"/>
              </svg>
              <span class="dash-hero-icon" aria-hidden="true">\u2300</span>
            </div>
            <div>
              <div class="dash-hero-name">Heimdall</div>
              <div class="dash-hero-sub">
                <span class="status-dot" :class="status.status === 'online' ? 'online' : 'starting'" style="width:8px;height:8px;"></span>
                {{ status.status === 'online' ? 'Online' : 'Starting' }}
                <span class="dash-hero-sep">\u00b7</span>
                {{ uptime }}
              </div>
            </div>
          </div>
          <div class="dash-hero-actions">
            <button @click="reloadConfig" class="btn btn-ghost text-xs" :disabled="actionLoading.reload">
              {{ actionLoading.reload ? '...' : '\u21bb Reload' }}
            </button>
            <button @click="clearSessions" class="btn btn-ghost text-xs" :disabled="actionLoading.clearSessions">
              {{ actionLoading.clearSessions ? '...' : '\u2715 Clear Sessions' }}
            </button>
            <button @click="stopAllLoops" class="btn btn-ghost text-xs" :disabled="actionLoading.stopLoops || (status.loop_count || 0) === 0">
              {{ actionLoading.stopLoops ? '...' : '\u25a0 Stop Loops' }}
            </button>
          </div>
          <div v-if="actionMessage" class="dash-hero-toast" :class="actionMessage.ok ? 'text-green-400' : 'text-red-400'" role="status" aria-live="polite">
            {{ actionMessage.text }}
          </div>
        </div>

        <!-- Stat cards grid -->
        <div class="grid grid-cols-2 md:grid-cols-5 gap-3 mb-4">
          <div v-for="s in stats" :key="s.label"
               class="hm-card stat-card dash-stat"
               :class="s.highlight ? 'dash-stat-highlight' : ''">
            <div class="dash-stat-header">
              <span class="dash-stat-icon" :class="s.iconColor">{{ s.icon }}</span>
              <span class="dash-stat-label">{{ s.label }}</span>
            </div>
            <div class="dash-stat-value" :class="s.color || ''">{{ s.value }}</div>
            <div v-if="s.sub" class="dash-stat-sub" :class="s.subColor || ''">{{ s.sub }}</div>
          </div>
        </div>

        <!-- Health indicators bar -->
        <div class="dash-health-bar hm-card mb-4" v-if="healthIndicators.length > 0" role="region" aria-label="System health">
          <div class="hm-section-title" style="margin-bottom:0.5rem;">System Health</div>
          <div class="dash-health-items">
            <div v-for="h in healthIndicators" :key="h.label" class="dash-health-item">
              <span class="dash-health-dot" :class="'dash-health-' + h.status" role="img" :aria-label="h.status"></span>
              <span class="dash-health-label">{{ h.label }}</span>
              <span v-if="h.detail" class="dash-health-detail">{{ h.detail }}</span>
            </div>
          </div>
        </div>

        <!-- Main grid: 2 columns on large, 1 on small -->
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-4">

          <!-- Active Agents panel -->
          <div class="hm-card dash-panel">
            <div class="dash-panel-header">
              <span class="dash-panel-title">Active Agents</span>
              <span class="badge badge-info" v-if="agents.length > 0">{{ agents.length }}</span>
            </div>
            <div v-if="agents.length === 0" class="dash-empty">
              <span class="dash-empty-icon">\u2693</span>
              <span>No active agents</span>
            </div>
            <div v-else class="dash-agent-list">
              <div v-for="a in agents" :key="a.id" class="dash-agent-item">
                <div class="dash-agent-top">
                  <span class="dash-agent-dot" :class="'dash-agent-' + a.status"></span>
                  <span class="dash-agent-label">{{ a.label }}</span>
                  <span class="dash-agent-iters">{{ a.iteration_count }} iters</span>
                </div>
                <div class="dash-agent-goal">{{ a.goal }}</div>
                <div class="dash-agent-meta">
                  <span>{{ formatDuration(a.runtime_seconds) }}</span>
                  <span v-if="a.tools_used.length > 0" class="dash-agent-tools">{{ a.tools_used.length }} tools</span>
                </div>
              </div>
            </div>
          </div>

          <!-- Recent Activity -->
          <div class="hm-card dash-panel">
            <div class="dash-panel-header">
              <span class="dash-panel-title">
                Recent Activity
                <span v-if="newEventCount > 0" class="badge badge-success" style="font-size:0.625rem;margin-left:4px;">+{{ newEventCount }}</span>
              </span>
              <button @click="fetchActivity" class="btn btn-ghost text-xs" :disabled="activityLoading" style="padding:2px 8px;">
                {{ activityLoading ? '...' : '\u21bb' }}
              </button>
            </div>
            <div v-if="activityLoading && activity.length === 0" class="dash-empty"><span>Loading...</span></div>
            <div v-else-if="activity.length === 0" class="dash-empty">
              <span class="dash-empty-icon">\u2042</span>
              <span>No recent activity</span>
            </div>
            <div v-else class="dash-activity-list">
              <div v-for="(a, i) in activity" :key="a._key || i"
                   class="dash-activity-item"
                   :class="{ 'flash-new': a._isNew }">
                <span class="dash-activity-dot" :class="a.error ? 'dot-error' : 'dot-ok'"></span>
                <span class="dash-activity-tool">{{ a.tool_name }}</span>
                <span class="dash-activity-time">{{ formatTime(a.timestamp) }}</span>
              </div>
            </div>
          </div>

          <!-- Connected Guilds + Errors stacked -->
          <div class="space-y-4">
            <!-- Guilds -->
            <div class="hm-card dash-panel">
              <div class="dash-panel-header">
                <span class="dash-panel-title">Guilds</span>
              </div>
              <div v-if="!status.guilds || status.guilds.length === 0" class="dash-empty">
                <span class="dash-empty-icon">\u2302</span>
                <span>No guilds</span>
              </div>
              <div v-else class="space-y-1.5">
                <div v-for="g in status.guilds" :key="g.id" class="dash-guild-item">
                  <span class="status-dot online" style="width:6px;height:6px;"></span>
                  <span>{{ g.name }}</span>
                  <span v-if="g.member_count" class="dash-guild-count">{{ g.member_count }}</span>
                </div>
              </div>
            </div>

            <!-- Recent Errors -->
            <div class="hm-card dash-panel">
              <div class="dash-panel-header">
                <span class="dash-panel-title">Recent Errors</span>
                <span v-if="errors.length > 0" class="badge badge-danger" style="font-size:0.625rem;">{{ errors.length }}</span>
              </div>
              <div v-if="errors.length === 0" class="dash-empty">
                <span class="dash-empty-icon">\u2713</span>
                <span>All clear</span>
              </div>
              <div v-else class="dash-error-list">
                <div v-for="(e, i) in errors" :key="i" class="dash-error-item">
                  <div class="dash-error-top">
                    <span class="text-red-400">\u26a0</span>
                    <span class="dash-error-tool">{{ e.tool_name }}</span>
                    <span class="dash-error-time">{{ formatTime(e.timestamp) }}</span>
                  </div>
                  <div v-if="e.error_message" class="dash-error-msg">{{ e.error_message }}</div>
                </div>
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
    const agents = ref([]);
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
      if (d > 0) parts.push(`${d}d`);
      if (h > 0) parts.push(`${h}h`);
      if (parts.length === 0 || (d === 0 && h === 0)) parts.push(`${m}m`);
      return parts.join(' ');
    });

    // Uptime ring: 0-100% of a day, full circle = 24h
    const uptimeRingOffset = computed(() => {
      const s = status.value.uptime_seconds || 0;
      const pct = Math.min(s / 86400, 1);
      return 125.66 * (1 - pct);
    });

    const stats = computed(() => {
      const s = status.value;
      return [
        {
          label: 'Guilds', value: s.guild_count ?? 0,
          icon: '\u2302', iconColor: 'text-blue-400',
        },
        {
          label: 'Sessions', value: s.session_count ?? 0,
          icon: '\u2630', iconColor: 'text-yellow-400',
        },
        {
          label: 'Tools', value: s.tool_count ?? 0,
          icon: '\u2692', iconColor: 'text-purple-400',
          sub: `${s.skill_count ?? 0} skills`, subColor: 'text-gray-500',
        },
        {
          label: 'Loops', value: s.loop_count ?? 0,
          icon: '\u27f3', iconColor: 'text-green-400',
          color: s.loop_count > 0 ? 'text-green-400' : '',
          highlight: s.loop_count > 0,
        },
        {
          label: 'Agents', value: s.agent_running ?? 0,
          icon: '\u2693', iconColor: 'text-cyan-400',
          sub: s.agent_count > 0 ? `${s.agent_count} total` : '',
          subColor: 'text-gray-500',
          highlight: (s.agent_running ?? 0) > 0,
        },
        {
          label: 'Processes', value: s.process_running ?? 0,
          icon: '\u2699', iconColor: 'text-orange-400',
          sub: s.process_count > 0 ? `${s.process_count} total` : '',
          subColor: 'text-gray-500',
          highlight: (s.process_running ?? 0) > 0,
        },
        {
          label: 'Schedules', value: s.schedule_count ?? 0,
          icon: '\u23f0', iconColor: 'text-amber-400',
        },
        {
          label: 'Users', value: s.user_count ?? 0,
          icon: '\u263a', iconColor: 'text-indigo-400',
        },
        {
          label: 'Knowledge', value: '-',
          icon: '\u2261', iconColor: 'text-teal-400',
        },
        {
          label: 'Alerts', value: (s.monitoring && s.monitoring.active_alerts) || 0,
          icon: '\u2691', iconColor: 'text-red-400',
          color: (s.monitoring && s.monitoring.active_alerts > 0) ? 'text-red-400' : '',
          highlight: (s.monitoring && s.monitoring.active_alerts > 0),
        },
      ];
    });

    const healthIndicators = computed(() => {
      const s = status.value;
      const items = [];
      // Bot status
      items.push({
        label: 'Bot',
        status: s.status === 'online' ? 'ok' : 'warn',
        detail: s.status === 'online' ? 'Online' : 'Starting',
      });
      // Monitoring
      const mon = s.monitoring || {};
      if (mon.enabled) {
        const hasAlerts = mon.active_alerts > 0;
        items.push({
          label: 'Monitoring',
          status: hasAlerts ? 'error' : 'ok',
          detail: hasAlerts ? `${mon.active_alerts} alert${mon.active_alerts > 1 ? 's' : ''}` : `${mon.checks} checks`,
        });
      }
      // Loops health
      if ((s.loop_count || 0) > 0) {
        items.push({
          label: 'Loops',
          status: 'ok',
          detail: `${s.loop_count} active`,
        });
      }
      // Agents
      if ((s.agent_running || 0) > 0) {
        items.push({
          label: 'Agents',
          status: 'ok',
          detail: `${s.agent_running} running`,
        });
      }
      // Processes
      if ((s.process_running || 0) > 0) {
        items.push({
          label: 'Processes',
          status: 'ok',
          detail: `${s.process_running} running`,
        });
      }
      return items;
    });

    function formatTime(ts) {
      if (!ts) return '\u2014';
      try {
        const d = new Date(ts);
        return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
      } catch { return ts; }
    }

    function formatDuration(secs) {
      if (!secs) return '0s';
      if (secs < 60) return `${Math.round(secs)}s`;
      const m = Math.floor(secs / 60);
      const s = Math.round(secs % 60);
      return s > 0 ? `${m}m ${s}s` : `${m}m`;
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
        activity.value = await api.get('/api/audit?limit=10');
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

    async function fetchAgents() {
      try {
        const data = await api.get('/api/agents');
        agents.value = data.filter(a => a.status === 'running');
      } catch { /* ignore */ }
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
      fetchAgents();
    }

    // Auto-refresh
    let statusInterval = null;
    let agentInterval = null;

    function onEvent(data) {
      if (data.payload && data.payload.tool_name) {
        const entry = { ...data.payload, _isNew: true, _key: ++eventKeyCounter };
        activity.value.unshift(entry);
        if (activity.value.length > 10) activity.value.pop();
        newEventCount.value++;
        if (entry.error) {
          errors.value.unshift(entry);
          if (errors.value.length > 5) errors.value.pop();
        }
        setTimeout(() => { entry._isNew = false; }, 1500);
        clearTimeout(onEvent._resetTimer);
        onEvent._resetTimer = setTimeout(() => { newEventCount.value = 0; }, 10000);
      }
    }

    onMounted(async () => {
      await Promise.all([fetchStatus(), fetchActivity(), fetchErrors(), fetchAgents()]);
      statusInterval = setInterval(fetchStatus, 15000);
      agentInterval = setInterval(fetchAgents, 10000);
      ws.subscribe('events', onEvent);
    });

    onUnmounted(() => {
      if (statusInterval) clearInterval(statusInterval);
      if (agentInterval) clearInterval(agentInterval);
      ws.unsubscribe('events', onEvent);
    });

    return {
      status, loading, error, uptime, uptimeRingOffset, stats,
      healthIndicators,
      activity, activityLoading, newEventCount,
      errors, errorsLoading,
      agents,
      actionLoading, actionMessage,
      fetchActivity, fetchStatus, formatTime, formatDuration, retry,
      reloadConfig, clearSessions, stopAllLoops,
    };
  },
};
