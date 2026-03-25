/**
 * Loki Management UI — Sessions Page
 * View active sessions, conversation history, clear sessions
 */
import { api, ws } from '../api.js';

const { ref, computed, onMounted, onUnmounted } = Vue;

export default {
  template: `
    <div class="p-6">
      <div class="flex items-center justify-between mb-4">
        <h1 class="text-xl font-semibold">Sessions</h1>
        <button @click="fetchSessions" class="btn btn-ghost text-xs" :disabled="loading">
          {{ loading ? 'Loading...' : 'Refresh' }}
        </button>
      </div>

      <div v-if="loading && sessions.length === 0" class="flex items-center gap-2 text-gray-400">
        <div class="spinner"></div> Loading sessions...
      </div>
      <div v-else-if="error" class="loki-card border-red-900">
        <p class="text-red-400">{{ error }}</p>
      </div>
      <div v-else-if="sessions.length === 0" class="loki-card">
        <p class="text-gray-400">No active sessions</p>
      </div>
      <div v-else>
        <table class="loki-table">
          <thead>
            <tr>
              <th>Channel</th>
              <th>Messages</th>
              <th>Last Active</th>
              <th>Summary</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="s in sessions" :key="s.channel_id"
                @click="toggleSession(s.channel_id)"
                style="cursor:pointer;">
              <td class="font-mono text-sm">{{ s.channel_id }}</td>
              <td>
                <span class="badge badge-info">{{ s.message_count }}</span>
              </td>
              <td class="text-gray-400 text-sm">{{ formatAge(s.last_active) }}</td>
              <td>
                <span v-if="s.has_summary" class="badge badge-success">yes</span>
                <span v-else class="text-gray-500 text-xs">—</span>
              </td>
              <td @click.stop>
                <button @click="confirmClear(s.channel_id)" class="btn btn-danger text-xs">Clear</button>
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <!-- Expanded session detail -->
      <div v-if="expandedId" class="mt-4">
        <div class="loki-card">
          <div class="flex items-center justify-between mb-3">
            <div class="text-sm font-medium">
              Session: <span class="font-mono">{{ expandedId }}</span>
            </div>
            <button @click="expandedId = null" class="btn btn-ghost text-xs">Close</button>
          </div>

          <div v-if="detailLoading" class="flex items-center gap-2 text-gray-400 text-sm">
            <div class="spinner" style="width:14px;height:14px;border-width:2px;"></div> Loading...
          </div>
          <div v-else-if="detail">
            <!-- Summary banner -->
            <div v-if="detail.summary" class="mb-3 p-2 rounded bg-gray-900 border border-gray-800 text-sm text-gray-300">
              <span class="text-gray-500 text-xs font-medium">Summary:</span> {{ detail.summary }}
            </div>

            <!-- Message list -->
            <div class="session-messages space-y-2 max-h-96 overflow-y-auto pr-1" style="scrollbar-gutter: stable;">
              <div v-for="(m, i) in detail.messages" :key="i"
                   class="p-2 rounded text-sm"
                   :class="messageClass(m.role)">
                <div class="flex items-center gap-2 mb-1">
                  <span class="badge" :class="roleBadge(m.role)">{{ m.role }}</span>
                  <span v-if="m.user_id" class="text-gray-500 text-xs font-mono">{{ m.user_id }}</span>
                  <span class="text-gray-500 text-xs ml-auto">{{ formatTimestamp(m.timestamp) }}</span>
                </div>
                <div class="whitespace-pre-wrap break-words text-gray-200" style="font-size:0.8125rem;">{{ truncateContent(m.content) }}</div>
              </div>
            </div>
            <div v-if="detail.messages && detail.messages.length === 0" class="text-gray-500 text-sm">No messages in this session</div>
          </div>
        </div>
      </div>

      <!-- Confirm clear modal -->
      <div v-if="clearTarget" class="modal-overlay" @click.self="clearTarget = null">
        <div class="modal-content">
          <h3 class="text-lg font-semibold mb-2">Clear Session</h3>
          <p class="text-gray-400 text-sm mb-4">
            Clear all conversation history for channel <span class="font-mono">{{ clearTarget }}</span>? This cannot be undone.
          </p>
          <div class="flex gap-2 justify-end">
            <button @click="clearTarget = null" class="btn btn-ghost">Cancel</button>
            <button @click="clearSession" class="btn btn-danger" :disabled="clearing">
              {{ clearing ? 'Clearing...' : 'Clear Session' }}
            </button>
          </div>
        </div>
      </div>
    </div>`,

  setup() {
    const sessions = ref([]);
    const loading = ref(true);
    const error = ref(null);
    const expandedId = ref(null);
    const detail = ref(null);
    const detailLoading = ref(false);
    const clearTarget = ref(null);
    const clearing = ref(false);

    function formatAge(ts) {
      if (!ts) return '—';
      const diff = (Date.now() / 1000) - ts;
      if (diff < 60) return 'just now';
      if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
      if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
      return `${Math.floor(diff / 86400)}d ago`;
    }

    function formatTimestamp(ts) {
      if (!ts) return '';
      try {
        const d = new Date(ts * 1000);
        return d.toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
      } catch { return ''; }
    }

    function messageClass(role) {
      if (role === 'user') return 'bg-gray-900/50 border border-gray-800';
      if (role === 'assistant') return 'bg-indigo-950/30 border border-indigo-900/30';
      return 'bg-gray-900/30 border border-gray-800/50';
    }

    function roleBadge(role) {
      if (role === 'user') return 'badge-info';
      if (role === 'assistant') return 'badge-success';
      return 'badge-warning';
    }

    function truncateContent(content) {
      if (!content) return '';
      if (content.length > 2000) return content.slice(0, 2000) + '\n... (truncated)';
      return content;
    }

    async function fetchSessions() {
      loading.value = true;
      error.value = null;
      try {
        sessions.value = await api.get('/api/sessions');
      } catch (e) {
        error.value = e.message;
      }
      loading.value = false;
    }

    async function toggleSession(channelId) {
      if (expandedId.value === channelId) {
        expandedId.value = null;
        detail.value = null;
        return;
      }
      expandedId.value = channelId;
      detail.value = null;
      detailLoading.value = true;
      try {
        detail.value = await api.get(`/api/sessions/${channelId}`);
      } catch (e) {
        detail.value = { messages: [], summary: '' };
      }
      detailLoading.value = false;
    }

    function confirmClear(channelId) {
      clearTarget.value = channelId;
    }

    async function clearSession() {
      if (!clearTarget.value) return;
      clearing.value = true;
      try {
        await api.del(`/api/sessions/${clearTarget.value}`);
        // If we had this session expanded, close it
        if (expandedId.value === clearTarget.value) {
          expandedId.value = null;
          detail.value = null;
        }
        await fetchSessions();
      } catch { /* ignore */ }
      clearing.value = false;
      clearTarget.value = null;
    }

    // WebSocket: refresh on new message events
    function onEvent(data) {
      if (data.payload && data.payload.channel_id) {
        fetchSessions();
      }
    }

    onMounted(() => {
      fetchSessions();
      ws.subscribe('events', onEvent);
    });

    onUnmounted(() => {
      ws.unsubscribe('events', onEvent);
    });

    return {
      sessions, loading, error,
      expandedId, detail, detailLoading,
      clearTarget, clearing,
      formatAge, formatTimestamp, messageClass, roleBadge, truncateContent,
      fetchSessions, toggleSession, confirmClear, clearSession,
    };
  },
};
