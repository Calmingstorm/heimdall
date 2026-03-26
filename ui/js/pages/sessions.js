/**
 * Loki Management UI — Sessions Page
 * View active sessions, conversation history, bulk manage, export
 * Auto-refreshes via debounced WebSocket events
 */
import { api, ws } from '../api.js';

const { ref, computed, onMounted, onUnmounted, nextTick } = Vue;

export default {
  template: `
    <div class="p-6 page-fade-in">
      <div class="flex items-center justify-between mb-4 flex-wrap gap-2">
        <h1 class="text-xl font-semibold">Sessions</h1>
        <div class="flex items-center gap-2">
          <button v-if="selected.size > 0" @click="confirmBulkClear"
                  class="btn btn-danger text-xs">
            Clear Selected ({{ selected.size }})
          </button>
          <button @click="fetchSessions" class="btn btn-ghost text-xs" :disabled="loading">
            {{ loading ? 'Loading...' : 'Refresh' }}
          </button>
        </div>
      </div>

      <!-- Skeleton loading -->
      <div v-if="loading && sessions.length === 0" class="space-y-2">
        <div v-for="n in 4" :key="n" class="skeleton skeleton-row"></div>
      </div>
      <div v-else-if="error" class="loki-card border-red-900 error-state">
        <span class="error-icon">\u26A0</span>
        <p class="text-red-400">{{ error }}</p>
        <button @click="retry" class="btn btn-ghost text-xs">Retry</button>
      </div>
      <div v-else-if="sessions.length === 0" class="loki-card empty-state">
        <span class="empty-state-icon">\u{1F4AC}</span>
        <span class="empty-state-text">No active sessions</span>
        <span class="empty-state-hint">Sessions appear when users interact with Loki via Discord or the chat interface</span>
      </div>
      <div v-else>
        <!-- Select all -->
        <div class="flex items-center gap-2 mb-2 text-sm text-gray-400">
          <label class="flex items-center gap-1 cursor-pointer">
            <input type="checkbox" :checked="allSelected" @change="toggleSelectAll"
                   class="session-checkbox" />
            <span>Select all ({{ sessions.length }})</span>
          </label>
        </div>

        <div class="space-y-2">
          <div v-for="s in sessions" :key="s.channel_id"
               class="session-card loki-card"
               :class="{ 'flash-new': s._updated, 'session-selected': selected.has(s.channel_id) }">
            <!-- Header row -->
            <div class="flex items-center gap-3 cursor-pointer" @click="toggleSession(s.channel_id)">
              <input type="checkbox" :checked="selected.has(s.channel_id)"
                     @click.stop @change="toggleSelect(s.channel_id)"
                     class="session-checkbox" />
              <div class="flex-1 min-w-0">
                <div class="flex items-center gap-2 flex-wrap">
                  <span class="font-mono text-sm">{{ s.channel_id }}</span>
                  <span class="badge" :class="s.source === 'web' ? 'badge-info' : 'badge-success'">
                    {{ s.source }}
                  </span>
                  <span class="badge badge-info">{{ s.message_count }} msg</span>
                  <span v-if="s.has_summary" class="badge badge-warning" title="Session has compacted summary">summary</span>
                </div>
                <div class="text-xs text-gray-500 mt-1">
                  Active {{ formatAge(s.last_active) }} · Created {{ formatAge(s.created_at) }}
                  <span v-if="s.last_user_id"> · Last user: <span class="font-mono">{{ s.last_user_id }}</span></span>
                </div>
              </div>
              <div class="flex items-center gap-1" @click.stop>
                <button @click="exportSession(s.channel_id, 'json')" class="btn btn-ghost text-xs" title="Export JSON">
                  JSON
                </button>
                <button @click="exportSession(s.channel_id, 'text')" class="btn btn-ghost text-xs" title="Export text">
                  TXT
                </button>
                <button @click="confirmClear(s.channel_id)" class="btn btn-danger text-xs">Clear</button>
              </div>
            </div>

            <!-- Preview (last 2 messages) -->
            <div v-if="s.preview && s.preview.length > 0 && expandedId !== s.channel_id"
                 class="session-preview mt-2 pt-2 border-t border-gray-800">
              <div v-for="(p, i) in s.preview" :key="i" class="flex gap-2 text-xs mb-1 last:mb-0">
                <span class="session-preview-role" :class="p.role === 'user' ? 'text-cyan-400' : 'text-indigo-400'">
                  {{ p.role === 'user' ? 'USER' : 'LOKI' }}:
                </span>
                <span class="text-gray-400 truncate">{{ p.content || '(empty)' }}</span>
              </div>
            </div>

            <!-- Expanded session detail -->
            <div v-if="expandedId === s.channel_id" class="mt-3 pt-3 border-t border-gray-800">
              <div v-if="detailLoading" class="flex items-center gap-2 text-gray-400 text-sm">
                <div class="spinner" style="width:14px;height:14px;border-width:2px;"></div> Loading...
              </div>
              <div v-else-if="detail">
                <!-- Summary banner -->
                <div v-if="detail.summary" class="mb-3 p-2 rounded bg-gray-900 border border-gray-800 text-sm text-gray-300">
                  <span class="text-gray-500 text-xs font-medium uppercase">Compacted Summary</span>
                  <div class="mt-1">{{ detail.summary }}</div>
                </div>

                <!-- Message list -->
                <div class="session-messages space-y-2 max-h-96 overflow-y-auto pr-1" style="scrollbar-gutter: stable;">
                  <div v-for="(m, i) in detail.messages" :key="i"
                       class="session-msg p-2 rounded text-sm"
                       :class="messageClass(m.role)">
                    <div class="flex items-center gap-2 mb-1">
                      <span class="badge" :class="roleBadge(m.role)">{{ m.role }}</span>
                      <span v-if="m.user_id" class="text-gray-500 text-xs font-mono">{{ m.user_id }}</span>
                      <span class="text-gray-600 text-xs ml-auto" :title="formatFullTimestamp(m.timestamp)">
                        {{ formatTimestamp(m.timestamp) }}
                      </span>
                    </div>
                    <div class="whitespace-pre-wrap break-words text-gray-200 session-msg-content">{{ truncateContent(m.content) }}</div>
                  </div>
                </div>
                <div v-if="detail.messages && detail.messages.length === 0" class="text-gray-500 text-sm">No messages in this session</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Confirm clear modal (single) -->
      <div v-if="clearTarget" class="modal-overlay" @click.self="clearTarget = null" @keyup.escape="clearTarget = null">
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

      <!-- Confirm bulk clear modal -->
      <div v-if="bulkClearing" class="modal-overlay" @click.self="bulkClearing = false" @keyup.escape="bulkClearing = false">
        <div class="modal-content">
          <h3 class="text-lg font-semibold mb-2">Clear Selected Sessions</h3>
          <p class="text-gray-400 text-sm mb-4">
            Clear <strong>{{ selected.size }}</strong> selected session(s)? This cannot be undone.
          </p>
          <div class="flex gap-2 justify-end">
            <button @click="bulkClearing = false" class="btn btn-ghost">Cancel</button>
            <button @click="doBulkClear" class="btn btn-danger" :disabled="clearing">
              {{ clearing ? 'Clearing...' : 'Clear All Selected' }}
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
    const selected = ref(new Set());
    const bulkClearing = ref(false);

    const allSelected = computed(() =>
      sessions.value.length > 0 && selected.value.size === sessions.value.length
    );

    function formatAge(ts) {
      if (!ts) return '\u2014';
      const diff = (Date.now() / 1000) - ts;
      if (diff < 60) return 'just now';
      if (diff < 3600) {
        const m = Math.floor(diff / 60);
        return `${m} minute${m !== 1 ? 's' : ''} ago`;
      }
      if (diff < 86400) {
        const h = Math.floor(diff / 3600);
        return `${h} hour${h !== 1 ? 's' : ''} ago`;
      }
      const d = Math.floor(diff / 86400);
      return `${d} day${d !== 1 ? 's' : ''} ago`;
    }

    function formatTimestamp(ts) {
      if (!ts) return '';
      try {
        const d = new Date(ts * 1000);
        return d.toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
      } catch { return ''; }
    }

    function formatFullTimestamp(ts) {
      if (!ts) return '';
      try {
        return new Date(ts * 1000).toLocaleString();
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

    function retry() {
      error.value = null;
      fetchSessions();
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

    // Selection
    function toggleSelect(channelId) {
      const s = new Set(selected.value);
      if (s.has(channelId)) s.delete(channelId);
      else s.add(channelId);
      selected.value = s;
    }

    function toggleSelectAll() {
      if (allSelected.value) {
        selected.value = new Set();
      } else {
        selected.value = new Set(sessions.value.map(s => s.channel_id));
      }
    }

    // Single clear
    function confirmClear(channelId) {
      clearTarget.value = channelId;
    }

    async function clearSession() {
      if (!clearTarget.value) return;
      clearing.value = true;
      try {
        await api.del(`/api/sessions/${clearTarget.value}`);
        if (expandedId.value === clearTarget.value) {
          expandedId.value = null;
          detail.value = null;
        }
        selected.value.delete(clearTarget.value);
        await fetchSessions();
      } catch { /* ignore */ }
      clearing.value = false;
      clearTarget.value = null;
    }

    // Bulk clear
    function confirmBulkClear() {
      bulkClearing.value = true;
    }

    async function doBulkClear() {
      if (selected.value.size === 0) return;
      clearing.value = true;
      try {
        await api.post('/api/sessions/clear-bulk', {
          channel_ids: [...selected.value],
        });
        if (selected.value.has(expandedId.value)) {
          expandedId.value = null;
          detail.value = null;
        }
        selected.value = new Set();
        await fetchSessions();
      } catch { /* ignore */ }
      clearing.value = false;
      bulkClearing.value = false;
    }

    // Export
    function exportSession(channelId, format) {
      const token = api._token;
      let url = `/api/sessions/${channelId}/export?format=${format}`;
      if (token) url += `&token=${encodeURIComponent(token)}`;
      // Use a hidden link to trigger download
      const a = document.createElement('a');
      a.href = url;
      a.download = `session-${channelId}.${format === 'text' ? 'txt' : 'json'}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    }

    // WebSocket: debounced refresh on new message events
    let debounceTimer = null;
    function onEvent(data) {
      if (data.payload && data.payload.channel_id) {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => {
          fetchSessions();
          if (expandedId.value && data.payload.channel_id === expandedId.value) {
            // Refresh expanded detail
            api.get(`/api/sessions/${expandedId.value}`)
              .then(d => { detail.value = d; })
              .catch(() => {});
          }
        }, 2000);
      }
    }

    onMounted(() => {
      fetchSessions();
      ws.subscribe('events', onEvent);
    });

    onUnmounted(() => {
      ws.unsubscribe('events', onEvent);
      clearTimeout(debounceTimer);
    });

    return {
      sessions, loading, error,
      expandedId, detail, detailLoading,
      clearTarget, clearing,
      selected, allSelected, bulkClearing,
      formatAge, formatTimestamp, formatFullTimestamp,
      messageClass, roleBadge, truncateContent,
      fetchSessions, retry, toggleSession,
      toggleSelect, toggleSelectAll,
      confirmClear, clearSession,
      confirmBulkClear, doBulkClear,
      exportSession,
    };
  },
};
