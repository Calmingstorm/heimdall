/**
 * Heimdall Management UI — Sessions Page (Redesigned)
 * Conversation threading, filter presets, sort options, visual improvements
 */
import { api, ws } from '../api.js';

const { ref, computed, onMounted, onUnmounted, nextTick, watch } = Vue;

const FILTER_PRESETS = [
  { id: 'all', name: 'All Sessions', icon: '\u2630', filters: {} },
  { id: 'active', name: 'Recently Active', icon: '\u26A1', filters: { minAge: 0, maxAge: 3600 } },
  { id: 'discord', name: 'Discord Only', icon: '\u{1F4AC}', filters: { source: 'discord' } },
  { id: 'web', name: 'Web Only', icon: '\u{1F310}', filters: { source: 'web' } },
  { id: 'long', name: 'Long Conversations', icon: '\u{1F4D6}', filters: { minMessages: 10 } },
  { id: 'compacted', name: 'Compacted', icon: '\u{1F5DC}', filters: { hasCompaction: true } },
];

const SORT_OPTIONS = [
  { value: 'last_active', label: 'Last Active', icon: '\u{1F551}' },
  { value: 'created_at', label: 'Created', icon: '\u{1F4C5}' },
  { value: 'message_count', label: 'Message Count', icon: '\u{1F4CA}' },
];

export default {
  template: `
    <div class="p-6 page-fade-in">
      <!-- Header -->
      <div class="flex items-center justify-between mb-4 flex-wrap gap-2">
        <div>
          <h1 class="text-xl font-semibold">Sessions</h1>
          <p class="text-xs text-gray-500 mt-0.5" v-if="sessions.length > 0">
            {{ sessions.length }} session{{ sessions.length !== 1 ? 's' : '' }}
            <span v-if="filteredSessions.length !== sessions.length">
              · {{ filteredSessions.length }} shown
            </span>
          </p>
        </div>
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

      <!-- Filter presets bar -->
      <div class="sess-filter-bar mb-3">
        <div class="flex gap-1.5 flex-wrap items-center">
          <button v-for="preset in filterPresets" :key="preset.id"
                  @click="applyPreset(preset.id)"
                  class="sess-preset-chip"
                  :class="{ 'sess-preset-active': activePreset === preset.id }">
            <span class="sess-preset-icon">{{ preset.icon }}</span>
            <span>{{ preset.name }}</span>
          </button>
        </div>
        <div class="flex gap-2 items-center mt-2">
          <!-- Search -->
          <input v-model="searchQuery" type="text" class="hm-input flex-1"
                 placeholder="Search channels, users..." style="min-width: 140px; max-width: 300px;" />
          <!-- Sort -->
          <select v-model="sortBy" class="hm-select">
            <option v-for="opt in sortOptions" :key="opt.value" :value="opt.value">
              {{ opt.icon }} {{ opt.label }}
            </option>
          </select>
          <button @click="sortAsc = !sortAsc" class="btn btn-ghost text-xs"
                  :title="sortAsc ? 'Ascending' : 'Descending'">
            {{ sortAsc ? '\u2191' : '\u2193' }}
          </button>
        </div>
        <!-- Custom preset save -->
        <div v-if="hasActiveFilters && activePreset === 'all'" class="mt-2 flex items-center gap-2">
          <button @click="showSavePreset = !showSavePreset" class="btn btn-ghost text-xs">
            Save as preset
          </button>
          <template v-if="showSavePreset">
            <input v-model="newPresetName" type="text" class="hm-input text-xs"
                   placeholder="Preset name..." style="max-width: 180px;" />
            <button @click="saveCustomPreset" class="btn btn-primary text-xs" :disabled="!newPresetName.trim()">
              Save
            </button>
          </template>
        </div>
        <!-- Custom presets -->
        <div v-if="customPresets.length > 0" class="flex gap-1.5 flex-wrap mt-2">
          <button v-for="cp in customPresets" :key="cp.id"
                  @click="applyCustomPreset(cp)"
                  class="sess-preset-chip sess-preset-custom"
                  :class="{ 'sess-preset-active': activePreset === cp.id }">
            <span>\u2605</span>
            <span>{{ cp.name }}</span>
            <span class="sess-preset-remove" @click.stop="removeCustomPreset(cp.id)"
                  title="Remove preset">&times;</span>
          </button>
        </div>
      </div>

      <!-- Skeleton loading -->
      <div v-if="loading && sessions.length === 0" class="space-y-2">
        <div v-for="n in 4" :key="n" class="skeleton skeleton-row"></div>
      </div>
      <div v-else-if="error" class="hm-card border-red-900 error-state" role="alert">
        <span class="error-icon" aria-hidden="true">\u26A0</span>
        <p class="text-red-400">{{ error }}</p>
        <button @click="retry" class="btn btn-ghost text-xs">Retry</button>
      </div>
      <div v-else-if="sessions.length === 0" class="hm-card empty-state">
        <span class="empty-state-icon">\u{1F4AC}</span>
        <span class="empty-state-text">No active sessions</span>
        <span class="empty-state-hint">Sessions appear when users interact with Heimdall via Discord or the chat interface</span>
      </div>
      <div v-else-if="filteredSessions.length === 0" class="hm-card empty-state">
        <span class="empty-state-icon">\u{1F50D}</span>
        <span class="empty-state-text">No sessions match the current filter</span>
        <button @click="resetFilters" class="btn btn-ghost text-xs mt-2">Clear Filters</button>
      </div>
      <div v-else>
        <!-- Select all -->
        <div class="flex items-center gap-2 mb-2 text-sm text-gray-400">
          <label class="flex items-center gap-1 cursor-pointer">
            <input type="checkbox" :checked="allSelected" @change="toggleSelectAll"
                   class="session-checkbox" />
            <span>Select all ({{ filteredSessions.length }})</span>
          </label>
        </div>

        <div class="space-y-2">
          <div v-for="s in filteredSessions" :key="s.channel_id"
               class="session-card hm-card"
               :class="{ 'flash-new': s._updated, 'session-selected': selected.has(s.channel_id) }">
            <!-- Header row -->
            <div class="flex items-center gap-3 cursor-pointer" @click="toggleSession(s.channel_id)">
              <input type="checkbox" :checked="selected.has(s.channel_id)"
                     @click.stop @change="toggleSelect(s.channel_id)"
                     class="session-checkbox" />
              <div class="sess-source-icon" :class="s.source === 'web' ? 'sess-source-web' : 'sess-source-discord'">
                {{ s.source === 'web' ? '\u{1F310}' : '\u{1F4AC}' }}
              </div>
              <div class="flex-1 min-w-0">
                <div class="flex items-center gap-2 flex-wrap">
                  <span class="font-mono text-sm font-medium">{{ s.channel_id }}</span>
                  <span class="badge badge-info">{{ s.message_count }} msg</span>
                  <span v-if="s.has_summary" class="badge badge-warning" title="Session has compacted summary">compacted</span>
                </div>
                <div class="text-xs text-gray-500 mt-1">
                  Active {{ formatAge(s.last_active) }} · Created {{ formatAge(s.created_at) }}
                  <span v-if="s.last_user_id"> · <span class="font-mono">{{ s.last_user_id }}</span></span>
                </div>
              </div>
              <div class="flex items-center gap-1" @click.stop>
                <span class="sess-expand-icon" :class="{ 'sess-expanded': expandedId === s.channel_id }">
                  \u25B6
                </span>
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
                  {{ p.role === 'user' ? 'USER' : 'HEIMDALL' }}:
                </span>
                <span class="text-gray-400 truncate">{{ p.content || '(empty)' }}</span>
              </div>
            </div>

            <!-- Expanded session detail with conversation threading -->
            <div v-if="expandedId === s.channel_id" class="mt-3 pt-3 border-t border-gray-800">
              <div v-if="detailLoading" class="flex items-center gap-2 text-gray-400 text-sm">
                <div class="spinner" style="width:14px;height:14px;border-width:2px;"></div> Loading...
              </div>
              <div v-else-if="detail">
                <!-- Summary banner -->
                <div v-if="detail.summary" class="sess-summary-banner mb-3">
                  <div class="sess-summary-label">Compacted Summary</div>
                  <div class="mt-1 text-sm text-gray-300">{{ detail.summary }}</div>
                </div>

                <!-- Thread view toggle -->
                <div class="flex items-center gap-2 mb-3">
                  <button @click="threadView = 'threaded'" class="sess-view-btn"
                          :class="{ 'sess-view-active': threadView === 'threaded' }">
                    Threaded
                  </button>
                  <button @click="threadView = 'flat'" class="sess-view-btn"
                          :class="{ 'sess-view-active': threadView === 'flat' }">
                    Flat
                  </button>
                  <span class="text-xs text-gray-500 ml-2" v-if="detail.messages">
                    {{ detail.messages.length }} message{{ detail.messages.length !== 1 ? 's' : '' }}
                    <span v-if="threadView === 'threaded' && threads.length > 0">
                      · {{ threads.length }} thread{{ threads.length !== 1 ? 's' : '' }}
                    </span>
                  </span>
                </div>

                <!-- THREADED view -->
                <div v-if="threadView === 'threaded'" class="max-h-96 overflow-y-auto pr-1" style="scrollbar-gutter: stable;">
                  <div v-for="(thread, ti) in threads" :key="ti" class="mb-4">
                    <div class="flex items-center gap-2 mb-2 px-2 py-1 bg-gray-800 rounded cursor-pointer select-none"
                         @click="toggleThread(ti)" role="button" tabindex="0"
                         @keydown.enter="toggleThread(ti)" @keydown.space.prevent="toggleThread(ti)"
                         :aria-expanded="!collapsedThreads.has(ti)">
                      <span class="text-xs font-bold text-amber-400">#{{ ti + 1 }}</span>
                      <span class="text-xs text-gray-300">{{ threadSummary(thread) }}</span>
                      <span class="text-xs bg-gray-700 px-1.5 py-0.5 rounded text-gray-300">{{ thread.length }} msg</span>
                      <span class="text-xs text-gray-500 ml-auto" v-if="thread[0]">{{ formatTimestamp(thread[0].timestamp) }}</span>
                      <span class="text-xs text-gray-500">{{ collapsedThreads.has(ti) ? '\u25B6' : '\u25BC' }}</span>
                    </div>
                    <div v-if="!collapsedThreads.has(ti)" class="space-y-2 pl-2">
                      <div v-for="(m, mi) in thread" :key="mi"
                           class="p-2 rounded text-sm"
                           :class="messageClass(m.role)">
                        <div class="flex items-center gap-2 mb-1">
                          <span class="badge" :class="roleBadge(m.role)">{{ m.role }}</span>
                          <span v-if="m.user_id" class="text-gray-500 text-xs font-mono">{{ m.user_id }}</span>
                          <span class="text-gray-600 text-xs ml-auto" :title="formatFullTimestamp(m.timestamp)">
                            {{ formatTimestamp(m.timestamp) }}
                          </span>
                        </div>
                        <div class="whitespace-pre-wrap break-words text-gray-200 text-sm">{{ truncateContent(m.content) }}</div>
                      </div>
                    </div>
                  </div>
                  <div v-if="threads.length === 0 && detail.messages && detail.messages.length === 0"
                       class="text-gray-500 text-sm">No messages in this session</div>
                </div>

                <!-- FLAT view (original) -->
                <div v-else class="session-messages space-y-2 max-h-96 overflow-y-auto pr-1" style="scrollbar-gutter: stable;">
                  <div v-for="(m, i) in detail.messages" :key="i"
                       class="session-msg p-2 rounded text-sm"
                       :class="messageClass(m.role)">
                    <div class="flex items-center gap-2 mb-1">
                      <span class="sess-role-dot" :class="roleDotClass(m.role)"></span>
                      <span class="badge" :class="roleBadge(m.role)">{{ m.role }}</span>
                      <span v-if="m.user_id" class="text-gray-500 text-xs font-mono">{{ m.user_id }}</span>
                      <span class="text-gray-600 text-xs ml-auto" :title="formatFullTimestamp(m.timestamp)">
                        {{ formatTimestamp(m.timestamp) }}
                      </span>
                    </div>
                    <div class="whitespace-pre-wrap break-words text-gray-200 session-msg-content">{{ truncateContent(m.content) }}</div>
                  </div>
                  <div v-if="detail.messages && detail.messages.length === 0" class="text-gray-500 text-sm">No messages in this session</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Confirm clear modal (single) -->
      <div v-if="clearTarget" class="modal-overlay" @click.self="clearTarget = null" @keyup.escape="clearTarget = null" role="dialog" aria-modal="true" aria-labelledby="sess-clear-title">
        <div class="modal-content">
          <h3 id="sess-clear-title" class="text-lg font-semibold mb-2">Clear Session</h3>
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
      <div v-if="bulkClearing" class="modal-overlay" @click.self="bulkClearing = false" @keyup.escape="bulkClearing = false" role="dialog" aria-modal="true" aria-labelledby="sess-bulk-clear-title">
        <div class="modal-content">
          <h3 id="sess-bulk-clear-title" class="text-lg font-semibold mb-2">Clear Selected Sessions</h3>
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

    // Filter + sort state
    const activePreset = ref('all');
    const searchQuery = ref('');
    const sortBy = ref('last_active');
    const sortAsc = ref(false);
    const filterPresets = FILTER_PRESETS;
    const sortOptions = SORT_OPTIONS;

    // Custom presets (localStorage)
    const customPresets = ref([]);
    const showSavePreset = ref(false);
    const newPresetName = ref('');

    // Thread view
    const threadView = ref('flat');
    const collapsedThreads = ref(new Set());

    // Load custom presets from localStorage
    function loadCustomPresets() {
      try {
        const saved = localStorage.getItem('heimdall-session-presets');
        if (saved) customPresets.value = JSON.parse(saved);
      } catch { /* ignore */ }
    }

    function saveCustomPresetsToStorage() {
      try {
        localStorage.setItem('heimdall-session-presets', JSON.stringify(customPresets.value));
      } catch { /* ignore */ }
    }

    const hasActiveFilters = computed(() =>
      searchQuery.value.trim() !== '' || activePreset.value !== 'all'
    );

    // Computed: apply filters + sort
    const filteredSessions = computed(() => {
      let result = [...sessions.value];
      const preset = FILTER_PRESETS.find(p => p.id === activePreset.value);
      const filters = preset ? preset.filters : {};

      // Source filter
      if (filters.source) {
        result = result.filter(s => s.source === filters.source);
      }
      // Min messages
      if (filters.minMessages) {
        result = result.filter(s => s.message_count >= filters.minMessages);
      }
      // Compacted
      if (filters.hasCompaction) {
        result = result.filter(s => s.has_summary);
      }
      // Recently active (within maxAge seconds)
      if (filters.maxAge != null) {
        const now = Date.now() / 1000;
        result = result.filter(s => s.last_active && (now - s.last_active) <= filters.maxAge);
      }

      // Search
      if (searchQuery.value.trim()) {
        const q = searchQuery.value.toLowerCase().trim();
        result = result.filter(s =>
          (s.channel_id || '').toLowerCase().includes(q) ||
          (s.last_user_id || '').toLowerCase().includes(q) ||
          (s.source || '').toLowerCase().includes(q)
        );
      }

      // Sort
      const key = sortBy.value;
      const dir = sortAsc.value ? 1 : -1;
      result.sort((a, b) => {
        const av = a[key] || 0;
        const bv = b[key] || 0;
        return (av - bv) * dir;
      });

      return result;
    });

    // Thread grouping: group messages into user→assistant turns
    const threads = computed(() => {
      if (!detail.value || !detail.value.messages) return [];
      const msgs = detail.value.messages;
      if (msgs.length === 0) return [];

      const groups = [];
      let current = [];

      for (const m of msgs) {
        if (m.role === 'user' && current.length > 0) {
          groups.push(current);
          current = [];
        }
        current.push(m);
      }
      if (current.length > 0) groups.push(current);
      return groups;
    });

    const allSelected = computed(() =>
      filteredSessions.value.length > 0 && selected.value.size === filteredSessions.value.length
    );

    function threadSummary(thread) {
      const userMsg = thread.find(m => m.role === 'user');
      if (userMsg && userMsg.content) {
        const text = userMsg.content.slice(0, 120);
        return text.length < userMsg.content.length ? text + '...' : text;
      }
      return '(no user message)';
    }

    function toggleThread(index) {
      const s = new Set(collapsedThreads.value);
      if (s.has(index)) s.delete(index);
      else s.add(index);
      collapsedThreads.value = s;
    }

    function applyPreset(presetId) {
      activePreset.value = presetId;
    }

    function applyCustomPreset(cp) {
      activePreset.value = cp.id;
      if (cp.filters.searchQuery != null) searchQuery.value = cp.filters.searchQuery;
      if (cp.filters.sortBy) sortBy.value = cp.filters.sortBy;
    }

    function saveCustomPreset() {
      if (!newPresetName.value.trim()) return;
      const preset = {
        id: 'custom-' + Date.now(),
        name: newPresetName.value.trim(),
        filters: {
          searchQuery: searchQuery.value,
          sortBy: sortBy.value,
        },
      };
      customPresets.value = [...customPresets.value, preset];
      saveCustomPresetsToStorage();
      showSavePreset.value = false;
      newPresetName.value = '';
    }

    function removeCustomPreset(id) {
      customPresets.value = customPresets.value.filter(p => p.id !== id);
      saveCustomPresetsToStorage();
      if (activePreset.value === id) activePreset.value = 'all';
    }

    function resetFilters() {
      activePreset.value = 'all';
      searchQuery.value = '';
      sortBy.value = 'last_active';
      sortAsc.value = false;
    }

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

    function threadMsgClass(role) {
      if (role === 'user') return 'sess-msg-user';
      if (role === 'assistant') return 'sess-msg-assistant';
      return 'sess-msg-system';
    }

    function roleBadge(role) {
      if (role === 'user') return 'badge-info';
      if (role === 'assistant') return 'badge-success';
      return 'badge-warning';
    }

    function roleDotClass(role) {
      if (role === 'user') return 'sess-dot-user';
      if (role === 'assistant') return 'sess-dot-assistant';
      return 'sess-dot-system';
    }

    function roleLabelClass(role) {
      if (role === 'user') return 'text-cyan-400';
      if (role === 'assistant') return 'text-indigo-400';
      return 'text-gray-500';
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
        collapsedThreads.value = new Set();
        return;
      }
      expandedId.value = channelId;
      detail.value = null;
      detailLoading.value = true;
      collapsedThreads.value = new Set();
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
        selected.value = new Set(filteredSessions.value.map(s => s.channel_id));
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
            api.get(`/api/sessions/${expandedId.value}`)
              .then(d => { detail.value = d; })
              .catch(() => {});
          }
        }, 2000);
      }
    }

    onMounted(() => {
      loadCustomPresets();
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
      // Filter/sort
      activePreset, searchQuery, sortBy, sortAsc,
      filterPresets, sortOptions,
      filteredSessions, hasActiveFilters,
      // Custom presets
      customPresets, showSavePreset, newPresetName,
      // Thread view
      threadView, threads, collapsedThreads,
      // Methods
      formatAge, formatTimestamp, formatFullTimestamp,
      messageClass, threadMsgClass, roleBadge, roleDotClass, roleLabelClass,
      truncateContent, threadSummary,
      fetchSessions, retry, toggleSession,
      toggleSelect, toggleSelectAll,
      confirmClear, clearSession,
      confirmBulkClear, doBulkClear,
      exportSession,
      applyPreset, applyCustomPreset, saveCustomPreset, removeCustomPreset,
      resetFilters, toggleThread,
    };
  },
};
