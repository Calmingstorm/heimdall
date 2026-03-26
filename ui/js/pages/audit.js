/**
 * Loki Management UI — Audit Page
 * Searchable tool execution audit log with filters and pagination
 */
import { api } from '../api.js';

const { ref, computed, onMounted } = Vue;

export default {
  template: `
    <div class="p-6 page-fade-in">
      <div class="flex items-center justify-between mb-4">
        <h1 class="text-xl font-semibold">Audit Log</h1>
        <button @click="fetchAudit" class="btn btn-ghost text-xs" :disabled="loading">
          {{ loading ? 'Loading...' : 'Refresh' }}
        </button>
      </div>

      <!-- Filters -->
      <div class="loki-card mb-4">
        <div class="grid grid-cols-1 md:grid-cols-4 gap-3">
          <div>
            <label class="text-gray-400 text-xs block mb-1">Tool</label>
            <input v-model="filters.tool" type="text" class="loki-input"
                   placeholder="e.g. run_command" @keyup.enter="fetchAudit" />
          </div>
          <div>
            <label class="text-gray-400 text-xs block mb-1">User</label>
            <input v-model="filters.user" type="text" class="loki-input"
                   placeholder="User ID or name" @keyup.enter="fetchAudit" />
          </div>
          <div>
            <label class="text-gray-400 text-xs block mb-1">Keyword</label>
            <input v-model="filters.keyword" type="text" class="loki-input"
                   placeholder="Search in output..." @keyup.enter="fetchAudit" />
          </div>
          <div>
            <label class="text-gray-400 text-xs block mb-1">Date</label>
            <input v-model="filters.date" type="date" class="loki-input" @change="fetchAudit" />
          </div>
        </div>
        <div class="flex gap-2 mt-3">
          <button @click="fetchAudit" class="btn btn-primary text-xs">Search</button>
          <button @click="clearFilters" class="btn btn-ghost text-xs">Clear Filters</button>
          <div class="flex-1"></div>
          <div class="flex items-center gap-2">
            <label class="text-gray-400 text-xs">Limit:</label>
            <select v-model="filters.limit" class="loki-input" style="width:auto;min-width:70px;" @change="fetchAudit">
              <option :value="25">25</option>
              <option :value="50">50</option>
              <option :value="100">100</option>
              <option :value="200">200</option>
            </select>
          </div>
        </div>
      </div>

      <!-- Results -->
      <div v-if="loading && entries.length === 0" class="space-y-2">
        <div v-for="n in 5" :key="n" class="skeleton skeleton-row"></div>
      </div>
      <div v-else-if="error" class="loki-card border-red-900 error-state">
        <span class="error-icon">\u26A0</span>
        <p class="text-red-400">{{ error }}</p>
        <button @click="fetchAudit" class="btn btn-ghost text-xs">Retry</button>
      </div>
      <div v-else-if="entries.length === 0" class="loki-card empty-state">
        <span class="empty-state-icon">\u{1F4DD}</span>
        <span class="empty-state-text">No audit entries found</span>
        <span class="empty-state-hint">Try adjusting your filters or wait for tool executions to appear</span>
      </div>
      <div v-else>
        <div class="text-xs text-gray-500 mb-2">Showing {{ entries.length }} entries</div>
        <div class="table-responsive">
        <table class="loki-table">
          <thead>
            <tr>
              <th>Timestamp</th>
              <th>Tool</th>
              <th>User</th>
              <th>Host</th>
              <th>Duration</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(e, i) in entries" :key="i"
                @click="toggleExpand(i)" style="cursor:pointer;"
                :class="expandedIdx === i ? 'bg-gray-800/50' : ''">
              <td class="text-xs text-gray-400 font-mono whitespace-nowrap">{{ formatTs(e.timestamp) }}</td>
              <td class="font-mono text-xs">{{ e.tool || e.tool_name || '—' }}</td>
              <td class="text-xs text-gray-400">{{ e.user || e.user_id || '—' }}</td>
              <td class="text-xs text-gray-400 font-mono">{{ e.host || '—' }}</td>
              <td class="text-xs text-gray-400">
                {{ e.duration ? (e.duration < 1 ? (e.duration * 1000).toFixed(0) + 'ms' : e.duration.toFixed(1) + 's') : '—' }}
              </td>
              <td>
                <span v-if="e.error" class="badge badge-danger">error</span>
                <span v-else class="badge badge-success">ok</span>
              </td>
            </tr>
          </tbody>
        </table>
        </div>

        <!-- Expanded detail -->
        <div v-if="expandedIdx !== null && entries[expandedIdx]" class="mt-3 loki-card">
          <div class="flex items-center justify-between mb-2">
            <span class="text-sm font-medium font-mono">{{ entries[expandedIdx].tool || entries[expandedIdx].tool_name }}</span>
            <button @click="expandedIdx = null" class="btn btn-ghost text-xs">Close</button>
          </div>

          <div v-if="entries[expandedIdx].input || entries[expandedIdx].tool_input" class="mb-3">
            <div class="text-gray-400 text-xs mb-1">Input</div>
            <pre class="p-2 rounded bg-gray-900 text-xs text-gray-300 overflow-x-auto font-mono max-h-40 overflow-y-auto">{{ formatDetail(entries[expandedIdx].input || entries[expandedIdx].tool_input) }}</pre>
          </div>

          <div v-if="entries[expandedIdx].output || entries[expandedIdx].result">
            <div class="text-gray-400 text-xs mb-1">Output</div>
            <pre class="p-2 rounded bg-gray-900 text-xs text-gray-300 overflow-x-auto font-mono max-h-60 overflow-y-auto whitespace-pre-wrap break-all">{{ truncate(formatDetail(entries[expandedIdx].output || entries[expandedIdx].result), 5000) }}</pre>
          </div>

          <div v-if="entries[expandedIdx].error" class="mt-2">
            <div class="text-red-400 text-xs mb-1">Error</div>
            <pre class="p-2 rounded bg-red-950/30 text-xs text-red-300 overflow-x-auto font-mono">{{ entries[expandedIdx].error }}</pre>
          </div>
        </div>
      </div>
    </div>`,

  setup() {
    const entries = ref([]);
    const loading = ref(true);
    const error = ref(null);
    const expandedIdx = ref(null);

    const filters = ref({
      tool: '',
      user: '',
      keyword: '',
      date: '',
      limit: 50,
    });

    function formatTs(ts) {
      if (!ts) return '—';
      try {
        const d = new Date(ts);
        if (isNaN(d.getTime())) return ts;
        return d.toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', second: '2-digit' });
      } catch { return ts; }
    }

    function formatDetail(obj) {
      if (!obj) return '';
      if (typeof obj === 'string') return obj;
      try {
        return JSON.stringify(obj, null, 2);
      } catch {
        return String(obj);
      }
    }

    function truncate(text, max) {
      if (!text) return '';
      return text.length > max ? text.slice(0, max) + '\n... (truncated)' : text;
    }

    function toggleExpand(idx) {
      expandedIdx.value = expandedIdx.value === idx ? null : idx;
    }

    function clearFilters() {
      filters.value = { tool: '', user: '', keyword: '', date: '', limit: 50 };
      fetchAudit();
    }

    async function fetchAudit() {
      loading.value = true;
      error.value = null;
      expandedIdx.value = null;
      try {
        const params = new URLSearchParams();
        if (filters.value.tool) params.set('tool', filters.value.tool);
        if (filters.value.user) params.set('user', filters.value.user);
        if (filters.value.keyword) params.set('q', filters.value.keyword);
        if (filters.value.date) params.set('date', filters.value.date);
        params.set('limit', String(filters.value.limit));
        const qs = params.toString();
        const data = await api.get(`/api/audit${qs ? '?' + qs : ''}`);
        entries.value = Array.isArray(data) ? data : [];
      } catch (e) {
        error.value = e.message;
      }
      loading.value = false;
    }

    onMounted(() => { fetchAudit(); });

    return {
      entries, loading, error, expandedIdx, filters,
      formatTs, formatDetail, truncate, toggleExpand, clearFilters, fetchAudit,
    };
  },
};
