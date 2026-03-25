/**
 * Loki Management UI — Logs Page
 * Live log tail via WebSocket with filtering and auto-scroll
 */
import { api, ws } from '../api.js';

const { ref, computed, onMounted, onUnmounted, nextTick, watch } = Vue;

export default {
  template: `
    <div class="p-6 flex flex-col" style="height: calc(100vh - 56px);">
      <div class="flex items-center justify-between mb-3">
        <h1 class="text-xl font-semibold">Logs</h1>
        <div class="flex gap-2 items-center">
          <button @click="togglePause" class="btn text-xs" :class="paused ? 'btn-primary' : 'btn-ghost'">
            {{ paused ? 'Resume' : 'Pause' }}
          </button>
          <button @click="clearLogs" class="btn btn-ghost text-xs">Clear</button>
          <button @click="exportLogs" class="btn btn-ghost text-xs">Export</button>
        </div>
      </div>

      <!-- Filters -->
      <div class="flex gap-2 mb-3 flex-wrap">
        <select v-model="levelFilter" class="loki-input" style="width:auto;min-width:100px;">
          <option value="">All Levels</option>
          <option value="INFO">INFO</option>
          <option value="WARNING">WARNING</option>
          <option value="ERROR">ERROR</option>
        </select>
        <input v-model="textFilter" type="text" class="loki-input flex-1"
               placeholder="Filter logs..." style="min-width:150px;" />
        <label class="flex items-center gap-1.5 text-xs text-gray-400 select-none cursor-pointer">
          <input type="checkbox" v-model="autoScroll" class="rounded" />
          Auto-scroll
        </label>
      </div>

      <!-- Status bar -->
      <div class="flex items-center gap-3 mb-2 text-xs text-gray-500">
        <div class="flex items-center gap-1.5">
          <span class="status-dot" :class="subscribed ? 'online' : 'offline'"></span>
          {{ subscribed ? 'Live' : 'Disconnected' }}
        </div>
        <span>{{ filteredLogs.length }} / {{ logs.length }} entries</span>
        <span v-if="paused" class="badge badge-warning">Paused</span>
      </div>

      <!-- Log output -->
      <div ref="logContainer"
           class="flex-1 overflow-y-auto bg-gray-950 border border-gray-800 rounded p-3 font-mono text-xs"
           style="min-height:200px;">
        <div v-if="filteredLogs.length === 0" class="text-gray-500 text-center py-8">
          {{ logs.length === 0 ? 'Waiting for log entries...' : 'No entries match the current filter' }}
        </div>
        <div v-for="(entry, i) in filteredLogs" :key="i"
             class="py-0.5 leading-relaxed whitespace-pre-wrap break-all"
             :class="logClass(entry)">
          <span class="text-gray-600">{{ entry.ts || '' }}</span>
          <span class="mx-1" :class="levelClass(entry.level)">{{ entry.level || 'INFO' }}</span>
          <span>{{ entry.text || entry.raw || '' }}</span>
        </div>
      </div>
    </div>`,

  setup() {
    const logs = ref([]);
    const paused = ref(false);
    const autoScroll = ref(true);
    const levelFilter = ref('');
    const textFilter = ref('');
    const subscribed = ref(false);
    const logContainer = ref(null);
    const MAX_LOGS = 2000;

    // Buffer entries while paused
    const pauseBuffer = ref([]);

    const filteredLogs = computed(() => {
      let result = logs.value;
      if (levelFilter.value) {
        result = result.filter(e => (e.level || 'INFO') === levelFilter.value);
      }
      if (textFilter.value) {
        const q = textFilter.value.toLowerCase();
        result = result.filter(e => {
          const text = (e.text || e.raw || '').toLowerCase();
          const tool = (e.tool || '').toLowerCase();
          return text.includes(q) || tool.includes(q);
        });
      }
      return result;
    });

    function parseLogEntry(data) {
      // WebSocket log data may be a parsed audit entry or raw text
      if (data.payload) {
        const p = data.payload;
        return {
          ts: p.timestamp ? new Date(p.timestamp).toLocaleTimeString() : new Date().toLocaleTimeString(),
          level: p.level || (p.error ? 'ERROR' : 'INFO'),
          text: p.tool ? `[${p.tool}] ${p.summary || p.output || ''}`.trim() : (p.message || p.summary || JSON.stringify(p)),
          tool: p.tool || '',
          raw: typeof data.payload === 'string' ? data.payload : null,
        };
      }
      // Raw string
      if (typeof data === 'string') {
        return { ts: new Date().toLocaleTimeString(), level: 'INFO', text: data, tool: '', raw: data };
      }
      return {
        ts: new Date().toLocaleTimeString(),
        level: 'INFO',
        text: JSON.stringify(data),
        tool: '',
        raw: null,
      };
    }

    function onLog(data) {
      const entry = parseLogEntry(data);
      if (paused.value) {
        pauseBuffer.value.push(entry);
        return;
      }
      addEntry(entry);
    }

    function addEntry(entry) {
      logs.value.push(entry);
      if (logs.value.length > MAX_LOGS) {
        logs.value = logs.value.slice(-MAX_LOGS);
      }
      if (autoScroll.value) {
        nextTick(() => scrollToBottom());
      }
    }

    function scrollToBottom() {
      const el = logContainer.value;
      if (el) el.scrollTop = el.scrollHeight;
    }

    function togglePause() {
      paused.value = !paused.value;
      if (!paused.value && pauseBuffer.value.length > 0) {
        // Flush buffered entries
        for (const entry of pauseBuffer.value) {
          addEntry(entry);
        }
        pauseBuffer.value = [];
      }
    }

    function clearLogs() {
      logs.value = [];
      pauseBuffer.value = [];
    }

    function exportLogs() {
      const text = filteredLogs.value.map(e => `${e.ts} ${e.level} ${e.text}`).join('\n');
      const blob = new Blob([text], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `loki-logs-${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.txt`;
      a.click();
      URL.revokeObjectURL(url);
    }

    function logClass(entry) {
      if (entry.level === 'ERROR') return 'text-red-400';
      if (entry.level === 'WARNING') return 'text-yellow-400';
      return 'text-gray-300';
    }

    function levelClass(level) {
      if (level === 'ERROR') return 'text-red-500 font-semibold';
      if (level === 'WARNING') return 'text-yellow-500';
      return 'text-blue-500';
    }

    onMounted(() => {
      ws.subscribe('logs', onLog);
      subscribed.value = ws.connected;
      ws.onStatusChange = (connected) => { subscribed.value = connected; };
    });

    onUnmounted(() => {
      ws.unsubscribe('logs', onLog);
    });

    return {
      logs, paused, autoScroll, levelFilter, textFilter,
      subscribed, logContainer, filteredLogs,
      togglePause, clearLogs, exportLogs, logClass, levelClass,
    };
  },
};
