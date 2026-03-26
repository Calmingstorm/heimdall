/**
 * Loki Management UI — Logs Page
 * Live log tail via WebSocket with filtering and auto-scroll
 */
import { api, ws } from '../api.js';

const { ref, computed, onMounted, onUnmounted, nextTick, watch } = Vue;

const LOG_LEVELS = ['INFO', 'WARNING', 'ERROR'];

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
      <div class="flex gap-2 mb-3 flex-wrap items-center">
        <!-- Level chips -->
        <div class="flex gap-1">
          <button v-for="lvl in levels" :key="lvl"
                  @click="toggleLevel(lvl)"
                  class="log-chip"
                  :class="[levelChipClass(lvl), { 'log-chip-active': levelFilter === lvl }]">
            {{ lvl }}
          </button>
          <button v-if="levelFilter" @click="levelFilter = ''" class="log-chip log-chip-clear">ALL</button>
        </div>

        <div class="flex-1" style="min-width:0;">
          <div class="flex gap-1.5 items-center">
            <input v-model="textFilter" type="text" class="loki-input flex-1"
                   :placeholder="useRegex ? 'Regex pattern...' : 'Filter logs...'"
                   :class="{ 'border-red-700': regexError }"
                   style="min-width:120px;" />
            <button @click="useRegex = !useRegex" class="btn text-xs"
                    :class="useRegex ? 'btn-primary' : 'btn-ghost'"
                    title="Toggle regex filtering">.*</button>
          </div>
          <div v-if="regexError" class="text-red-400 text-xs mt-0.5">{{ regexError }}</div>
        </div>

        <label class="flex items-center gap-1.5 text-xs text-gray-400 select-none cursor-pointer flex-shrink-0">
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
        <span class="font-mono">{{ filteredLogs.length.toLocaleString() }} / {{ logs.length.toLocaleString() }} lines</span>
        <span v-if="paused" class="badge badge-warning">Paused ({{ pauseBuffer.length }} buffered)</span>
        <span v-if="copiedIndex !== null" class="text-green-400">Copied!</span>
      </div>

      <!-- Log output -->
      <div class="relative flex-1" style="min-height:200px;">
        <div ref="logContainer" @scroll="onScroll"
             class="absolute inset-0 overflow-y-auto bg-gray-950 border border-gray-800 rounded p-3 font-mono text-xs">
          <div v-if="filteredLogs.length === 0" class="text-gray-500 text-center py-8">
            {{ logs.length === 0 ? 'Waiting for log entries...' : 'No entries match the current filter' }}
          </div>
          <div v-for="(entry, i) in filteredLogs" :key="i"
               class="log-line py-0.5 leading-relaxed whitespace-pre-wrap break-all"
               :class="logLineClass(entry)">
            <span class="log-ts text-gray-600 cursor-pointer hover:text-gray-400"
                  @click="copyLine(entry, i)"
                  title="Click to copy line">{{ entry.ts || '' }}</span>
            <span class="log-level mx-1" :class="levelClass(entry.level)">{{ entry.level || 'INFO' }}</span>
            <span>{{ entry.text || entry.raw || '' }}</span>
          </div>
        </div>

        <!-- Jump to bottom -->
        <button v-if="showJumpBottom" @click="jumpToBottom"
                class="log-jump-btn">
          &#x2193; Jump to bottom
        </button>
      </div>
    </div>`,

  setup() {
    const logs = ref([]);
    const paused = ref(false);
    const autoScroll = ref(true);
    const levelFilter = ref('');
    const textFilter = ref('');
    const useRegex = ref(false);
    const subscribed = ref(false);
    const logContainer = ref(null);
    const showJumpBottom = ref(false);
    const copiedIndex = ref(null);
    const MAX_LOGS = 2000;
    const levels = LOG_LEVELS;

    // Buffer entries while paused
    const pauseBuffer = ref([]);

    const regexError = computed(() => {
      if (!useRegex.value || !textFilter.value) return null;
      try {
        new RegExp(textFilter.value, 'i');
        return null;
      } catch (e) {
        return e.message;
      }
    });

    const filteredLogs = computed(() => {
      let result = logs.value;
      if (levelFilter.value) {
        result = result.filter(e => (e.level || 'INFO') === levelFilter.value);
      }
      if (textFilter.value && !regexError.value) {
        if (useRegex.value) {
          try {
            const re = new RegExp(textFilter.value, 'i');
            result = result.filter(e => {
              const text = (e.text || e.raw || '');
              const tool = (e.tool || '');
              return re.test(text) || re.test(tool);
            });
          } catch { /* invalid regex, skip filtering */ }
        } else {
          const q = textFilter.value.toLowerCase();
          result = result.filter(e => {
            const text = (e.text || e.raw || '').toLowerCase();
            const tool = (e.tool || '').toLowerCase();
            return text.includes(q) || tool.includes(q);
          });
        }
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
      if (el) {
        const distFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
        el.scrollTo({ top: el.scrollHeight, behavior: distFromBottom < 500 ? 'smooth' : 'instant' });
      }
    }

    function jumpToBottom() {
      autoScroll.value = true;
      showJumpBottom.value = false;
      nextTick(() => scrollToBottom());
    }

    function onScroll() {
      const el = logContainer.value;
      if (!el) return;
      const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
      showJumpBottom.value = !atBottom && logs.value.length > 0;
      // Auto-disable auto-scroll when user scrolls up
      if (!atBottom && autoScroll.value) {
        autoScroll.value = false;
      }
    }

    function togglePause() {
      paused.value = !paused.value;
      if (!paused.value && pauseBuffer.value.length > 0) {
        for (const entry of pauseBuffer.value) {
          addEntry(entry);
        }
        pauseBuffer.value = [];
      }
    }

    function clearLogs() {
      logs.value = [];
      pauseBuffer.value = [];
      showJumpBottom.value = false;
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

    function copyLine(entry, index) {
      const line = `${entry.ts} ${entry.level} ${entry.text || entry.raw || ''}`;
      navigator.clipboard.writeText(line).then(() => {
        copiedIndex.value = index;
        setTimeout(() => { copiedIndex.value = null; }, 1500);
      }).catch(() => {});
    }

    function toggleLevel(lvl) {
      levelFilter.value = levelFilter.value === lvl ? '' : lvl;
    }

    function logLineClass(entry) {
      if (entry.level === 'ERROR') return 'log-line-error';
      if (entry.level === 'WARNING') return 'log-line-warning';
      return 'text-gray-300';
    }

    function levelClass(level) {
      if (level === 'ERROR') return 'text-red-500 font-semibold';
      if (level === 'WARNING') return 'text-yellow-500';
      return 'text-blue-500';
    }

    function levelChipClass(lvl) {
      if (lvl === 'ERROR') return 'log-chip-error';
      if (lvl === 'WARNING') return 'log-chip-warning';
      return 'log-chip-info';
    }

    // Track WS connection status without overriding global handler
    let statusCheckInterval = null;

    onMounted(() => {
      ws.subscribe('logs', onLog);
      subscribed.value = ws.connected;
      statusCheckInterval = setInterval(() => {
        subscribed.value = ws.connected;
      }, 2000);
    });

    onUnmounted(() => {
      ws.unsubscribe('logs', onLog);
      if (statusCheckInterval) clearInterval(statusCheckInterval);
    });

    return {
      logs, paused, autoScroll, levelFilter, textFilter, useRegex,
      subscribed, logContainer, filteredLogs, pauseBuffer,
      showJumpBottom, copiedIndex, regexError, levels,
      togglePause, clearLogs, exportLogs, logLineClass, levelClass,
      levelChipClass, toggleLevel, copyLine, jumpToBottom, onScroll,
    };
  },
};
