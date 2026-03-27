/**
 * Heimdall Management UI — Logs Page (Redesigned)
 * Live log tail with timeline visualization, filter presets, time range filtering
 */
import { api, ws } from '../api.js';

const { ref, computed, onMounted, onUnmounted, nextTick, watch } = Vue;

const LOG_LEVELS = ['INFO', 'WARNING', 'ERROR'];

const LOG_PRESETS = [
  { id: 'all', name: 'All Logs', icon: '\u2630', filters: {} },
  { id: 'errors', name: 'Errors Only', icon: '\u274C', filters: { level: 'ERROR' } },
  { id: 'warnings', name: 'Warnings+', icon: '\u26A0', filters: { levels: ['WARNING', 'ERROR'] } },
  { id: 'tools', name: 'Tool Activity', icon: '\u{1F527}', filters: { hasToolName: true } },
  { id: 'recent-errors', name: 'Recent Errors', icon: '\u{1F525}', filters: { level: 'ERROR', timeRange: 'last_1h' } },
];

const TIME_RANGES = [
  { value: '', label: 'All Time' },
  { value: 'last_5m', label: 'Last 5 min', seconds: 300 },
  { value: 'last_15m', label: 'Last 15 min', seconds: 900 },
  { value: 'last_1h', label: 'Last 1 hour', seconds: 3600 },
  { value: 'last_4h', label: 'Last 4 hours', seconds: 14400 },
  { value: 'last_24h', label: 'Last 24 hours', seconds: 86400 },
];

export default {
  template: `
    <div class="p-6 page-fade-in flex flex-col" style="height: calc(100vh - 56px);">
      <div class="flex items-center justify-between mb-3 flex-wrap gap-2">
        <div>
          <h1 class="text-xl font-semibold">Logs</h1>
          <p class="text-xs text-gray-500 mt-0.5" v-if="logs.length > 0">
            {{ filteredLogs.length.toLocaleString() }} / {{ logs.length.toLocaleString() }} entries
          </p>
        </div>
        <div class="flex gap-2 items-center">
          <button @click="togglePause" class="btn text-xs" :class="paused ? 'btn-primary' : 'btn-ghost'">
            {{ paused ? 'Resume' : 'Pause' }}
          </button>
          <button @click="clearLogs" class="btn btn-ghost text-xs">Clear</button>
          <button @click="exportLogs" class="btn btn-ghost text-xs">Export</button>
        </div>
      </div>

      <!-- Filter presets bar -->
      <div class="logs-filter-bar mb-2">
        <div class="flex gap-1.5 flex-wrap items-center">
          <button v-for="preset in logPresets" :key="preset.id"
                  @click="applyLogPreset(preset)"
                  class="sess-preset-chip"
                  :class="{ 'sess-preset-active': activeLogPreset === preset.id }">
            <span class="sess-preset-icon">{{ preset.icon }}</span>
            <span>{{ preset.name }}</span>
          </button>
        </div>
      </div>

      <!-- Filters row -->
      <div class="flex gap-2 mb-2 flex-wrap items-center">
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

        <!-- Time range -->
        <select v-model="timeRange" class="hm-select text-xs">
          <option v-for="tr in timeRanges" :key="tr.value" :value="tr.value">{{ tr.label }}</option>
        </select>

        <div class="flex-1" style="min-width:0;">
          <div class="flex gap-1.5 items-center">
            <input v-model="textFilter" type="text" class="hm-input flex-1"
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

      <!-- Custom preset save bar -->
      <div class="flex gap-2 items-center mb-2 flex-wrap">
        <button v-if="hasActiveLogFilters" @click="showSaveLogPreset = !showSaveLogPreset"
                class="btn btn-ghost text-xs">Save as preset</button>
        <template v-if="showSaveLogPreset">
          <input v-model="newLogPresetName" type="text" class="hm-input text-xs"
                 placeholder="Preset name..." style="max-width: 180px;" />
          <button @click="saveLogCustomPreset" class="btn btn-primary text-xs"
                  :disabled="!newLogPresetName.trim()">Save</button>
        </template>
        <!-- Custom presets -->
        <button v-for="cp in customLogPresets" :key="cp.id"
                @click="applyCustomLogPreset(cp)"
                class="sess-preset-chip sess-preset-custom"
                :class="{ 'sess-preset-active': activeLogPreset === cp.id }">
          <span>\u2605</span>
          <span>{{ cp.name }}</span>
          <span class="sess-preset-remove" @click.stop="removeLogCustomPreset(cp.id)">&times;</span>
        </button>
      </div>

      <!-- Timeline visualization -->
      <div v-if="logs.length > 0" class="logs-timeline mb-2">
        <div class="logs-timeline-header">
          <span class="text-xs text-gray-500">Activity Timeline</span>
          <span class="text-xs text-gray-600">{{ timelineSpanLabel }}</span>
        </div>
        <div class="logs-timeline-chart">
          <div v-for="(bucket, bi) in timelineBuckets" :key="bi"
               class="logs-timeline-bar-wrap"
               :title="bucket.label + ': ' + bucket.total + ' entries'"
               @click="jumpToTimelineBucket(bucket)">
            <div class="logs-timeline-bar">
              <div v-if="bucket.errors > 0" class="logs-timeline-segment logs-tl-error"
                   :style="{ height: segmentHeight(bucket.errors, timelineMax) }"></div>
              <div v-if="bucket.warnings > 0" class="logs-timeline-segment logs-tl-warning"
                   :style="{ height: segmentHeight(bucket.warnings, timelineMax) }"></div>
              <div v-if="bucket.info > 0" class="logs-timeline-segment logs-tl-info"
                   :style="{ height: segmentHeight(bucket.info, timelineMax) }"></div>
            </div>
            <span class="logs-timeline-label" v-if="bi % timelineLabelSkip === 0">{{ bucket.shortLabel }}</span>
          </div>
        </div>
      </div>

      <!-- Status bar -->
      <div class="flex items-center gap-3 mb-2 text-xs text-gray-500 flex-wrap">
        <div class="flex items-center gap-1.5">
          <span class="status-dot" :class="subscribed ? 'online' : 'offline'"></span>
          {{ subscribed ? 'Live' : 'Disconnected' }}
        </div>
        <span class="font-mono">{{ filteredLogs.length.toLocaleString() }} / {{ logs.length.toLocaleString() }} lines</span>
        <span v-if="paused" class="badge badge-warning">Paused ({{ pauseBuffer.length }} buffered)</span>
        <span v-if="timeRange" class="badge badge-info">{{ timeRangeLabel }}</span>
        <span v-if="copiedIndex !== null" class="text-green-400">Copied!</span>
      </div>

      <!-- Log output -->
      <div class="relative flex-1" style="min-height:200px;">
        <div ref="logContainer" @scroll="onScroll"
             class="absolute inset-0 overflow-y-auto bg-gray-950 border border-gray-800 rounded p-3 font-mono text-xs">
          <div v-if="filteredLogs.length === 0" class="empty-state" style="padding:2rem 0;">
            <span class="empty-state-icon">{{ logs.length === 0 ? '\u{1F4C4}' : '\u{1F50D}' }}</span>
            <span class="empty-state-text">{{ logs.length === 0 ? 'Waiting for log entries...' : 'No entries match the current filter' }}</span>
          </div>
          <div v-for="(entry, i) in filteredLogs" :key="i"
               class="log-line py-0.5 leading-relaxed whitespace-pre-wrap break-all"
               :class="logLineClass(entry)">
            <span class="log-ts text-gray-600 cursor-pointer hover:text-gray-400"
                  @click="copyLine(entry, i)"
                  title="Click to copy line">{{ entry.ts || '' }}</span>
            <span class="log-level mx-1" :class="levelClass(entry.level)">{{ entry.level || 'INFO' }}</span>
            <span v-if="entry.tool" class="logs-tool-badge">{{ entry.tool }}</span>
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
    const logPresets = LOG_PRESETS;
    const timeRanges = TIME_RANGES;

    // Filter presets state
    const activeLogPreset = ref('all');
    const timeRange = ref('');
    const customLogPresets = ref([]);
    const showSaveLogPreset = ref(false);
    const newLogPresetName = ref('');

    // Buffer entries while paused
    const pauseBuffer = ref([]);

    // Load custom presets
    function loadCustomLogPresets() {
      try {
        const saved = localStorage.getItem('heimdall-log-presets');
        if (saved) customLogPresets.value = JSON.parse(saved);
      } catch { /* ignore */ }
    }

    function saveCustomLogPresetsToStorage() {
      try {
        localStorage.setItem('heimdall-log-presets', JSON.stringify(customLogPresets.value));
      } catch { /* ignore */ }
    }

    const hasActiveLogFilters = computed(() =>
      levelFilter.value !== '' || textFilter.value.trim() !== '' || timeRange.value !== ''
    );

    const timeRangeLabel = computed(() => {
      const tr = TIME_RANGES.find(t => t.value === timeRange.value);
      return tr ? tr.label : '';
    });

    const regexError = computed(() => {
      if (!useRegex.value || !textFilter.value) return null;
      try {
        new RegExp(textFilter.value, 'i');
        return null;
      } catch (e) {
        return e.message;
      }
    });

    // Timeline: bucket logs by time intervals
    const TIMELINE_BUCKETS = 24;
    const timelineBuckets = computed(() => {
      if (logs.value.length === 0) return [];
      const buckets = [];
      const now = new Date();
      const spanMs = 3600 * 1000; // 1 hour per bucket

      for (let i = TIMELINE_BUCKETS - 1; i >= 0; i--) {
        const start = new Date(now.getTime() - (i + 1) * spanMs);
        const end = new Date(now.getTime() - i * spanMs);
        buckets.push({
          start, end,
          label: formatBucketLabel(start, end),
          shortLabel: end.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          total: 0, info: 0, warnings: 0, errors: 0,
        });
      }

      for (const entry of logs.value) {
        if (!entry._time) continue;
        const t = entry._time.getTime();
        for (const b of buckets) {
          if (t >= b.start.getTime() && t < b.end.getTime()) {
            b.total++;
            if (entry.level === 'ERROR') b.errors++;
            else if (entry.level === 'WARNING') b.warnings++;
            else b.info++;
            break;
          }
        }
      }
      return buckets;
    });

    const timelineMax = computed(() => {
      let max = 1;
      for (const b of timelineBuckets.value) {
        if (b.total > max) max = b.total;
      }
      return max;
    });

    const timelineSpanLabel = computed(() => {
      if (timelineBuckets.value.length === 0) return '';
      return 'Last 24 hours';
    });

    const timelineLabelSkip = computed(() => {
      // Show every Nth label to avoid crowding
      return TIMELINE_BUCKETS <= 12 ? 1 : Math.ceil(TIMELINE_BUCKETS / 8);
    });

    function formatBucketLabel(start, end) {
      const fmt = { hour: '2-digit', minute: '2-digit' };
      return start.toLocaleTimeString([], fmt) + ' - ' + end.toLocaleTimeString([], fmt);
    }

    function segmentHeight(count, max) {
      if (!max || !count) return '0px';
      const pct = Math.max(2, (count / max) * 100);
      return pct + '%';
    }

    function jumpToTimelineBucket(bucket) {
      // Find first log entry in this bucket and scroll to it
      const idx = filteredLogs.value.findIndex(e =>
        e._time && e._time.getTime() >= bucket.start.getTime() && e._time.getTime() < bucket.end.getTime()
      );
      if (idx >= 0 && logContainer.value) {
        const lines = logContainer.value.querySelectorAll('.log-line');
        if (lines[idx]) {
          lines[idx].scrollIntoView({ behavior: 'smooth', block: 'center' });
          autoScroll.value = false;
        }
      }
    }

    const filteredLogs = computed(() => {
      let result = logs.value;

      // Level filter
      if (levelFilter.value) {
        result = result.filter(e => (e.level || 'INFO') === levelFilter.value);
      }

      // Time range filter
      if (timeRange.value) {
        const tr = TIME_RANGES.find(t => t.value === timeRange.value);
        if (tr && tr.seconds) {
          const cutoff = new Date(Date.now() - tr.seconds * 1000);
          result = result.filter(e => e._time && e._time >= cutoff);
        }
      }

      // Text filter
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
      if (data.type === 'log' && data.line) {
        try {
          const entry = typeof data.line === 'string' ? JSON.parse(data.line) : data.line;
          const time = entry.timestamp ? new Date(entry.timestamp) : new Date();
          return {
            ts: time.toLocaleTimeString(),
            _time: time,
            level: entry.error ? 'ERROR' : 'INFO',
            text: entry.tool_name
              ? `[${entry.tool_name}] ${entry.result_summary || ''}`.trim()
              : (entry.message || JSON.stringify(entry)),
            tool: entry.tool_name || '',
            raw: null,
          };
        } catch {
          return { ts: new Date().toLocaleTimeString(), _time: new Date(), level: 'INFO', text: String(data.line), tool: '', raw: String(data.line) };
        }
      }
      if (data.payload) {
        const p = data.payload;
        const time = p.timestamp ? new Date(p.timestamp) : new Date();
        return {
          ts: time.toLocaleTimeString(),
          _time: time,
          level: p.error ? 'ERROR' : 'INFO',
          text: p.tool_name
            ? `[${p.tool_name}] ${p.result_summary || ''}`.trim()
            : (p.message || JSON.stringify(p)),
          tool: p.tool_name || '',
          raw: null,
        };
      }
      if (typeof data === 'string') {
        return { ts: new Date().toLocaleTimeString(), _time: new Date(), level: 'INFO', text: data, tool: '', raw: data };
      }
      return {
        ts: new Date().toLocaleTimeString(),
        _time: new Date(),
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
      a.download = `heimdall-logs-${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.txt`;
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
      activeLogPreset.value = 'all';
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

    // Preset management
    function applyLogPreset(preset) {
      activeLogPreset.value = preset.id;
      const f = preset.filters;
      levelFilter.value = f.level || '';
      timeRange.value = f.timeRange || '';
      textFilter.value = f.text || '';
      // For multi-level presets like "warnings+", handle differently
      if (f.levels) levelFilter.value = f.levels[0] || '';
      if (f.hasToolName) textFilter.value = '';
    }

    function applyCustomLogPreset(cp) {
      activeLogPreset.value = cp.id;
      levelFilter.value = cp.filters.level || '';
      timeRange.value = cp.filters.timeRange || '';
      textFilter.value = cp.filters.text || '';
    }

    function saveLogCustomPreset() {
      if (!newLogPresetName.value.trim()) return;
      const preset = {
        id: 'custom-' + Date.now(),
        name: newLogPresetName.value.trim(),
        filters: {
          level: levelFilter.value,
          timeRange: timeRange.value,
          text: textFilter.value,
        },
      };
      customLogPresets.value = [...customLogPresets.value, preset];
      saveCustomLogPresetsToStorage();
      showSaveLogPreset.value = false;
      newLogPresetName.value = '';
    }

    function removeLogCustomPreset(id) {
      customLogPresets.value = customLogPresets.value.filter(p => p.id !== id);
      saveCustomLogPresetsToStorage();
      if (activeLogPreset.value === id) activeLogPreset.value = 'all';
    }

    // Track WS connection status
    let statusCheckInterval = null;

    onMounted(() => {
      loadCustomLogPresets();
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
      logPresets, timeRanges, timeRange,
      activeLogPreset, customLogPresets,
      showSaveLogPreset, newLogPresetName,
      hasActiveLogFilters, timeRangeLabel,
      // Timeline
      timelineBuckets, timelineMax, timelineSpanLabel, timelineLabelSkip,
      // Methods
      togglePause, clearLogs, exportLogs, logLineClass, levelClass,
      levelChipClass, toggleLevel, copyLine, jumpToBottom, onScroll,
      applyLogPreset, applyCustomLogPreset,
      saveLogCustomPreset, removeLogCustomPreset,
      segmentHeight, jumpToTimelineBucket,
    };
  },
};
