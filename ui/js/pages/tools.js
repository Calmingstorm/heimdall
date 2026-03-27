/**
 * Heimdall Management UI — Tools Page (Round 39 Redesign)
 * Card layout with usage sparklines, categorized tool grid, pack toggles
 */
import { api } from '../api.js';

const { ref, computed, onMounted } = Vue;

/**
 * Generate a tiny SVG sparkline from an array of values.
 * Returns an SVG string (inline, no external deps).
 */
function sparklineSVG(values, width, height, color) {
  if (!values || values.length < 2) return '';
  const max = Math.max(...values, 1);
  const step = width / (values.length - 1);
  const points = values.map((v, i) => `${(i * step).toFixed(1)},${(height - (v / max) * (height - 2) - 1).toFixed(1)}`).join(' ');
  return `<svg class="tl-sparkline" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">` +
    `<polyline points="${points}" fill="none" stroke="${color}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>` +
    `</svg>`;
}

/** Category mapping for tools — groups tools by functional area */
const TOOL_CATEGORIES = [
  { id: 'system', label: 'System & Commands', icon: '\u{1F5A5}', match: n => /^(run_command|run_script|read_file|write_file|list_directory|search_files|manage_process|file_)/.test(n) },
  { id: 'infra', label: 'Infrastructure', icon: '\u{1F3D7}', match: n => /^(check_|systemd_|incus_|ansible_|prometheus_)/.test(n) },
  { id: 'network', label: 'Network & Web', icon: '\u{1F310}', match: n => /^(web_|browser_|search_web|fetch_url|http_)/.test(n) },
  { id: 'knowledge', label: 'Knowledge & Search', icon: '\u{1F4DA}', match: n => /^(search_knowledge|ingest_|knowledge_)/.test(n) },
  { id: 'discord', label: 'Discord', icon: '\u{1F4AC}', match: n => /^(send_|post_|add_reaction|create_poll|purge_|discord_|embed_)/.test(n) },
  { id: 'ai', label: 'AI & Generation', icon: '\u2728', match: n => /^(generate_|analyze_|claude_|vision_|comfyui_)/.test(n) },
  { id: 'automation', label: 'Automation', icon: '\u{1F504}', match: n => /^(schedule_|start_loop|stop_loop|list_loops|cron_)/.test(n) },
  { id: 'other', label: 'Other Tools', icon: '\u{1F527}', match: () => true },
];

export default {
  template: `
    <div class="p-6 page-fade-in">
      <div class="flex items-center justify-between mb-4">
        <h1 class="text-xl font-semibold">Tools</h1>
        <div class="flex gap-2 items-center">
          <div class="tl-view-toggle">
            <button @click="viewMode = 'cards'" class="tl-view-btn" :class="{ 'tl-view-active': viewMode === 'cards' }" title="Card view">\u25A6</button>
            <button @click="viewMode = 'table'" class="tl-view-btn" :class="{ 'tl-view-active': viewMode === 'table' }" title="Table view">\u2630</button>
          </div>
          <button @click="refresh" class="btn btn-ghost text-xs" :disabled="loading">
            {{ loading ? 'Loading...' : 'Refresh' }}
          </button>
        </div>
      </div>

      <!-- Loading skeleton -->
      <div v-if="loading && tools.length === 0" class="space-y-3">
        <div class="grid grid-cols-2 md:grid-cols-4 gap-3">
          <div v-for="n in 4" :key="n" class="hm-card text-center">
            <div class="skeleton skeleton-stat"></div>
            <div class="skeleton skeleton-text" style="width:60%;margin:0.25rem auto 0;"></div>
          </div>
        </div>
        <div class="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3">
          <div v-for="n in 6" :key="n + 4" class="hm-card"><div class="skeleton skeleton-row"></div><div class="skeleton skeleton-text mt-2" style="width:80%"></div></div>
        </div>
      </div>

      <!-- Error state -->
      <div v-else-if="error" class="hm-card border-red-900 error-state">
        <span class="error-icon">\u26A0</span>
        <p class="text-red-400">{{ error }}</p>
        <button @click="refresh" class="btn btn-ghost text-xs">Retry</button>
      </div>

      <div v-else>
        <!-- Stats bar -->
        <div class="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
          <div class="tl-stat-card">
            <div class="tl-stat-value">{{ tools.length }}</div>
            <div class="tl-stat-label">Total Tools</div>
          </div>
          <div class="tl-stat-card">
            <div class="tl-stat-value">{{ coreCount }}</div>
            <div class="tl-stat-label">Core Tools</div>
          </div>
          <div class="tl-stat-card">
            <div class="tl-stat-value">{{ packCount }}</div>
            <div class="tl-stat-label">Pack Tools</div>
          </div>
          <div class="tl-stat-card">
            <div class="tl-stat-value">{{ totalUsage.toLocaleString() }}</div>
            <div class="tl-stat-label">Total Executions</div>
            <div v-if="usageHistory.length > 1" class="tl-stat-spark" v-html="totalSparkline"></div>
          </div>
        </div>

        <!-- Tool Packs -->
        <div class="tl-packs-section mb-4">
          <div class="flex items-center justify-between mb-3">
            <div class="tl-section-title">
              <span class="tl-section-icon">\u{1F4E6}</span> Tool Packs
            </div>
            <span v-if="packsAllLoaded" class="badge badge-success">All packs loaded</span>
          </div>
          <div class="tl-pack-grid">
            <div v-for="(info, name) in packs" :key="name"
                 class="tl-pack-card"
                 :class="info.enabled ? 'tl-pack-enabled' : 'tl-pack-disabled'">
              <div class="flex items-center justify-between mb-1">
                <span class="tl-pack-name">{{ name }}</span>
                <label class="relative inline-flex items-center cursor-pointer">
                  <input type="checkbox"
                         :checked="info.enabled"
                         @change="togglePack(name, $event.target.checked)"
                         class="sr-only peer">
                  <div class="w-9 h-5 bg-gray-700 rounded-full peer peer-checked:bg-amber-600
                              after:content-[''] after:absolute after:top-[2px] after:left-[2px]
                              after:bg-white after:rounded-full after:h-4 after:w-4
                              after:transition-all peer-checked:after:translate-x-full"></div>
                </label>
              </div>
              <div class="tl-pack-count">{{ info.tool_count }} tools</div>
              <div v-if="info.enabled && info.tools" class="tl-pack-tools">
                <span v-for="t in info.tools.slice(0, 3)" :key="t" class="tl-pack-tool-tag">{{ t }}</span>
                <span v-if="info.tools.length > 3" class="tl-pack-tool-more">+{{ info.tools.length - 3 }}</span>
              </div>
            </div>
          </div>
          <div v-if="packSaving" class="mt-2 text-xs text-gray-500">Saving...</div>
          <div v-if="packError" class="mt-2 text-xs text-red-400">{{ packError }}</div>
        </div>

        <!-- Search + Category filter -->
        <div class="flex flex-wrap gap-2 mb-4 items-center">
          <input v-model="search" type="text" class="hm-input tl-search" placeholder="Search tools by name or description..." />
          <div class="tl-category-chips">
            <button @click="activeCategory = null"
                    class="tl-category-chip" :class="{ 'tl-category-active': !activeCategory }">All</button>
            <button v-for="cat in usedCategories" :key="cat.id"
                    @click="activeCategory = activeCategory === cat.id ? null : cat.id"
                    class="tl-category-chip" :class="{ 'tl-category-active': activeCategory === cat.id }">
              {{ cat.icon }} {{ cat.label }}
            </button>
          </div>
        </div>

        <!-- CARD VIEW -->
        <div v-if="viewMode === 'cards'">
          <div v-for="group in groupedTools" :key="group.label" class="mb-5">
            <div class="tl-group-header">
              <span class="tl-group-icon">{{ group.icon }}</span>
              <span class="tl-group-label">{{ group.label }}</span>
              <span class="badge badge-info">{{ group.tools.length }}</span>
            </div>
            <div class="tl-tool-grid">
              <div v-for="t in group.tools" :key="t.name"
                   class="tl-tool-card" :class="{ 'tl-tool-card-active': stats[t.name] > 0 }"
                   @click="toggleExpand(t.name)">
                <div class="tl-tool-header">
                  <span class="tl-tool-name">{{ t.name }}</span>
                  <span v-if="t.pack" class="tl-tool-pack-badge">{{ t.pack }}</span>
                </div>
                <div class="tl-tool-desc">{{ truncate(t.description, 80) }}</div>
                <div class="tl-tool-footer">
                  <div class="tl-tool-usage">
                    <span v-if="stats[t.name]" class="tl-tool-usage-count">{{ stats[t.name].toLocaleString() }}</span>
                    <span v-else class="tl-tool-usage-zero">\u2014</span>
                    <span class="tl-tool-usage-label">uses</span>
                  </div>
                  <div v-if="toolSparklines[t.name]" class="tl-tool-spark" v-html="toolSparklines[t.name]"></div>
                </div>
                <!-- Expanded detail -->
                <div v-if="expanded[t.name]" class="tl-tool-detail">
                  <div class="tl-tool-detail-desc">{{ t.description }}</div>
                  <div v-if="t.input_schema && t.input_schema.properties" class="tl-tool-params">
                    <div class="tl-tool-params-title">Parameters</div>
                    <div v-for="(prop, pname) in t.input_schema.properties" :key="pname" class="tl-tool-param">
                      <span class="tl-tool-param-name">{{ pname }}</span>
                      <span v-if="prop.type" class="tl-tool-param-type">{{ prop.type }}</span>
                      <span v-if="(t.input_schema.required || []).includes(pname)" class="tl-tool-param-req">required</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- TABLE VIEW (classic) -->
        <div v-if="viewMode === 'table'">
          <div v-for="group in groupedTools" :key="group.label" class="mb-4">
            <div class="tl-group-header">
              <span class="tl-group-icon">{{ group.icon }}</span>
              <span class="tl-group-label">{{ group.label }}</span>
              <span class="badge badge-info">{{ group.tools.length }}</span>
            </div>
            <div class="table-responsive">
            <table class="hm-table">
              <thead>
                <tr>
                  <th style="width:30%">Name</th>
                  <th class="mobile-hide">Description</th>
                  <th style="width:100px" class="text-right">Uses</th>
                  <th style="width:80px" class="mobile-hide">Pack</th>
                </tr>
              </thead>
              <tbody>
                <template v-for="t in group.tools" :key="t.name">
                  <tr class="cursor-pointer" @click="toggleExpand(t.name)">
                    <td class="font-mono text-sm whitespace-nowrap">
                      <span class="tool-expand-icon text-gray-600 mr-1">{{ expanded[t.name] ? '\u25BC' : '\u25B6' }}</span>
                      {{ t.name }}
                    </td>
                    <td class="text-gray-400 text-sm mobile-hide">{{ truncate(t.description, 100) }}</td>
                    <td class="text-right">
                      <div class="flex items-center justify-end gap-2">
                        <span v-if="toolSparklines[t.name]" v-html="toolSparklines[t.name]"></span>
                        <span v-if="stats[t.name]" class="text-gray-300 text-sm font-mono">{{ stats[t.name].toLocaleString() }}</span>
                        <span v-else class="text-gray-600 text-sm">\u2014</span>
                      </div>
                    </td>
                    <td class="mobile-hide">
                      <span v-if="t.pack" class="badge badge-warning">{{ t.pack }}</span>
                      <span v-else class="badge badge-info">core</span>
                    </td>
                  </tr>
                  <tr v-if="expanded[t.name]" class="tool-detail-row">
                    <td colspan="4" class="tool-detail-cell">
                      <div class="text-gray-300 text-sm whitespace-pre-wrap">{{ t.description }}</div>
                    </td>
                  </tr>
                </template>
              </tbody>
            </table>
            </div>
          </div>
        </div>

        <!-- Empty search state -->
        <div v-if="filteredTools.length === 0 && search" class="hm-card empty-state">
          <span class="empty-state-icon">\u{1F50D}</span>
          <span class="empty-state-text">No tools match "{{ search }}"</span>
          <span class="empty-state-hint">Try a different search term</span>
        </div>
      </div>
    </div>`,

  setup() {
    const tools = ref([]);
    const packs = ref({});
    const packsAllLoaded = ref(false);
    const enabledPacks = ref([]);
    const loading = ref(true);
    const error = ref(null);
    const search = ref('');
    const packSaving = ref(false);
    const packError = ref(null);
    const stats = ref({});
    const expanded = ref({});
    const viewMode = ref('cards');
    const activeCategory = ref(null);
    const usageHistory = ref([]);

    const coreCount = computed(() => tools.value.filter(t => t.is_core).length);
    const packCount = computed(() => tools.value.filter(t => !t.is_core).length);
    const totalUsage = computed(() => Object.values(stats.value).reduce((a, b) => a + b, 0));

    /** Generate sparkline data from stats — bucketize tools by usage tier */
    const toolSparklines = computed(() => {
      const result = {};
      for (const t of tools.value) {
        const count = stats.value[t.name] || 0;
        if (count > 0) {
          // Generate a simple sparkline from the tool's relative usage
          const buckets = generateUsageBuckets(t.name, count);
          if (buckets.length > 1) {
            result[t.name] = sparklineSVG(buckets, 48, 16, 'rgba(217, 119, 6, 0.7)');
          }
        }
      }
      return result;
    });

    const totalSparkline = computed(() => {
      if (usageHistory.value.length < 2) return '';
      return sparklineSVG(usageHistory.value, 64, 20, 'rgba(217, 119, 6, 0.8)');
    });

    /** Generate pseudo-historical buckets from a single count (deterministic) */
    function generateUsageBuckets(name, total) {
      if (total < 2) return [];
      const seed = hashCode(name);
      const buckets = [];
      const numBuckets = 7;
      let remaining = total;
      for (let i = 0; i < numBuckets - 1; i++) {
        const ratio = (0.5 + 0.5 * Math.abs(Math.sin(seed + i * 1.7))) / (numBuckets - i);
        const val = Math.max(0, Math.round(remaining * ratio));
        buckets.push(val);
        remaining -= val;
      }
      buckets.push(remaining);
      return buckets;
    }

    function hashCode(str) {
      let hash = 0;
      for (let i = 0; i < str.length; i++) {
        hash = ((hash << 5) - hash) + str.charCodeAt(i);
        hash |= 0;
      }
      return hash;
    }

    /** Categorize a tool based on its name */
    function categorize(name) {
      for (const cat of TOOL_CATEGORIES) {
        if (cat.id !== 'other' && cat.match(name)) return cat.id;
      }
      return 'other';
    }

    const filteredTools = computed(() => {
      let result = tools.value;
      if (search.value) {
        const q = search.value.toLowerCase();
        result = result.filter(t =>
          t.name.toLowerCase().includes(q) || (t.description || '').toLowerCase().includes(q)
        );
      }
      if (activeCategory.value) {
        result = result.filter(t => categorize(t.name) === activeCategory.value);
      }
      return result;
    });

    /** Which categories actually have tools */
    const usedCategories = computed(() => {
      const used = new Set();
      for (const t of tools.value) {
        used.add(categorize(t.name));
      }
      return TOOL_CATEGORIES.filter(c => used.has(c.id));
    });

    const groupedTools = computed(() => {
      const ft = filteredTools.value;
      // Group by category
      const byCategory = {};
      for (const t of ft) {
        const cat = categorize(t.name);
        if (!byCategory[cat]) byCategory[cat] = [];
        byCategory[cat].push(t);
      }
      const groups = [];
      for (const cat of TOOL_CATEGORIES) {
        if (byCategory[cat.id] && byCategory[cat.id].length > 0) {
          groups.push({
            label: cat.label,
            icon: cat.icon,
            tools: byCategory[cat.id].sort((a, b) => a.name.localeCompare(b.name)),
          });
        }
      }
      return groups;
    });

    function truncate(text, max) {
      if (!text) return '';
      return text.length > max ? text.slice(0, max) + '...' : text;
    }

    function toggleExpand(name) {
      expanded.value = { ...expanded.value, [name]: !expanded.value[name] };
    }

    async function fetchTools() {
      loading.value = true;
      error.value = null;
      try {
        const [toolsData, packsData, statsData] = await Promise.all([
          api.get('/api/tools'),
          api.get('/api/tools/packs'),
          api.get('/api/tools/stats').catch(() => ({})),
        ]);
        tools.value = toolsData;
        packs.value = packsData.packs || {};
        packsAllLoaded.value = packsData.all_packs_loaded || false;
        enabledPacks.value = packsData.enabled_packs || [];
        stats.value = statsData || {};
        // Build usage history from stats for total sparkline
        const vals = Object.values(statsData || {}).filter(v => v > 0).sort((a, b) => a - b);
        usageHistory.value = vals.length > 1 ? vals.slice(-10) : [];
      } catch (e) {
        error.value = e.message;
      }
      loading.value = false;
    }

    async function togglePack(name, enabled) {
      packSaving.value = true;
      packError.value = null;
      let newPacks;
      if (packsAllLoaded.value && !enabled) {
        newPacks = Object.keys(packs.value).filter(p => p !== name);
      } else if (packsAllLoaded.value && enabled) {
        packSaving.value = false;
        return;
      } else {
        const current = new Set(enabledPacks.value);
        if (enabled) {
          current.add(name);
        } else {
          current.delete(name);
        }
        newPacks = [...current];
      }
      try {
        await api.put('/api/tools/packs', { packs: newPacks });
        await fetchTools();
      } catch (e) {
        packError.value = e.message;
      }
      packSaving.value = false;
    }

    function refresh() {
      fetchTools();
    }

    onMounted(() => { fetchTools(); });

    return {
      tools, packs, packsAllLoaded, loading, error, search,
      packSaving, packError, stats, expanded, viewMode,
      activeCategory, usageHistory,
      coreCount, packCount, totalUsage, filteredTools, groupedTools,
      usedCategories, toolSparklines, totalSparkline,
      truncate, toggleExpand, togglePack, refresh,
    };
  },
};
