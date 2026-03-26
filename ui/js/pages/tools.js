/**
 * Loki Management UI — Tools Page
 * Tool list grouped by pack, toggle packs, search/filter, usage stats
 */
import { api } from '../api.js';

const { ref, computed, onMounted } = Vue;

export default {
  template: `
    <div class="p-6">
      <div class="flex items-center justify-between mb-4">
        <h1 class="text-xl font-semibold">Tools</h1>
        <button @click="refresh" class="btn btn-ghost text-xs" :disabled="loading">
          {{ loading ? 'Loading...' : 'Refresh' }}
        </button>
      </div>

      <div v-if="loading && tools.length === 0" class="space-y-3">
        <div class="grid grid-cols-2 md:grid-cols-4 gap-3">
          <div v-for="n in 4" :key="n" class="loki-card text-center">
            <div class="skeleton skeleton-stat"></div>
            <div class="skeleton skeleton-text" style="width:60%;margin:0.25rem auto 0;"></div>
          </div>
        </div>
        <div v-for="n in 5" :key="n + 4" class="skeleton skeleton-row"></div>
      </div>
      <div v-else-if="error" class="loki-card border-red-900 error-state">
        <p class="text-red-400">{{ error }}</p>
        <button @click="refresh" class="btn btn-ghost text-xs">Retry</button>
      </div>
      <div v-else>
        <!-- Summary -->
        <div class="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
          <div class="loki-card text-center">
            <div class="text-2xl font-bold">{{ tools.length }}</div>
            <div class="text-gray-400 text-xs">Total Tools</div>
          </div>
          <div class="loki-card text-center">
            <div class="text-2xl font-bold">{{ coreCount }}</div>
            <div class="text-gray-400 text-xs">Core Tools</div>
          </div>
          <div class="loki-card text-center">
            <div class="text-2xl font-bold">{{ packCount }}</div>
            <div class="text-gray-400 text-xs">Pack Tools</div>
          </div>
          <div class="loki-card text-center">
            <div class="text-2xl font-bold">{{ totalUsage.toLocaleString() }}</div>
            <div class="text-gray-400 text-xs">Total Executions</div>
          </div>
        </div>

        <!-- Tool Packs -->
        <div class="loki-card mb-4">
          <div class="flex items-center justify-between mb-3">
            <div class="text-gray-400 text-sm font-medium">Tool Packs</div>
            <span v-if="packsAllLoaded" class="badge badge-success">All packs loaded</span>
          </div>
          <div class="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-2">
            <div v-for="(info, name) in packs" :key="name"
                 class="tool-pack-card p-3 rounded border"
                 :class="info.enabled ? 'border-green-900/50 bg-green-950/20' : 'border-gray-800 bg-gray-900/30 opacity-60'">
              <div class="flex items-center justify-between mb-1">
                <span class="text-sm font-medium">{{ name }}</span>
                <label class="relative inline-flex items-center cursor-pointer">
                  <input type="checkbox"
                         :checked="info.enabled"
                         @change="togglePack(name, $event.target.checked)"
                         class="sr-only peer">
                  <div class="w-9 h-5 bg-gray-700 rounded-full peer peer-checked:bg-indigo-600
                              after:content-[''] after:absolute after:top-[2px] after:left-[2px]
                              after:bg-white after:rounded-full after:h-4 after:w-4
                              after:transition-all peer-checked:after:translate-x-full"></div>
                </label>
              </div>
              <div class="text-gray-500 text-xs">{{ info.tool_count }} tools</div>
              <div v-if="info.enabled" class="flex flex-wrap gap-1 mt-2">
                <span v-for="t in (info.tools || []).slice(0, 3)" :key="t"
                      class="text-xs font-mono text-gray-500">{{ t }}</span>
                <span v-if="(info.tools || []).length > 3" class="text-xs text-gray-600">
                  +{{ info.tools.length - 3 }}
                </span>
              </div>
            </div>
          </div>
          <div v-if="packSaving" class="mt-2 text-xs text-gray-500">Saving...</div>
          <div v-if="packError" class="mt-2 text-xs text-red-400">{{ packError }}</div>
        </div>

        <!-- Search -->
        <div class="mb-4">
          <input v-model="search" type="text" class="loki-input" placeholder="Search tools by name or description..." />
        </div>

        <!-- Tool list grouped -->
        <div v-for="group in groupedTools" :key="group.label" class="mb-4">
          <div class="flex items-center gap-2 mb-2">
            <span class="text-sm font-medium">{{ group.label }}</span>
            <span class="badge badge-info">{{ group.tools.length }}</span>
          </div>
          <table class="loki-table">
            <thead>
              <tr>
                <th style="width:30%">Name</th>
                <th>Description</th>
                <th style="width:80px" class="text-right">Uses</th>
                <th style="width:80px">Pack</th>
              </tr>
            </thead>
            <tbody>
              <template v-for="t in group.tools" :key="t.name">
                <tr class="cursor-pointer" @click="toggleExpand(t.name)">
                  <td class="font-mono text-sm whitespace-nowrap">
                    <span class="tool-expand-icon text-gray-600 mr-1">{{ expanded[t.name] ? '\u25BC' : '\u25B6' }}</span>
                    {{ t.name }}
                  </td>
                  <td class="text-gray-400 text-sm">{{ truncate(t.description, 100) }}</td>
                  <td class="text-right">
                    <span v-if="stats[t.name]" class="text-gray-300 text-sm font-mono">{{ stats[t.name].toLocaleString() }}</span>
                    <span v-else class="text-gray-600 text-sm">—</span>
                  </td>
                  <td>
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

        <div v-if="filteredTools.length === 0 && search" class="loki-card">
          <p class="text-gray-400 text-sm">No tools match "{{ search }}"</p>
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

    const coreCount = computed(() => tools.value.filter(t => t.is_core).length);
    const packCount = computed(() => tools.value.filter(t => !t.is_core).length);
    const totalUsage = computed(() => Object.values(stats.value).reduce((a, b) => a + b, 0));

    const filteredTools = computed(() => {
      if (!search.value) return tools.value;
      const q = search.value.toLowerCase();
      return tools.value.filter(t =>
        t.name.toLowerCase().includes(q) || (t.description || '').toLowerCase().includes(q)
      );
    });

    const groupedTools = computed(() => {
      const ft = filteredTools.value;
      const core = ft.filter(t => t.is_core);
      const byPack = {};
      for (const t of ft) {
        if (t.pack) {
          if (!byPack[t.pack]) byPack[t.pack] = [];
          byPack[t.pack].push(t);
        }
      }
      const groups = [];
      if (core.length > 0) groups.push({ label: 'Core Tools', tools: core });
      for (const [name, list] of Object.entries(byPack).sort((a, b) => a[0].localeCompare(b[0]))) {
        groups.push({ label: `${name} pack`, tools: list });
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
      packSaving, packError, stats, expanded,
      coreCount, packCount, totalUsage, filteredTools, groupedTools,
      truncate, toggleExpand, togglePack, refresh,
    };
  },
};
