/**
 * Loki Management UI — Config Page
 * Display config as structured form with sensitive field redaction + inline editing
 */
import { api } from '../api.js';

const { ref, computed, onMounted } = Vue;

export default {
  template: `
    <div class="p-6">
      <div class="flex items-center justify-between mb-4">
        <h1 class="text-xl font-semibold">Configuration</h1>
        <div class="flex gap-2">
          <button v-if="editing" @click="cancelEdit" class="btn btn-ghost text-xs">Cancel</button>
          <button v-if="editing" @click="saveConfig" class="btn btn-ghost text-xs text-green-400"
                  :disabled="saving">
            {{ saving ? 'Saving...' : 'Save Changes' }}
          </button>
          <button v-if="!editing" @click="startEdit" class="btn btn-ghost text-xs" :disabled="!config">
            Edit
          </button>
          <button @click="fetchConfig" class="btn btn-ghost text-xs" :disabled="loading">
            {{ loading ? 'Loading...' : 'Refresh' }}
          </button>
        </div>
      </div>

      <div v-if="saveError" class="loki-card border-red-900 mb-3">
        <p class="text-red-400 text-sm">{{ saveError }}</p>
      </div>
      <div v-if="saveSuccess" class="loki-card border-green-900 mb-3">
        <p class="text-green-400 text-sm">Config saved successfully.</p>
      </div>

      <div v-if="loading && !config" class="space-y-3">
        <div v-for="n in 4" :key="n" class="loki-card">
          <div class="skeleton skeleton-text" style="width:100px;"></div>
          <div class="skeleton skeleton-row mt-2"></div>
        </div>
      </div>
      <div v-else-if="error" class="loki-card border-red-900 error-state">
        <p class="text-red-400">{{ error }}</p>
        <button @click="fetchConfig" class="btn btn-ghost text-xs">Retry</button>
      </div>
      <div v-else-if="config">
        <!-- Edit mode: JSON editor -->
        <div v-if="editing" class="loki-card">
          <p class="text-xs text-gray-400 mb-2">Edit config as JSON (partial updates — only include fields you want to change). Sensitive fields (token, api_key, etc.) cannot be changed here.</p>
          <textarea v-model="editJson" rows="20"
                    class="w-full bg-gray-900 text-gray-300 font-mono text-xs p-3 rounded border border-gray-700 focus:border-blue-500 focus:outline-none"></textarea>
        </div>

        <!-- Read mode: structured view -->
        <div v-else class="space-y-3">
          <div v-for="(value, section) in config" :key="section" class="loki-card">
            <div class="flex items-center gap-2 mb-2 cursor-pointer select-none"
                 @click="toggleSection(section)">
              <span class="text-xs text-gray-500 font-mono">{{ expanded[section] ? '\u25BC' : '\u25B6' }}</span>
              <span class="text-sm font-medium">{{ section }}</span>
              <span class="badge badge-info text-xs">{{ typeof value === 'object' && value !== null ? Object.keys(value).length + ' fields' : typeof value }}</span>
            </div>

            <div v-if="expanded[section]">
              <!-- Object section -->
              <div v-if="typeof value === 'object' && value !== null && !Array.isArray(value)">
                <table class="loki-table">
                  <thead>
                    <tr>
                      <th style="width:30%">Key</th>
                      <th>Value</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr v-for="(v, k) in value" :key="k">
                      <td class="font-mono text-xs text-gray-400">{{ k }}</td>
                      <td>
                        <template v-if="isRedacted(v)">
                          <span class="text-gray-500 font-mono text-xs flex items-center gap-2">
                            {{ revealed[section + '.' + k] ? v : '\u2022\u2022\u2022\u2022\u2022\u2022\u2022\u2022' }}
                            <span class="badge badge-warning text-xs">sensitive</span>
                          </span>
                        </template>
                        <template v-else-if="typeof v === 'object' && v !== null">
                          <button @click="toggleNested(section + '.' + k)"
                                  class="btn btn-ghost text-xs">
                            {{ expandedNested[section + '.' + k] ? 'Collapse' : 'Expand' }}
                            ({{ Array.isArray(v) ? v.length + ' items' : Object.keys(v).length + ' fields' }})
                          </button>
                          <pre v-if="expandedNested[section + '.' + k]"
                               class="mt-2 p-2 rounded bg-gray-900 text-xs text-gray-300 overflow-x-auto font-mono">{{ formatJson(v) }}</pre>
                        </template>
                        <template v-else-if="typeof v === 'boolean'">
                          <span :class="v ? 'text-green-400' : 'text-red-400'" class="text-sm font-mono">{{ v }}</span>
                        </template>
                        <template v-else>
                          <span class="text-sm font-mono text-gray-300">{{ v === '' ? '(empty)' : v }}</span>
                        </template>
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>

              <!-- Array section -->
              <div v-else-if="Array.isArray(value)">
                <div v-if="value.length === 0" class="text-gray-500 text-sm pl-4">(empty list)</div>
                <ul v-else class="pl-4 space-y-1">
                  <li v-for="(item, i) in value" :key="i" class="text-sm font-mono text-gray-300">
                    {{ typeof item === 'object' ? JSON.stringify(item) : item }}
                  </li>
                </ul>
              </div>

              <!-- Scalar section -->
              <div v-else class="pl-4">
                <span class="text-sm font-mono text-gray-300">{{ value === '' ? '(empty)' : value }}</span>
              </div>
            </div>
          </div>
        </div>

        <div class="mt-4 text-xs text-gray-500">
          Fields marked <span class="badge badge-warning">sensitive</span> are redacted by the server and cannot be viewed or edited here.
        </div>
      </div>
    </div>`,

  setup() {
    const config = ref(null);
    const loading = ref(true);
    const error = ref(null);
    const expanded = ref({});
    const expandedNested = ref({});
    const revealed = ref({});
    const editing = ref(false);
    const editJson = ref('');
    const saving = ref(false);
    const saveError = ref(null);
    const saveSuccess = ref(false);

    function isRedacted(value) {
      return value === '\u2022\u2022\u2022\u2022\u2022\u2022\u2022\u2022';
    }

    function toggleSection(section) {
      expanded.value[section] = !expanded.value[section];
    }

    function toggleNested(key) {
      expandedNested.value[key] = !expandedNested.value[key];
    }

    function formatJson(obj) {
      try {
        return JSON.stringify(obj, null, 2);
      } catch {
        return String(obj);
      }
    }

    function startEdit() {
      saveError.value = null;
      saveSuccess.value = false;
      editJson.value = JSON.stringify(config.value, null, 2);
      editing.value = true;
    }

    function cancelEdit() {
      editing.value = false;
      editJson.value = '';
      saveError.value = null;
    }

    async function saveConfig() {
      saving.value = true;
      saveError.value = null;
      saveSuccess.value = false;
      try {
        const updates = JSON.parse(editJson.value);
        const result = await api.put('/api/config', updates);
        config.value = result;
        editing.value = false;
        editJson.value = '';
        saveSuccess.value = true;
        setTimeout(() => { saveSuccess.value = false; }, 3000);
      } catch (e) {
        saveError.value = e.message || 'Failed to save config';
      }
      saving.value = false;
    }

    async function fetchConfig() {
      loading.value = true;
      error.value = null;
      try {
        config.value = await api.get('/api/config');
        // Auto-expand all top-level sections
        for (const key of Object.keys(config.value)) {
          if (expanded.value[key] === undefined) {
            expanded.value[key] = true;
          }
        }
      } catch (e) {
        error.value = e.message;
      }
      loading.value = false;
    }

    onMounted(() => { fetchConfig(); });

    return {
      config, loading, error,
      expanded, expandedNested, revealed,
      editing, editJson, saving, saveError, saveSuccess,
      isRedacted, toggleSection, toggleNested, formatJson, fetchConfig,
      startEdit, cancelEdit, saveConfig,
    };
  },
};
