/**
 * Loki Management UI — Config Page
 * Form-based config editor with inline editing, toggle switches, and type-aware inputs.
 */
import { api } from '../api.js';

const { ref, computed, onMounted, nextTick } = Vue;

// Keys whose values are redacted by the server and must not be editable
const SENSITIVE_KEYS = new Set([
  'token', 'api_token', 'secret', 'ssh_key_path', 'credentials_path',
  'api_key', 'password',
]);

// Enum-like fields: path -> allowed values
const ENUM_FIELDS = {
  'logging.level': ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
};

const REDACTED = '\u2022\u2022\u2022\u2022\u2022\u2022\u2022\u2022';

function isSensitiveKey(key) {
  return SENSITIVE_KEYS.has(key);
}

function isRedacted(value) {
  return value === REDACTED;
}

/** Deep-clone a JSON-safe value. */
function deepClone(obj) {
  return JSON.parse(JSON.stringify(obj));
}

/** Compare two JSON-safe values for equality. */
function deepEqual(a, b) {
  return JSON.stringify(a) === JSON.stringify(b);
}

/**
 * Compute the minimal diff between original and edited config.
 * Only includes top-level sections that changed, and within object
 * sections only the changed keys.
 */
function computeDiff(original, edited) {
  const diff = {};
  for (const section of Object.keys(edited)) {
    if (!(section in original)) continue;
    const orig = original[section];
    const edit = edited[section];
    if (deepEqual(orig, edit)) continue;
    if (typeof orig === 'object' && orig !== null && !Array.isArray(orig)
        && typeof edit === 'object' && edit !== null && !Array.isArray(edit)) {
      const sectionDiff = {};
      for (const k of Object.keys(edit)) {
        if (!deepEqual(orig[k], edit[k])) sectionDiff[k] = edit[k];
      }
      if (Object.keys(sectionDiff).length > 0) diff[section] = sectionDiff;
    } else {
      diff[section] = edit;
    }
  }
  return diff;
}

export default {
  template: `
    <div class="p-6 page-fade-in">
      <div class="flex items-center justify-between mb-4">
        <h1 class="text-xl font-semibold">Configuration</h1>
        <div class="flex gap-2">
          <template v-if="editing">
            <button @click="cancelEdit" class="btn btn-ghost text-xs">Cancel</button>
            <button @click="saveConfig" class="btn btn-primary text-xs" :disabled="saving || !hasChanges">
              {{ saving ? 'Saving...' : 'Save Changes' }}
            </button>
          </template>
          <template v-else>
            <button @click="startEdit" class="btn btn-ghost text-xs" :disabled="!config">Edit</button>
            <button @click="fetchConfig" class="btn btn-ghost text-xs" :disabled="loading">
              {{ loading ? 'Loading...' : 'Refresh' }}
            </button>
          </template>
        </div>
      </div>

      <!-- Toast -->
      <div v-if="toast" :class="['toast', toast.type === 'success' ? 'toast-success' : 'toast-error']">
        {{ toast.message }}
      </div>

      <!-- Loading skeleton -->
      <div v-if="loading && !config" class="space-y-3">
        <div v-for="n in 4" :key="n" class="loki-card">
          <div class="skeleton skeleton-text" style="width:100px;"></div>
          <div class="skeleton skeleton-row mt-2"></div>
        </div>
      </div>

      <!-- Error state -->
      <div v-else-if="error" class="loki-card border-red-900 error-state">
        <span class="error-icon">\u26A0</span>
        <p class="text-red-400">{{ error }}</p>
        <button @click="fetchConfig" class="btn btn-ghost text-xs">Retry</button>
      </div>

      <!-- Config sections -->
      <div v-else-if="config" class="space-y-3">
        <div v-for="(value, section) in displayConfig" :key="section" class="loki-card">
          <!-- Section header -->
          <div class="flex items-center gap-2 mb-2 cursor-pointer select-none"
               @click="toggleSection(section)">
            <span class="text-xs text-gray-500 font-mono">{{ expanded[section] ? '\\u25BC' : '\\u25B6' }}</span>
            <span class="text-sm font-medium">{{ section }}</span>
            <span class="badge badge-info text-xs" v-if="typeof value === 'object' && value !== null && !Array.isArray(value)">
              {{ Object.keys(value).length }} fields
            </span>
            <span v-if="editing && sectionChanged(section)" class="badge badge-warning text-xs">modified</span>
          </div>

          <div v-if="expanded[section]">
            <!-- Scalar top-level field (e.g. timezone) -->
            <div v-if="typeof value !== 'object' || value === null" class="pl-4">
              <template v-if="editing">
                <input class="loki-input font-mono text-sm" style="max-width:300px"
                       :value="getEdited(section)"
                       @input="setEdited(section, null, $event.target.value)" />
              </template>
              <template v-else>
                <span class="text-sm font-mono text-gray-300">{{ value === '' ? '(empty)' : value }}</span>
              </template>
            </div>

            <!-- Object section -->
            <div v-else-if="!Array.isArray(value)">
              <table class="loki-table">
                <thead>
                  <tr>
                    <th style="width:30%">Key</th>
                    <th>Value</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="(v, k) in value" :key="k"
                      :class="{'field-changed': editing && fieldChanged(section, k)}">
                    <td class="font-mono text-xs text-gray-400">{{ k }}</td>
                    <td>
                      <!-- Sensitive field: always masked, never editable -->
                      <template v-if="isSensitiveKey(k) || isRedacted(v)">
                        <span class="text-gray-500 font-mono text-xs flex items-center gap-2">
                          {{ REDACTED }}
                          <span class="badge badge-warning text-xs">sensitive</span>
                        </span>
                      </template>

                      <!-- Boolean: toggle switch -->
                      <template v-else-if="typeof v === 'boolean'">
                        <div class="flex items-center gap-2">
                          <label class="toggle-switch" v-if="editing">
                            <input type="checkbox" :checked="getEditedField(section, k)"
                                   @change="setEdited(section, k, $event.target.checked)" />
                            <span class="toggle-slider"></span>
                          </label>
                          <span :class="getDisplayBool(section, k) ? 'text-green-400' : 'text-red-400'"
                                class="text-sm font-mono">
                            {{ getDisplayBool(section, k) }}
                          </span>
                        </div>
                      </template>

                      <!-- Enum field: dropdown -->
                      <template v-else-if="editing && getEnumOptions(section, k)">
                        <select class="loki-select"
                                :value="getEditedField(section, k)"
                                @change="setEdited(section, k, $event.target.value)">
                          <option v-for="opt in getEnumOptions(section, k)" :key="opt" :value="opt">{{ opt }}</option>
                        </select>
                      </template>

                      <!-- Number field -->
                      <template v-else-if="typeof v === 'number'">
                        <template v-if="editing">
                          <input type="number" class="loki-input font-mono text-sm" style="max-width:200px"
                                 :value="getEditedField(section, k)"
                                 @input="setEdited(section, k, Number($event.target.value))" />
                        </template>
                        <template v-else>
                          <span class="text-sm font-mono text-gray-300">{{ v }}</span>
                        </template>
                      </template>

                      <!-- Array field: tags -->
                      <template v-else-if="Array.isArray(v)">
                        <div class="flex flex-wrap gap-1 items-center">
                          <template v-if="editing">
                            <span v-for="(item, i) in getEditedField(section, k)" :key="i" class="config-tag">
                              {{ typeof item === 'object' ? JSON.stringify(item) : item }}
                              <button @click="removeArrayItem(section, k, i)">&times;</button>
                            </span>
                            <button class="btn btn-ghost text-xs" @click="addArrayItem(section, k)">+ Add</button>
                          </template>
                          <template v-else>
                            <template v-if="v.length === 0">
                              <span class="text-gray-500 text-sm">(empty list)</span>
                            </template>
                            <span v-else v-for="(item, i) in v" :key="i" class="config-tag">
                              {{ typeof item === 'object' ? JSON.stringify(item) : item }}
                            </span>
                          </template>
                        </div>
                      </template>

                      <!-- Nested object: expandable JSON -->
                      <template v-else-if="typeof v === 'object' && v !== null">
                        <div>
                          <button @click="toggleNested(section + '.' + k)" class="btn btn-ghost text-xs">
                            {{ expandedNested[section + '.' + k] ? 'Collapse' : 'Expand' }}
                            ({{ Object.keys(v).length }} fields)
                          </button>
                          <div v-if="expandedNested[section + '.' + k]" class="mt-2">
                            <template v-if="editing">
                              <textarea class="loki-input font-mono text-xs" rows="6"
                                        :value="formatJson(getEditedField(section, k))"
                                        @blur="setEditedJson(section, k, $event.target.value)"></textarea>
                              <p class="text-xs text-gray-500 mt-1">Edit as JSON</p>
                            </template>
                            <pre v-else
                                 class="p-2 rounded bg-gray-900 text-xs text-gray-300 overflow-x-auto font-mono">{{ formatJson(v) }}</pre>
                          </div>
                        </div>
                      </template>

                      <!-- String field: text input -->
                      <template v-else>
                        <template v-if="editing">
                          <input class="loki-input font-mono text-sm"
                                 :value="getEditedField(section, k)"
                                 @input="setEdited(section, k, $event.target.value)" />
                        </template>
                        <template v-else>
                          <span class="text-sm font-mono text-gray-300">{{ v === '' ? '(empty)' : v }}</span>
                        </template>
                      </template>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>

            <!-- Array top-level section -->
            <div v-else>
              <div v-if="value.length === 0" class="text-gray-500 text-sm pl-4">(empty list)</div>
              <ul v-else class="pl-4 space-y-1">
                <li v-for="(item, i) in value" :key="i" class="text-sm font-mono text-gray-300">
                  {{ typeof item === 'object' ? JSON.stringify(item) : item }}
                </li>
              </ul>
            </div>
          </div>
        </div>

        <div class="mt-4 text-xs text-gray-500">
          Fields marked <span class="badge badge-warning">sensitive</span> are redacted by the server and cannot be edited here.
        </div>
      </div>
    </div>`,

  setup() {
    const config = ref(null);        // Original config from server
    const editValues = ref(null);    // Editable copy during edit mode
    const loading = ref(true);
    const error = ref(null);
    const expanded = ref({});
    const expandedNested = ref({});
    const editing = ref(false);
    const saving = ref(false);
    const toast = ref(null);

    // Which config to display: edited copy when editing, original otherwise
    const displayConfig = computed(() => editing.value && editValues.value ? editValues.value : config.value);

    const hasChanges = computed(() => {
      if (!config.value || !editValues.value) return false;
      return !deepEqual(config.value, editValues.value);
    });

    function sectionChanged(section) {
      if (!config.value || !editValues.value) return false;
      return !deepEqual(config.value[section], editValues.value[section]);
    }

    function fieldChanged(section, key) {
      if (!config.value || !editValues.value) return false;
      const orig = config.value[section];
      const edit = editValues.value[section];
      if (!orig || !edit) return false;
      return !deepEqual(orig[key], edit[key]);
    }

    function getEdited(section) {
      return editValues.value ? editValues.value[section] : config.value[section];
    }

    function getEditedField(section, key) {
      const src = editValues.value || config.value;
      return src[section] ? src[section][key] : undefined;
    }

    function getDisplayBool(section, key) {
      const src = editing.value && editValues.value ? editValues.value : config.value;
      return src[section] ? src[section][key] : false;
    }

    function setEdited(section, key, value) {
      if (!editValues.value) return;
      if (key === null) {
        editValues.value[section] = value;
      } else {
        if (!editValues.value[section]) editValues.value[section] = {};
        editValues.value[section][key] = value;
      }
      // Force reactivity
      editValues.value = { ...editValues.value };
    }

    function setEditedJson(section, key, jsonStr) {
      try {
        const parsed = JSON.parse(jsonStr);
        setEdited(section, key, parsed);
      } catch {
        // Invalid JSON — keep old value
      }
    }

    function getEnumOptions(section, key) {
      return ENUM_FIELDS[section + '.' + key] || null;
    }

    function removeArrayItem(section, key, index) {
      if (!editValues.value || !editValues.value[section]) return;
      const arr = [...editValues.value[section][key]];
      arr.splice(index, 1);
      setEdited(section, key, arr);
    }

    function addArrayItem(section, key) {
      if (!editValues.value || !editValues.value[section]) return;
      const arr = [...(editValues.value[section][key] || [])];
      const val = prompt('Enter new value:');
      if (val === null) return;
      arr.push(val);
      setEdited(section, key, arr);
    }

    function toggleSection(section) {
      expanded.value = { ...expanded.value, [section]: !expanded.value[section] };
    }

    function toggleNested(key) {
      expandedNested.value = { ...expandedNested.value, [key]: !expandedNested.value[key] };
    }

    function formatJson(obj) {
      try { return JSON.stringify(obj, null, 2); }
      catch { return String(obj); }
    }

    function showToast(type, message) {
      toast.value = { type, message };
      setTimeout(() => { toast.value = null; }, 3000);
    }

    function startEdit() {
      editValues.value = deepClone(config.value);
      editing.value = true;
    }

    function cancelEdit() {
      editing.value = false;
      editValues.value = null;
    }

    async function saveConfig() {
      if (!hasChanges.value) return;
      saving.value = true;
      try {
        const diff = computeDiff(config.value, editValues.value);
        if (Object.keys(diff).length === 0) {
          showToast('success', 'No changes to save.');
          saving.value = false;
          return;
        }
        const result = await api.put('/api/config', diff);
        config.value = result;
        editing.value = false;
        editValues.value = null;
        showToast('success', 'Config saved successfully.');
      } catch (e) {
        showToast('error', e.message || 'Failed to save config');
      }
      saving.value = false;
    }

    async function fetchConfig() {
      loading.value = true;
      error.value = null;
      try {
        config.value = await api.get('/api/config');
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
      config, displayConfig, editValues, loading, error,
      expanded, expandedNested, editing, saving, toast,
      hasChanges, REDACTED,
      isSensitiveKey, isRedacted, sectionChanged, fieldChanged,
      getEdited, getEditedField, getDisplayBool, setEdited, setEditedJson,
      getEnumOptions, removeArrayItem, addArrayItem,
      toggleSection, toggleNested, formatJson, showToast,
      fetchConfig, startEdit, cancelEdit, saveConfig,
    };
  },
};
