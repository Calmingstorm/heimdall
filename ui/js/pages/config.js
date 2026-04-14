/**
 * Heimdall Management UI — Config Page (Round 37 redesign)
 * Grouped sections, inline validation, undo/redo, diff view.
 */
import { api } from '../api.js';

const { ref, computed, onMounted, onUnmounted, watch } = Vue;

// Keys whose values are redacted by the server and must not be editable
const SENSITIVE_KEYS = new Set([
  'token', 'api_token', 'secret', 'ssh_key_path', 'credentials_path',
  'api_key', 'password',
]);

// Enum-like fields: path -> allowed values
const ENUM_FIELDS = {
  'logging.level': ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
};

// Validation rules: section.key -> { type, min?, max?, pattern?, message? }
const VALIDATION_RULES = {
  'discord.allowed_users': { type: 'array', itemType: 'string', message: 'Must be a list of user IDs' },
  'discord.channels': { type: 'array', itemType: 'string', message: 'Must be a list of channel IDs' },
  'openai_codex.max_tokens': { type: 'number', min: 1, max: 128000, message: 'Must be 1\u201c128000' },
  'sessions.max_history': { type: 'number', min: 1, max: 10000, message: 'Must be 1\u201310000' },
  'sessions.max_age_hours': { type: 'number', min: 1, message: 'Must be at least 1' },
  'learning.max_entries': { type: 'number', min: 1, message: 'Must be at least 1' },
  'learning.consolidation_target': { type: 'number', min: 1, message: 'Must be at least 1' },
  'monitoring.cooldown_minutes': { type: 'number', min: 0, message: 'Must be non-negative' },
  'browser.default_timeout_ms': { type: 'number', min: 100, message: 'Must be at least 100ms' },
  'browser.viewport_width': { type: 'number', min: 100, max: 7680, message: 'Must be 100\u20137680' },
  'browser.viewport_height': { type: 'number', min: 100, max: 4320, message: 'Must be 100\u20134320' },
  'tools.max_tool_iterations_chat': { type: 'number', min: 1, max: 500, message: 'Must be 1\u2013500' },
  'tools.max_tool_iterations_loop': { type: 'number', min: 1, max: 500, message: 'Must be 1\u2013500' },
};

// Section grouping: category -> { label, icon, sections[] }
const SECTION_GROUPS = [
  { key: 'core', label: 'Core', icon: '\u2699', sections: ['timezone', 'discord', 'logging', 'permissions'] },
  { key: 'llm', label: 'LLM & AI', icon: '\uD83E\uDDE0', sections: ['openai_codex', 'context'] },
  { key: 'data', label: 'Data & Storage', icon: '\uD83D\uDCBE', sections: ['sessions', 'learning', 'search', 'usage'] },
  { key: 'services', label: 'Services', icon: '\uD83D\uDD17', sections: ['webhook', 'monitoring', 'voice', 'browser', 'comfyui'] },
  { key: 'infra', label: 'Infrastructure', icon: '\uD83D\uDEE0', sections: ['tools'] },
  { key: 'ui', label: 'Web UI', icon: '\uD83C\uDF10', sections: ['web'] },
];

const REDACTED = '\u2022\u2022\u2022\u2022\u2022\u2022\u2022\u2022';
const MAX_UNDO = 50;

function isSensitiveKey(key) {
  return SENSITIVE_KEYS.has(key);
}

function isRedacted(value) {
  return value === REDACTED;
}

function deepClone(obj) {
  return JSON.parse(JSON.stringify(obj));
}

function deepEqual(a, b) {
  return JSON.stringify(a) === JSON.stringify(b);
}

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

/** Validate a single field value. Returns error string or null. */
function validateField(section, key, value) {
  const rule = VALIDATION_RULES[section + '.' + key];
  if (!rule) return null;
  if (rule.type === 'number') {
    const n = Number(value);
    if (isNaN(n)) return 'Must be a number';
    if (rule.min !== undefined && n < rule.min) return rule.message || 'Value too low';
    if (rule.max !== undefined && n > rule.max) return rule.message || 'Value too high';
  }
  if (rule.type === 'array' && !Array.isArray(value)) {
    return rule.message || 'Must be an array';
  }
  return null;
}

/** Format diff entries for display: [{section, key, oldVal, newVal}] */
function buildDiffEntries(original, edited) {
  const entries = [];
  for (const section of Object.keys(edited)) {
    if (!(section in original)) continue;
    const orig = original[section];
    const edit = edited[section];
    if (deepEqual(orig, edit)) continue;
    if (typeof orig === 'object' && orig !== null && !Array.isArray(orig)
        && typeof edit === 'object' && edit !== null && !Array.isArray(edit)) {
      for (const k of Object.keys(edit)) {
        if (!deepEqual(orig[k], edit[k])) {
          entries.push({ section, key: k, oldVal: orig[k], newVal: edit[k] });
        }
      }
    } else {
      entries.push({ section, key: null, oldVal: orig, newVal: edit });
    }
  }
  return entries;
}

export default {
  template: `
    <div class="p-6 page-fade-in">
      <!-- Header -->
      <div class="flex items-center justify-between mb-4 flex-wrap gap-2">
        <div>
          <h1 class="text-xl font-semibold">Configuration</h1>
          <p class="text-xs text-gray-500 mt-1" v-if="config">
            {{ sectionCount }} sections across {{ groupCount }} groups
          </p>
        </div>
        <div class="flex gap-2 items-center">
          <template v-if="editing">
            <button @click="undo" class="btn btn-ghost text-xs cfg-undo-btn" :disabled="!canUndo" title="Undo (Ctrl+Z)">
              \u21A9 Undo
            </button>
            <button @click="redo" class="btn btn-ghost text-xs cfg-redo-btn" :disabled="!canRedo" title="Redo (Ctrl+Y)">
              Redo \u21AA
            </button>
            <span class="cfg-change-count" v-if="changeCount > 0">
              {{ changeCount }} change{{ changeCount !== 1 ? 's' : '' }}
            </span>
            <button @click="cancelEdit" class="btn btn-ghost text-xs">Cancel</button>
            <button @click="showDiff" class="btn btn-ghost text-xs cfg-diff-btn" :disabled="!hasChanges">
              Review
            </button>
            <button @click="saveConfig" class="btn btn-primary text-xs" :disabled="saving || !hasChanges || hasErrors">
              {{ saving ? 'Saving...' : 'Save' }}
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
      <div v-if="toast" :class="['toast', toast.type === 'success' ? 'toast-success' : 'toast-error']" role="status" aria-live="polite">
        {{ toast.message }}
      </div>

      <!-- Loading skeleton -->
      <div v-if="loading && !config" class="space-y-3">
        <div v-for="n in 4" :key="n" class="hm-card">
          <div class="skeleton skeleton-text" style="width:100px;"></div>
          <div class="skeleton skeleton-row mt-2"></div>
        </div>
      </div>

      <!-- Error state -->
      <div v-else-if="error" class="hm-card border-red-900 error-state" role="alert">
        <span class="error-icon" aria-hidden="true">\u26A0</span>
        <p class="text-red-400">{{ error }}</p>
        <button @click="fetchConfig" class="btn btn-ghost text-xs">Retry</button>
      </div>

      <!-- Grouped config sections -->
      <div v-else-if="config" class="space-y-4">
        <div v-for="group in visibleGroups" :key="group.key" class="cfg-group">
          <!-- Group header -->
          <div class="cfg-group-header cursor-pointer select-none" @click="toggleGroup(group.key)"
               role="button" tabindex="0" @keydown.enter="toggleGroup(group.key)" @keydown.space.prevent="toggleGroup(group.key)"
               :aria-expanded="!!expandedGroups[group.key]">
            <span class="cfg-group-icon" aria-hidden="true">{{ group.icon }}</span>
            <span class="cfg-group-label">{{ group.label }}</span>
            <span class="badge badge-info text-xs">{{ group.sections.length }}</span>
            <span v-if="editing && groupChanged(group)" class="badge badge-warning text-xs">modified</span>
            <span class="cfg-group-arrow" aria-hidden="true">{{ expandedGroups[group.key] ? '\u25BC' : '\u25B6' }}</span>
          </div>

          <!-- Group content -->
          <div v-if="expandedGroups[group.key]" class="cfg-group-body">
            <div v-for="section in group.sections" :key="section" class="cfg-section">
              <!-- Section header -->
              <div class="cfg-section-header cursor-pointer select-none" @click="toggleSection(section)">
                <span class="text-xs text-gray-500 font-mono">{{ expanded[section] ? '\u25BC' : '\u25B6' }}</span>
                <span class="cfg-section-name">{{ section }}</span>
                <span class="badge badge-info text-xs"
                      v-if="typeof getDisplay(section) === 'object' && getDisplay(section) !== null && !Array.isArray(getDisplay(section))">
                  {{ Object.keys(getDisplay(section)).length }} fields
                </span>
                <span v-if="editing && sectionChanged(section)" class="badge badge-warning text-xs">modified</span>
              </div>

              <div v-if="expanded[section]" class="cfg-section-body">
                <!-- Scalar top-level field (e.g. timezone) -->
                <div v-if="typeof getDisplay(section) !== 'object' || getDisplay(section) === null" class="pl-4">
                  <template v-if="editing">
                    <input class="hm-input font-mono text-sm" style="max-width:300px"
                           :value="getEdited(section)"
                           @input="pushEdit(section, null, $event.target.value)" />
                  </template>
                  <template v-else>
                    <span class="text-sm font-mono text-gray-300">{{ getDisplay(section) === '' ? '(empty)' : getDisplay(section) }}</span>
                  </template>
                </div>

                <!-- Object section -->
                <div v-else-if="!Array.isArray(getDisplay(section))">
                  <table class="hm-table">
                    <thead>
                      <tr>
                        <th class="config-key-col" style="width:30%">Key</th>
                        <th>Value</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr v-for="(v, k) in getDisplay(section)" :key="k"
                          :class="{'field-changed': editing && fieldChanged(section, k)}">
                        <td class="font-mono text-xs text-gray-400">
                          {{ k }}
                          <div v-if="getValidationError(section, k)" class="cfg-field-error">
                            {{ getValidationError(section, k) }}
                          </div>
                        </td>
                        <td>
                          <!-- Sensitive field: always masked -->
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
                                       @change="pushEdit(section, k, $event.target.checked)" />
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
                            <select class="hm-select"
                                    :value="getEditedField(section, k)"
                                    @change="pushEdit(section, k, $event.target.value)">
                              <option v-for="opt in getEnumOptions(section, k)" :key="opt" :value="opt">{{ opt }}</option>
                            </select>
                          </template>

                          <!-- Number field -->
                          <template v-else-if="typeof v === 'number'">
                            <template v-if="editing">
                              <input type="number"
                                     :class="['hm-input font-mono text-sm', getValidationError(section, k) ? 'cfg-input-error' : '']"
                                     style="max-width:200px"
                                     :value="getEditedField(section, k)"
                                     @input="pushEdit(section, k, Number($event.target.value))" />
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
                                  <textarea class="hm-input font-mono text-xs" rows="6"
                                            :value="formatJson(getEditedField(section, k))"
                                            @blur="pushEditJson(section, k, $event.target.value)"></textarea>
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
                              <input class="hm-input font-mono text-sm"
                                     :value="getEditedField(section, k)"
                                     @input="pushEdit(section, k, $event.target.value)" />
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
                  <div v-if="getDisplay(section).length === 0" class="text-gray-500 text-sm pl-4">(empty list)</div>
                  <ul v-else class="pl-4 space-y-1">
                    <li v-for="(item, i) in getDisplay(section)" :key="i" class="text-sm font-mono text-gray-300">
                      {{ typeof item === 'object' ? JSON.stringify(item) : item }}
                    </li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Ungrouped sections (fallback) -->
        <div v-for="section in ungroupedSections" :key="section" class="hm-card">
          <div class="cfg-section-header cursor-pointer select-none" @click="toggleSection(section)">
            <span class="text-xs text-gray-500 font-mono">{{ expanded[section] ? '\u25BC' : '\u25B6' }}</span>
            <span class="cfg-section-name">{{ section }}</span>
          </div>
          <div v-if="expanded[section]" class="pl-4 mt-2">
            <pre class="text-xs font-mono text-gray-300">{{ formatJson(getDisplay(section)) }}</pre>
          </div>
        </div>

        <div class="mt-4 text-xs text-gray-500">
          Fields marked <span class="badge badge-warning">sensitive</span> are redacted by the server and cannot be edited here.
        </div>
      </div>

      <!-- Diff modal -->
      <div v-if="showDiffModal" class="modal-overlay" @click.self="showDiffModal = false" role="dialog" aria-modal="true" aria-labelledby="cfg-diff-title">
        <div class="modal-content" style="max-width:700px">
          <div class="flex items-center justify-between mb-4">
            <h2 id="cfg-diff-title" class="text-lg font-semibold">Review Changes</h2>
            <button @click="showDiffModal = false" class="btn btn-ghost text-xs">\u2715</button>
          </div>
          <div v-if="diffEntries.length === 0" class="text-gray-500 text-sm py-4 text-center">No changes to review.</div>
          <div v-else class="cfg-diff-list">
            <div v-for="(entry, i) in diffEntries" :key="i" class="cfg-diff-entry">
              <div class="cfg-diff-path">
                <span class="font-mono text-xs">{{ entry.section }}</span>
                <span v-if="entry.key" class="font-mono text-xs text-gray-500">{{ '.' + entry.key }}</span>
              </div>
              <div class="cfg-diff-values">
                <div class="cfg-diff-old">
                  <span class="cfg-diff-label">\u2212</span>
                  <span class="font-mono text-xs">{{ formatDiffVal(entry.oldVal) }}</span>
                </div>
                <div class="cfg-diff-new">
                  <span class="cfg-diff-label">+</span>
                  <span class="font-mono text-xs">{{ formatDiffVal(entry.newVal) }}</span>
                </div>
              </div>
            </div>
          </div>
          <div class="flex justify-end gap-2 mt-4">
            <button @click="showDiffModal = false" class="btn btn-ghost text-xs">Close</button>
            <button @click="showDiffModal = false; saveConfig()" class="btn btn-primary text-xs" :disabled="hasErrors">
              Save Changes
            </button>
          </div>
        </div>
      </div>
    </div>`,

  setup() {
    const config = ref(null);
    const editValues = ref(null);
    const loading = ref(true);
    const error = ref(null);
    const expanded = ref({});
    const expandedNested = ref({});
    const expandedGroups = ref({});
    const editing = ref(false);
    const saving = ref(false);
    const toast = ref(null);
    const showDiffModal = ref(false);

    // Undo/redo stacks
    const undoStack = ref([]);
    const redoStack = ref([]);

    const canUndo = computed(() => undoStack.value.length > 0);
    const canRedo = computed(() => redoStack.value.length > 0);

    const displayConfig = computed(() => editing.value && editValues.value ? editValues.value : config.value);

    const hasChanges = computed(() => {
      if (!config.value || !editValues.value) return false;
      return !deepEqual(config.value, editValues.value);
    });

    const changeCount = computed(() => {
      if (!config.value || !editValues.value) return 0;
      return buildDiffEntries(config.value, editValues.value).length;
    });

    // Validation errors: { 'section.key': 'error message' }
    const validationErrors = computed(() => {
      if (!editing.value || !editValues.value) return {};
      const errors = {};
      for (const section of Object.keys(editValues.value)) {
        const val = editValues.value[section];
        if (typeof val === 'object' && val !== null && !Array.isArray(val)) {
          for (const k of Object.keys(val)) {
            const err = validateField(section, k, val[k]);
            if (err) errors[section + '.' + k] = err;
          }
        }
      }
      return errors;
    });

    const hasErrors = computed(() => Object.keys(validationErrors.value).length > 0);

    const sectionCount = computed(() => config.value ? Object.keys(config.value).length : 0);
    const groupCount = computed(() => visibleGroups.value.length);

    const diffEntries = computed(() => {
      if (!config.value || !editValues.value) return [];
      return buildDiffEntries(config.value, editValues.value);
    });

    // Groups that have at least one section present in the config
    const visibleGroups = computed(() => {
      if (!config.value) return [];
      return SECTION_GROUPS.map(g => ({
        ...g,
        sections: g.sections.filter(s => s in config.value),
      })).filter(g => g.sections.length > 0);
    });

    // Sections not in any group
    const ungroupedSections = computed(() => {
      if (!config.value) return [];
      const grouped = new Set(SECTION_GROUPS.flatMap(g => g.sections));
      return Object.keys(config.value).filter(s => !grouped.has(s));
    });

    function getDisplay(section) {
      return displayConfig.value ? displayConfig.value[section] : null;
    }

    function sectionChanged(section) {
      if (!config.value || !editValues.value) return false;
      return !deepEqual(config.value[section], editValues.value[section]);
    }

    function groupChanged(group) {
      return group.sections.some(s => sectionChanged(s));
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

    function getValidationError(section, key) {
      return validationErrors.value[section + '.' + key] || null;
    }

    function getEnumOptions(section, key) {
      return ENUM_FIELDS[section + '.' + key] || null;
    }

    // --- Undo-aware edit operations ---

    function _applyEdit(section, key, value) {
      if (!editValues.value) return;
      if (key === null) {
        editValues.value[section] = value;
      } else {
        if (!editValues.value[section]) editValues.value[section] = {};
        editValues.value[section][key] = value;
      }
      editValues.value = { ...editValues.value };
    }

    function pushEdit(section, key, value) {
      if (!editValues.value) return;
      // Capture snapshot before change for undo
      const snapshot = deepClone(editValues.value);
      _applyEdit(section, key, value);
      undoStack.value.push(snapshot);
      if (undoStack.value.length > MAX_UNDO) undoStack.value.shift();
      redoStack.value = []; // clear redo on new edit
    }

    function pushEditJson(section, key, jsonStr) {
      try {
        const parsed = JSON.parse(jsonStr);
        pushEdit(section, key, parsed);
      } catch {
        // Invalid JSON — keep old value
      }
    }

    function undo() {
      if (undoStack.value.length === 0) return;
      redoStack.value.push(deepClone(editValues.value));
      editValues.value = undoStack.value.pop();
    }

    function redo() {
      if (redoStack.value.length === 0) return;
      undoStack.value.push(deepClone(editValues.value));
      editValues.value = redoStack.value.pop();
    }

    function removeArrayItem(section, key, index) {
      if (!editValues.value || !editValues.value[section]) return;
      const arr = [...editValues.value[section][key]];
      arr.splice(index, 1);
      pushEdit(section, key, arr);
    }

    function addArrayItem(section, key) {
      if (!editValues.value || !editValues.value[section]) return;
      const arr = [...(editValues.value[section][key] || [])];
      const val = prompt('Enter new value:');
      if (val === null) return;
      arr.push(val);
      pushEdit(section, key, arr);
    }

    function toggleSection(section) {
      expanded.value = { ...expanded.value, [section]: !expanded.value[section] };
    }

    function toggleGroup(key) {
      expandedGroups.value = { ...expandedGroups.value, [key]: !expandedGroups.value[key] };
    }

    function toggleNested(key) {
      expandedNested.value = { ...expandedNested.value, [key]: !expandedNested.value[key] };
    }

    function formatJson(obj) {
      try { return JSON.stringify(obj, null, 2); }
      catch { return String(obj); }
    }

    function formatDiffVal(val) {
      if (val === null || val === undefined) return 'null';
      if (typeof val === 'object') return JSON.stringify(val, null, 2);
      return String(val);
    }

    function showToast(type, message) {
      toast.value = { type, message };
      setTimeout(() => { toast.value = null; }, 3000);
    }

    function startEdit() {
      editValues.value = deepClone(config.value);
      editing.value = true;
      undoStack.value = [];
      redoStack.value = [];
    }

    function cancelEdit() {
      editing.value = false;
      editValues.value = null;
      undoStack.value = [];
      redoStack.value = [];
    }

    function showDiff() {
      showDiffModal.value = true;
    }

    async function saveConfig() {
      if (!hasChanges.value || hasErrors.value) return;
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
        undoStack.value = [];
        redoStack.value = [];
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
        // Expand all groups by default
        for (const g of SECTION_GROUPS) {
          if (expandedGroups.value[g.key] === undefined) {
            expandedGroups.value[g.key] = true;
          }
        }
      } catch (e) {
        error.value = e.message;
      }
      loading.value = false;
    }

    // Keyboard shortcuts for undo/redo
    function handleKeydown(e) {
      if (!editing.value) return;
      if ((e.ctrlKey || e.metaKey) && !e.shiftKey && e.key === 'z') {
        e.preventDefault();
        undo();
      } else if ((e.ctrlKey || e.metaKey) && (e.key === 'y' || (e.shiftKey && e.key === 'z') || (e.shiftKey && e.key === 'Z'))) {
        e.preventDefault();
        redo();
      }
    }

    onMounted(() => {
      fetchConfig();
      document.addEventListener('keydown', handleKeydown);
    });

    onUnmounted(() => {
      document.removeEventListener('keydown', handleKeydown);
    });

    return {
      config, displayConfig, editValues, loading, error,
      expanded, expandedNested, expandedGroups, editing, saving, toast,
      hasChanges, hasErrors, changeCount, REDACTED,
      showDiffModal, diffEntries,
      canUndo, canRedo,
      sectionCount, groupCount, visibleGroups, ungroupedSections,
      validationErrors,
      isSensitiveKey, isRedacted, sectionChanged, groupChanged, fieldChanged,
      getDisplay, getEdited, getEditedField, getDisplayBool,
      pushEdit, pushEditJson, getValidationError,
      getEnumOptions, removeArrayItem, addArrayItem,
      toggleSection, toggleGroup, toggleNested, formatJson, formatDiffVal,
      showToast, showDiff,
      fetchConfig, startEdit, cancelEdit, saveConfig, undo, redo,
    };
  },
};
