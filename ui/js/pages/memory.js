/**
 * Heimdall Management UI — Memory Page
 * Table view of persistent memory with copy, scope badges, and bulk delete
 */
import { api } from '../api.js';

const { ref, computed, onMounted } = Vue;

export default {
  template: `
    <div class="p-6 page-fade-in">
      <div class="flex items-center justify-between mb-4 flex-wrap gap-2">
        <h1 class="text-xl font-semibold">Memory</h1>
        <div class="flex gap-2">
          <button @click="showAdd = !showAdd" class="btn btn-primary text-xs">
            {{ showAdd ? 'Cancel' : 'Add Entry' }}
          </button>
          <button @click="fetchMemory" class="btn btn-ghost text-xs" :disabled="loading">
            {{ loading ? 'Loading...' : 'Refresh' }}
          </button>
        </div>
      </div>

      <!-- Summary stats -->
      <div v-if="!loading && scopes.length > 0" class="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
        <div class="hm-card text-center">
          <div class="text-2xl font-bold">{{ totalEntries }}</div>
          <div class="text-gray-400 text-xs">Total Entries</div>
        </div>
        <div class="hm-card text-center">
          <div class="text-2xl font-bold">{{ scopes.length }}</div>
          <div class="text-gray-400 text-xs">Scopes</div>
        </div>
        <div class="hm-card text-center">
          <div class="text-2xl font-bold">{{ selectedCount }}</div>
          <div class="text-gray-400 text-xs">Selected</div>
        </div>
        <div class="hm-card text-center">
          <button v-if="selectedCount > 0" @click="confirmBulkDelete"
                  class="btn btn-danger text-xs">
            Delete Selected ({{ selectedCount }})
          </button>
          <span v-else class="text-gray-600 text-xs">Select entries to delete</span>
        </div>
      </div>

      <!-- Add form -->
      <div v-if="showAdd" class="hm-card mb-4">
        <h2 class="text-sm font-medium mb-3">Add Memory Entry</h2>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-3 mb-3">
          <div>
            <label class="text-gray-400 text-xs block mb-1">Scope</label>
            <input v-model="addForm.scope" type="text" class="hm-input"
                   placeholder="e.g. global, user:12345" />
          </div>
          <div>
            <label class="text-gray-400 text-xs block mb-1">Key</label>
            <input v-model="addForm.key" type="text" class="hm-input"
                   placeholder="e.g. preferred_language" />
          </div>
        </div>
        <div class="mb-3">
          <label class="text-gray-400 text-xs block mb-1">Value</label>
          <textarea v-model="addForm.value" class="hm-input" rows="3"
                    placeholder="Enter value..."></textarea>
        </div>
        <div v-if="addError" class="mb-3 text-red-400 text-sm">{{ addError }}</div>
        <div v-if="addSuccess" class="mb-3 text-green-400 text-sm">{{ addSuccess }}</div>
        <button @click="doAdd" class="btn btn-primary text-xs" :disabled="adding">
          {{ adding ? 'Saving...' : 'Save' }}
        </button>
      </div>

      <!-- Action error toast -->
      <div v-if="actionError" class="hm-card border-red-900 mb-4">
        <div class="flex items-center justify-between">
          <p class="text-red-400 text-sm">{{ actionError }}</p>
          <button @click="actionError = null" class="btn btn-ghost text-xs">Dismiss</button>
        </div>
      </div>

      <!-- Loading / error -->
      <div v-if="loading && scopes.length === 0" class="space-y-2">
        <div v-for="n in 3" :key="n" class="skeleton skeleton-row"></div>
      </div>
      <div v-else-if="error" class="hm-card border-red-900 error-state">
        <span class="error-icon">\u26A0</span>
        <p class="text-red-400">{{ error }}</p>
        <button @click="fetchMemory" class="btn btn-ghost text-xs">Retry</button>
      </div>
      <div v-else-if="scopes.length === 0 && !showAdd" class="hm-card empty-state">
        <span class="empty-state-icon">\u{1F9E0}</span>
        <span class="empty-state-text">No memory entries</span>
        <span class="empty-state-hint">Click "Add Entry" or let Heimdall learn preferences through conversations</span>
      </div>

      <!-- Memory table per scope -->
      <div v-else class="space-y-4">
        <div v-for="scope in scopes" :key="scope.name" class="hm-card">
          <div class="flex items-center gap-2 mb-3 cursor-pointer select-none"
               @click="toggleScope(scope.name)">
            <span class="text-xs text-gray-500 font-mono">{{ expanded[scope.name] ? '\u25BC' : '\u25B6' }}</span>
            <span class="memory-scope-badge"
                  :class="scope.name === 'global' ? 'memory-scope-global' : 'memory-scope-user'">
              {{ scope.name }}
            </span>
            <span class="badge badge-info text-xs">{{ scope.count }} keys</span>
          </div>

          <div v-if="expanded[scope.name]">
            <div v-if="loadingScope === scope.name" class="flex items-center gap-2 text-gray-400 text-sm pl-4">
              <div class="spinner" style="width:14px;height:14px;border-width:2px;"></div> Loading...
            </div>
            <div v-else-if="scopeEntries[scope.name]" class="table-responsive">
              <table class="hm-table">
                <thead>
                  <tr>
                    <th style="width:30px">
                      <input type="checkbox"
                             :checked="isScopeAllSelected(scope.name)"
                             @change="toggleSelectAll(scope.name, $event.target.checked)"
                             class="memory-checkbox" />
                    </th>
                    <th style="width:25%">Key</th>
                    <th>Value</th>
                    <th style="width:140px">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="entry in scopeEntries[scope.name]" :key="entry.key">
                    <td>
                      <input type="checkbox"
                             :checked="isSelected(scope.name, entry.key)"
                             @change="toggleSelect(scope.name, entry.key)"
                             class="memory-checkbox" />
                    </td>
                    <td class="font-mono text-xs text-gray-400">{{ entry.key }}</td>
                    <td>
                      <div v-if="editingKey === scope.name + '/' + entry.key">
                        <textarea v-model="editValue" class="hm-input text-sm" rows="2"></textarea>
                        <div class="flex gap-1 mt-1">
                          <button @click="doEdit(scope.name, entry.key)" class="btn btn-primary text-xs" :disabled="saving">
                            {{ saving ? 'Saving...' : 'Save' }}
                          </button>
                          <button @click="editingKey = null" class="btn btn-ghost text-xs">Cancel</button>
                        </div>
                      </div>
                      <div v-else class="text-sm text-gray-300 whitespace-pre-wrap break-words"
                           style="max-height:6rem;overflow:hidden;">
                        {{ entry.value }}
                      </div>
                    </td>
                    <td>
                      <div class="flex gap-1">
                        <button @click="copyValue(entry.value)" class="btn btn-ghost text-xs"
                                :title="'Copy value'">
                          {{ copied === scope.name + '/' + entry.key ? 'Copied!' : 'Copy' }}
                        </button>
                        <button @click="startEdit(scope.name, entry.key, entry.value)" class="btn btn-ghost text-xs">Edit</button>
                        <button @click="confirmDelete(scope.name, entry.key)" class="btn btn-danger text-xs">Del</button>
                      </div>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

      <!-- Delete confirmation (single) -->
      <div v-if="deleteTarget" class="modal-overlay" @click.self="deleteTarget = null">
        <div class="modal-content">
          <h3 class="text-lg font-semibold mb-2">Delete Memory Entry</h3>
          <p class="text-gray-400 text-sm mb-4">
            Delete <span class="font-mono font-semibold text-gray-200">{{ deleteTarget.scope }}/{{ deleteTarget.key }}</span>? This cannot be undone.
          </p>
          <div class="flex gap-2 justify-end">
            <button @click="deleteTarget = null" class="btn btn-ghost">Cancel</button>
            <button @click="doDelete" class="btn btn-danger" :disabled="deleting">
              {{ deleting ? 'Deleting...' : 'Delete' }}
            </button>
          </div>
        </div>
      </div>

      <!-- Bulk delete confirmation -->
      <div v-if="showBulkDelete" class="modal-overlay" @click.self="showBulkDelete = false">
        <div class="modal-content">
          <h3 class="text-lg font-semibold mb-2">Bulk Delete</h3>
          <p class="text-gray-400 text-sm mb-4">
            Delete <span class="font-semibold text-gray-200">{{ selectedCount }}</span> selected entries? This cannot be undone.
          </p>
          <div class="flex gap-2 justify-end">
            <button @click="showBulkDelete = false" class="btn btn-ghost">Cancel</button>
            <button @click="doBulkDelete" class="btn btn-danger" :disabled="deleting">
              {{ deleting ? 'Deleting...' : 'Delete All Selected' }}
            </button>
          </div>
        </div>
      </div>
    </div>`,

  setup() {
    const scopes = ref([]);
    const scopeEntries = ref({});
    const loading = ref(true);
    const error = ref(null);
    const expanded = ref({});
    const loadingScope = ref(null);

    // Add form
    const showAdd = ref(false);
    const addForm = ref({ scope: 'global', key: '', value: '' });
    const adding = ref(false);
    const addError = ref(null);
    const addSuccess = ref(null);

    // Edit
    const editingKey = ref(null);
    const editValue = ref('');
    const saving = ref(false);
    const actionError = ref(null);

    // Copy
    const copied = ref(null);

    // Selection
    const selected = ref(new Set());

    // Delete
    const deleteTarget = ref(null);
    const deleting = ref(false);
    const showBulkDelete = ref(false);

    const totalEntries = computed(() => scopes.value.reduce((sum, s) => sum + s.count, 0));
    const selectedCount = computed(() => selected.value.size);

    function isSelected(scope, key) {
      return selected.value.has(scope + '/' + key);
    }

    function toggleSelect(scope, key) {
      const id = scope + '/' + key;
      const next = new Set(selected.value);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      selected.value = next;
    }

    function isScopeAllSelected(scopeName) {
      const entries = scopeEntries.value[scopeName];
      if (!entries || entries.length === 0) return false;
      return entries.every(e => selected.value.has(scopeName + '/' + e.key));
    }

    function toggleSelectAll(scopeName, checked) {
      const entries = scopeEntries.value[scopeName];
      if (!entries) return;
      const next = new Set(selected.value);
      for (const e of entries) {
        const id = scopeName + '/' + e.key;
        if (checked) next.add(id);
        else next.delete(id);
      }
      selected.value = next;
    }

    async function fetchMemory() {
      loading.value = true;
      error.value = null;
      try {
        const data = await api.get('/api/memory');
        scopes.value = Object.entries(data).map(([name, info]) => ({
          name,
          keys: info.keys || [],
          count: info.count || 0,
        }));
      } catch (e) {
        error.value = e.message;
      }
      loading.value = false;
    }

    async function toggleScope(scopeName) {
      if (expanded.value[scopeName]) {
        expanded.value[scopeName] = false;
        return;
      }
      expanded.value[scopeName] = true;
      const scope = scopes.value.find(s => s.name === scopeName);
      if (!scope || scopeEntries.value[scopeName]) return;

      loadingScope.value = scopeName;
      const entries = [];
      for (const key of scope.keys) {
        try {
          const data = await api.get(`/api/memory/${encodeURIComponent(scopeName)}/${encodeURIComponent(key)}`);
          entries.push({ key, value: data.value || '' });
        } catch {
          entries.push({ key, value: '(error loading)' });
        }
      }
      scopeEntries.value[scopeName] = entries;
      loadingScope.value = null;
    }

    function startEdit(scope, key, value) {
      editingKey.value = scope + '/' + key;
      editValue.value = value;
    }

    async function doEdit(scope, key) {
      saving.value = true;
      actionError.value = null;
      try {
        await api.put(`/api/memory/${encodeURIComponent(scope)}/${encodeURIComponent(key)}`, {
          value: editValue.value,
        });
        const entries = scopeEntries.value[scope];
        if (entries) {
          const e = entries.find(e => e.key === key);
          if (e) e.value = editValue.value;
        }
        editingKey.value = null;
      } catch (e) {
        actionError.value = `Failed to save: ${e.message || 'unknown error'}`;
      }
      saving.value = false;
    }

    async function copyValue(value) {
      try {
        await navigator.clipboard.writeText(value);
        // Find which entry was copied for feedback
        for (const [scopeName, entries] of Object.entries(scopeEntries.value)) {
          const entry = entries.find(e => e.value === value);
          if (entry) {
            copied.value = scopeName + '/' + entry.key;
            setTimeout(() => { copied.value = null; }, 1500);
            return;
          }
        }
      } catch { /* clipboard not available */ }
    }

    async function doAdd() {
      addError.value = null;
      addSuccess.value = null;
      const s = addForm.value.scope.trim();
      const k = addForm.value.key.trim();
      const v = addForm.value.value.trim();
      if (!s) { addError.value = 'Scope is required'; return; }
      if (!k) { addError.value = 'Key is required'; return; }
      if (!v) { addError.value = 'Value is required'; return; }

      adding.value = true;
      try {
        await api.put(`/api/memory/${encodeURIComponent(s)}/${encodeURIComponent(k)}`, { value: v });
        addSuccess.value = 'Entry saved';
        addForm.value = { scope: 'global', key: '', value: '' };
        scopeEntries.value = {};
        await fetchMemory();
        setTimeout(() => { showAdd.value = false; addSuccess.value = null; }, 800);
      } catch (e) {
        addError.value = e.message;
      }
      adding.value = false;
    }

    function confirmDelete(scope, key) {
      deleteTarget.value = { scope, key };
    }

    async function doDelete() {
      if (!deleteTarget.value) return;
      deleting.value = true;
      actionError.value = null;
      const { scope, key } = deleteTarget.value;
      try {
        await api.del(`/api/memory/${encodeURIComponent(scope)}/${encodeURIComponent(key)}`);
        const entries = scopeEntries.value[scope];
        if (entries) {
          scopeEntries.value[scope] = entries.filter(e => e.key !== key);
        }
        const s = scopes.value.find(s => s.name === scope);
        if (s) {
          s.count--;
          s.keys = s.keys.filter(k => k !== key);
        }
        // Remove from selection
        const next = new Set(selected.value);
        next.delete(scope + '/' + key);
        selected.value = next;
      } catch (e) {
        actionError.value = `Failed to delete: ${e.message || 'unknown error'}`;
      }
      deleting.value = false;
      deleteTarget.value = null;
    }

    function confirmBulkDelete() {
      showBulkDelete.value = true;
    }

    async function doBulkDelete() {
      deleting.value = true;
      actionError.value = null;
      const entries = [];
      for (const id of selected.value) {
        const slash = id.indexOf('/');
        entries.push({ scope: id.slice(0, slash), key: id.slice(slash + 1) });
      }
      try {
        await api.post('/api/memory/bulk-delete', { entries });
        selected.value = new Set();
        scopeEntries.value = {};
        await fetchMemory();
      } catch (e) {
        actionError.value = `Bulk delete failed: ${e.message || 'unknown error'}`;
      }
      deleting.value = false;
      showBulkDelete.value = false;
    }

    onMounted(() => { fetchMemory(); });

    return {
      scopes, scopeEntries, loading, error, expanded, loadingScope,
      showAdd, addForm, adding, addError, addSuccess,
      editingKey, editValue, saving, actionError,
      copied,
      selected, selectedCount, totalEntries,
      deleteTarget, deleting, showBulkDelete,
      fetchMemory, toggleScope, startEdit, doEdit, copyValue, doAdd,
      confirmDelete, doDelete, confirmBulkDelete, doBulkDelete,
      isSelected, toggleSelect, isScopeAllSelected, toggleSelectAll,
    };
  },
};
