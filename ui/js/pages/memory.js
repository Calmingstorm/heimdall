/**
 * Loki Management UI — Memory Page
 * Tree view of persistent memory (global + per-user scopes)
 */
import { api } from '../api.js';

const { ref, computed, onMounted } = Vue;

export default {
  template: `
    <div class="p-6">
      <div class="flex items-center justify-between mb-4">
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

      <!-- Add form -->
      <div v-if="showAdd" class="loki-card mb-4">
        <h2 class="text-sm font-medium mb-3">Add Memory Entry</h2>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-3 mb-3">
          <div>
            <label class="text-gray-400 text-xs block mb-1">Scope</label>
            <input v-model="addForm.scope" type="text" class="loki-input"
                   placeholder="e.g. global, user:12345" />
          </div>
          <div>
            <label class="text-gray-400 text-xs block mb-1">Key</label>
            <input v-model="addForm.key" type="text" class="loki-input"
                   placeholder="e.g. preferred_language" />
          </div>
        </div>
        <div class="mb-3">
          <label class="text-gray-400 text-xs block mb-1">Value</label>
          <textarea v-model="addForm.value" class="loki-input" rows="3"
                    placeholder="Enter value..."></textarea>
        </div>
        <div v-if="addError" class="mb-3 text-red-400 text-sm">{{ addError }}</div>
        <div v-if="addSuccess" class="mb-3 text-green-400 text-sm">{{ addSuccess }}</div>
        <button @click="doAdd" class="btn btn-primary text-xs" :disabled="adding">
          {{ adding ? 'Saving...' : 'Save' }}
        </button>
      </div>

      <!-- Loading / error -->
      <div v-if="loading && scopes.length === 0" class="flex items-center gap-2 text-gray-400">
        <div class="spinner"></div> Loading memory...
      </div>
      <div v-else-if="error" class="loki-card border-red-900">
        <p class="text-red-400">{{ error }}</p>
      </div>
      <div v-else-if="scopes.length === 0 && !showAdd" class="loki-card">
        <p class="text-gray-400">No memory entries. Click "Add Entry" to create one.</p>
      </div>

      <!-- Scope tree -->
      <div v-else class="space-y-3">
        <div v-for="scope in scopes" :key="scope.name" class="loki-card">
          <div class="flex items-center gap-2 mb-2 cursor-pointer select-none"
               @click="toggleScope(scope.name)">
            <span class="text-xs text-gray-500 font-mono">{{ expanded[scope.name] ? '\u25BC' : '\u25B6' }}</span>
            <span class="text-sm font-medium">{{ scope.name }}</span>
            <span class="badge badge-info text-xs">{{ scope.count }} keys</span>
          </div>

          <div v-if="expanded[scope.name]">
            <div v-if="loadingScope === scope.name" class="flex items-center gap-2 text-gray-400 text-sm pl-4">
              <div class="spinner" style="width:14px;height:14px;border-width:2px;"></div> Loading...
            </div>
            <div v-else-if="scopeEntries[scope.name]">
              <table class="loki-table">
                <thead>
                  <tr>
                    <th style="width:30%">Key</th>
                    <th>Value</th>
                    <th style="width:120px">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="entry in scopeEntries[scope.name]" :key="entry.key">
                    <td class="font-mono text-xs text-gray-400">{{ entry.key }}</td>
                    <td>
                      <div v-if="editingKey === scope.name + '/' + entry.key">
                        <textarea v-model="editValue" class="loki-input text-sm" rows="2"></textarea>
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

      <!-- Delete confirmation -->
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

    // Delete
    const deleteTarget = ref(null);
    const deleting = ref(false);

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
      // Load entries for this scope
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
      try {
        await api.put(`/api/memory/${encodeURIComponent(scope)}/${encodeURIComponent(key)}`, {
          value: editValue.value,
        });
        // Update local state
        const entries = scopeEntries.value[scope];
        if (entries) {
          const e = entries.find(e => e.key === key);
          if (e) e.value = editValue.value;
        }
        editingKey.value = null;
      } catch { /* ignore */ }
      saving.value = false;
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
        // Invalidate scope cache and re-fetch
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
      const { scope, key } = deleteTarget.value;
      try {
        await api.del(`/api/memory/${encodeURIComponent(scope)}/${encodeURIComponent(key)}`);
        // Remove from local state
        const entries = scopeEntries.value[scope];
        if (entries) {
          scopeEntries.value[scope] = entries.filter(e => e.key !== key);
        }
        // Update scope count
        const s = scopes.value.find(s => s.name === scope);
        if (s) {
          s.count--;
          s.keys = s.keys.filter(k => k !== key);
        }
      } catch { /* ignore */ }
      deleting.value = false;
      deleteTarget.value = null;
    }

    onMounted(() => { fetchMemory(); });

    return {
      scopes, scopeEntries, loading, error, expanded, loadingScope,
      showAdd, addForm, adding, addError, addSuccess,
      editingKey, editValue, saving,
      deleteTarget, deleting,
      fetchMemory, toggleScope, startEdit, doEdit, doAdd, confirmDelete, doDelete,
    };
  },
};
