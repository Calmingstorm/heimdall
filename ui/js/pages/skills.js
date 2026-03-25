/**
 * Loki Management UI — Skills Page
 * List, create, edit, delete skills with code editor
 */
import { api } from '../api.js';

const { ref, onMounted } = Vue;

export default {
  template: `
    <div class="p-6">
      <div class="flex items-center justify-between mb-4">
        <h1 class="text-xl font-semibold">Skills</h1>
        <div class="flex gap-2">
          <button @click="showCreate" class="btn btn-primary text-xs">New Skill</button>
          <button @click="fetchSkills" class="btn btn-ghost text-xs" :disabled="loading">
            {{ loading ? 'Loading...' : 'Refresh' }}
          </button>
        </div>
      </div>

      <div v-if="loading && skills.length === 0" class="space-y-2">
        <div v-for="n in 4" :key="n" class="skeleton skeleton-row"></div>
      </div>
      <div v-else-if="error" class="loki-card border-red-900 error-state">
        <p class="text-red-400">{{ error }}</p>
        <button @click="fetchSkills" class="btn btn-ghost text-xs">Retry</button>
      </div>
      <div v-else-if="skills.length === 0 && !editing" class="loki-card">
        <p class="text-gray-400">No skills loaded. Create one to get started.</p>
      </div>
      <div v-else-if="!editing">
        <table class="loki-table">
          <thead>
            <tr>
              <th>Name</th>
              <th>Description</th>
              <th>Tools</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="s in skills" :key="s.name">
              <td class="font-mono text-sm whitespace-nowrap">{{ s.name }}</td>
              <td class="text-gray-400 text-sm">{{ s.description || '—' }}</td>
              <td>
                <span class="badge badge-info">{{ s.tool_count || 0 }}</span>
              </td>
              <td class="whitespace-nowrap">
                <button @click="editSkill(s)" class="btn btn-ghost text-xs">Edit</button>
                <button @click="confirmDelete(s.name)" class="btn btn-danger text-xs ml-1">Delete</button>
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <!-- Create/Edit form -->
      <div v-if="editing" class="loki-card mt-4">
        <div class="flex items-center justify-between mb-3">
          <h2 class="text-sm font-medium">
            {{ editMode === 'create' ? 'Create Skill' : 'Edit Skill: ' + editName }}
          </h2>
          <button @click="cancelEdit" class="btn btn-ghost text-xs">Cancel</button>
        </div>

        <div v-if="editMode === 'create'" class="mb-3">
          <label class="text-gray-400 text-xs block mb-1">Name</label>
          <input v-model="editName" type="text" class="loki-input" placeholder="my_skill" />
        </div>

        <div class="mb-3">
          <label class="text-gray-400 text-xs block mb-1">Code</label>
          <textarea v-model="editCode" class="loki-input" rows="16"
                    placeholder="# Skill code here..."></textarea>
        </div>

        <div v-if="editError" class="mb-3 text-red-400 text-sm">{{ editError }}</div>
        <div v-if="editSuccess" class="mb-3 text-green-400 text-sm">{{ editSuccess }}</div>

        <div class="flex gap-2">
          <button @click="saveSkill" class="btn btn-primary text-xs" :disabled="saving">
            {{ saving ? 'Saving...' : (editMode === 'create' ? 'Create' : 'Save') }}
          </button>
          <button @click="cancelEdit" class="btn btn-ghost text-xs">Cancel</button>
        </div>
      </div>

      <!-- Delete confirmation -->
      <div v-if="deleteTarget" class="modal-overlay" @click.self="deleteTarget = null">
        <div class="modal-content">
          <h3 class="text-lg font-semibold mb-2">Delete Skill</h3>
          <p class="text-gray-400 text-sm mb-4">
            Delete skill <span class="font-mono font-semibold text-gray-200">{{ deleteTarget }}</span>? This cannot be undone.
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
    const skills = ref([]);
    const loading = ref(true);
    const error = ref(null);

    // Editor state
    const editing = ref(false);
    const editMode = ref('create'); // 'create' | 'edit'
    const editName = ref('');
    const editCode = ref('');
    const editError = ref(null);
    const editSuccess = ref(null);
    const saving = ref(false);

    // Delete state
    const deleteTarget = ref(null);
    const deleting = ref(false);

    async function fetchSkills() {
      loading.value = true;
      error.value = null;
      try {
        skills.value = await api.get('/api/skills');
      } catch (e) {
        error.value = e.message;
      }
      loading.value = false;
    }

    function showCreate() {
      editing.value = true;
      editMode.value = 'create';
      editName.value = '';
      editCode.value = '';
      editError.value = null;
      editSuccess.value = null;
    }

    function editSkill(skill) {
      editing.value = true;
      editMode.value = 'edit';
      editName.value = skill.name;
      editCode.value = skill.code || '';
      editError.value = null;
      editSuccess.value = null;
    }

    function cancelEdit() {
      editing.value = false;
      editError.value = null;
      editSuccess.value = null;
    }

    async function saveSkill() {
      editError.value = null;
      editSuccess.value = null;
      const name = editName.value.trim();
      const code = editCode.value.trim();
      if (!name) { editError.value = 'Name is required'; return; }
      if (!code) { editError.value = 'Code is required'; return; }

      saving.value = true;
      try {
        if (editMode.value === 'create') {
          await api.post('/api/skills', { name, code });
          editSuccess.value = 'Skill created successfully';
        } else {
          await api.put(`/api/skills/${encodeURIComponent(name)}`, { code });
          editSuccess.value = 'Skill updated successfully';
        }
        await fetchSkills();
        // Auto-close after brief delay
        setTimeout(() => { editing.value = false; }, 800);
      } catch (e) {
        editError.value = e.message;
      }
      saving.value = false;
    }

    function confirmDelete(name) {
      deleteTarget.value = name;
    }

    async function doDelete() {
      if (!deleteTarget.value) return;
      deleting.value = true;
      try {
        await api.del(`/api/skills/${encodeURIComponent(deleteTarget.value)}`);
        await fetchSkills();
      } catch { /* ignore */ }
      deleting.value = false;
      deleteTarget.value = null;
    }

    onMounted(() => { fetchSkills(); });

    return {
      skills, loading, error,
      editing, editMode, editName, editCode, editError, editSuccess, saving,
      deleteTarget, deleting,
      fetchSkills, showCreate, editSkill, cancelEdit, saveSkill,
      confirmDelete, doDelete,
    };
  },
};
