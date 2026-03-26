/**
 * Loki Management UI — Skills Page
 * List, create, edit, delete, test skills with syntax highlighting
 */
import { api } from '../api.js';

const { ref, computed, onMounted, nextTick, watch } = Vue;

// Basic Python keyword highlighting (no external dependency)
function highlightPython(code) {
  if (!code) return '';
  // Escape HTML first
  let html = code
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
  // Strings (triple-quoted first, then single/double)
  html = html.replace(/("""[\s\S]*?"""|'''[\s\S]*?'''|"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')/g,
    '<span class="sk-str">$1</span>');
  // Comments
  html = html.replace(/(#[^\n]*)/g, '<span class="sk-cmt">$1</span>');
  // Keywords
  const kw = '\\b(def|class|return|if|elif|else|for|while|import|from|as|try|except|finally|raise|with|async|await|yield|pass|break|continue|and|or|not|in|is|None|True|False|self|lambda)\\b';
  html = html.replace(new RegExp(kw, 'g'), '<span class="sk-kw">$1</span>');
  // Decorators
  html = html.replace(/(@\w+)/g, '<span class="sk-dec">$1</span>');
  // Numbers
  html = html.replace(/\b(\d+\.?\d*)\b/g, '<span class="sk-num">$1</span>');
  return html;
}

export default {
  template: `
    <div class="p-6 page-fade-in">
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
        <span class="error-icon">\u26A0</span>
        <p class="text-red-400">{{ error }}</p>
        <button @click="fetchSkills" class="btn btn-ghost text-xs">Retry</button>
      </div>
      <div v-else-if="skills.length === 0 && !editing" class="loki-card empty-state">
        <span class="empty-state-icon">\u{1F9E9}</span>
        <span class="empty-state-text">No skills loaded</span>
        <span class="empty-state-hint">Click "New Skill" to create a custom tool</span>
      </div>
      <div v-else-if="!editing">
        <!-- Skill cards -->
        <div class="space-y-3">
          <div v-for="s in skills" :key="s.name" class="loki-card skill-card">
            <div class="flex items-center justify-between mb-2">
              <div class="flex items-center gap-2">
                <span class="font-mono text-sm font-semibold">{{ s.name }}</span>
                <span v-if="s.execution_count > 0" class="text-gray-500 text-xs font-mono">
                  {{ s.execution_count.toLocaleString() }} runs
                </span>
              </div>
              <div class="flex gap-1">
                <button @click="testSkill(s.name)"
                        class="btn btn-ghost text-xs"
                        :disabled="testing === s.name">
                  {{ testing === s.name ? 'Testing...' : 'Test' }}
                </button>
                <button @click="toggleCode(s.name)" class="btn btn-ghost text-xs">
                  {{ showCode[s.name] ? 'Hide Code' : 'View Code' }}
                </button>
                <button @click="editSkill(s)" class="btn btn-ghost text-xs">Edit</button>
                <button @click="confirmDelete(s.name)" class="btn btn-danger text-xs">Delete</button>
              </div>
            </div>
            <div class="text-gray-400 text-sm mb-1">{{ s.description || 'No description' }}</div>
            <div class="text-gray-600 text-xs">Loaded: {{ formatDate(s.loaded_at) }}</div>

            <!-- Test result -->
            <div v-if="testResults[s.name]" class="mt-2 p-2 rounded text-sm font-mono"
                 :class="testResults[s.name].is_error ? 'bg-red-950/30 border border-red-900/50 text-red-400' : 'bg-green-950/30 border border-green-900/50 text-green-400'">
              <div class="text-xs font-sans mb-1" :class="testResults[s.name].is_error ? 'text-red-500' : 'text-green-500'">
                {{ testResults[s.name].is_error ? 'Test Failed' : 'Test Passed' }}
              </div>
              <div class="whitespace-pre-wrap text-xs">{{ truncate(testResults[s.name].result, 500) }}</div>
            </div>

            <!-- Code preview with syntax highlighting -->
            <div v-if="showCode[s.name] && s.code" class="mt-2">
              <pre class="skill-code-block"><code v-html="highlight(s.code)"></code></pre>
            </div>
          </div>
        </div>
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
          <input v-model="editName" type="text" class="loki-input" placeholder="my_skill"
                 style="max-width:300px" />
          <div class="text-gray-600 text-xs mt-1">Lowercase, alphanumeric + underscores, starts with letter</div>
        </div>

        <div class="mb-3">
          <label class="text-gray-400 text-xs block mb-1">Code</label>
          <textarea v-model="editCode" class="loki-input skill-editor" rows="20"
                    placeholder="# Skill code here...&#10;&#10;SKILL_DEFINITION = {&#10;    'name': 'my_skill',&#10;    'description': 'What this skill does',&#10;    'input_schema': {&#10;        'type': 'object',&#10;        'properties': {},&#10;    },&#10;}&#10;&#10;async def execute(tool_input, context):&#10;    return 'result'"></textarea>
        </div>

        <div v-if="editError" class="mb-3 p-2 rounded bg-red-950/30 border border-red-900/50">
          <div class="text-red-400 text-sm font-semibold mb-1">Error</div>
          <div class="text-red-300 text-sm whitespace-pre-wrap">{{ editError }}</div>
        </div>
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
    const showCode = ref({});
    const testResults = ref({});
    const testing = ref(null);

    // Editor state
    const editing = ref(false);
    const editMode = ref('create');
    const editName = ref('');
    const editCode = ref('');
    const editError = ref(null);
    const editSuccess = ref(null);
    const saving = ref(false);

    // Delete state
    const deleteTarget = ref(null);
    const deleting = ref(false);

    function highlight(code) {
      return highlightPython(code);
    }

    function truncate(text, max) {
      if (!text) return '';
      return text.length > max ? text.slice(0, max) + '...' : text;
    }

    function formatDate(iso) {
      if (!iso) return '—';
      try {
        const d = new Date(iso);
        return d.toLocaleString();
      } catch { return iso; }
    }

    function toggleCode(name) {
      showCode.value = { ...showCode.value, [name]: !showCode.value[name] };
    }

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

    async function testSkill(name) {
      testing.value = name;
      delete testResults.value[name];
      testResults.value = { ...testResults.value };
      try {
        const result = await api.post(`/api/skills/${encodeURIComponent(name)}/test`);
        testResults.value = { ...testResults.value, [name]: result };
      } catch (e) {
        testResults.value = { ...testResults.value, [name]: { result: e.message, is_error: true } };
      }
      testing.value = null;
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
      skills, loading, error, showCode, testResults, testing,
      editing, editMode, editName, editCode, editError, editSuccess, saving,
      deleteTarget, deleting,
      highlight, truncate, formatDate, toggleCode,
      fetchSkills, testSkill, showCreate, editSkill, cancelEdit, saveSkill,
      confirmDelete, doDelete,
    };
  },
};
