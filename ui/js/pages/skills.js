/**
 * Heimdall Management UI — Skills Page (Round 39 Redesign)
 * Card layout, improved code editor with line numbers + syntax highlighting,
 * skill status indicators, search/filter
 */
import { api } from '../api.js';

const { ref, computed, onMounted, nextTick, watch, onUnmounted } = Vue;

// Python syntax highlighting (no external dependency)
function highlightPython(code) {
  if (!code) return '';
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
  // Built-in functions
  const builtins = '\\b(print|len|range|str|int|float|list|dict|set|tuple|type|isinstance|hasattr|getattr|setattr|super|property|staticmethod|classmethod|enumerate|zip|map|filter|sorted|reversed|any|all|min|max|sum|abs|round|open|format)\\b';
  html = html.replace(new RegExp(builtins, 'g'), '<span class="sk-builtin">$1</span>');
  // Decorators
  html = html.replace(/(@\w+)/g, '<span class="sk-dec">$1</span>');
  // Numbers
  html = html.replace(/\b(\d+\.?\d*)\b/g, '<span class="sk-num">$1</span>');
  return html;
}

/** Generate line numbers HTML for code display */
function lineNumbers(code) {
  if (!code) return '1';
  const count = code.split('\n').length;
  return Array.from({ length: count }, (_, i) => i + 1).join('\n');
}

export default {
  template: `
    <div class="p-6 page-fade-in">
      <div class="flex items-center justify-between mb-4 flex-wrap gap-2">
        <h1 class="text-xl font-semibold">Skills</h1>
        <div class="flex gap-2 items-center">
          <button @click="showCreate" class="btn btn-primary text-xs">New Skill</button>
          <button @click="fetchSkills" class="btn btn-ghost text-xs" :disabled="loading">
            {{ loading ? 'Loading...' : 'Refresh' }}
          </button>
        </div>
      </div>

      <!-- Stats summary -->
      <div v-if="skills.length > 0 && !editing" class="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
        <div class="sk-stat-card">
          <div class="sk-stat-value">{{ skills.length }}</div>
          <div class="sk-stat-label">Total Skills</div>
        </div>
        <div class="sk-stat-card">
          <div class="sk-stat-value">{{ enabledCount }}</div>
          <div class="sk-stat-label">Active Skills</div>
        </div>
        <div class="sk-stat-card">
          <div class="sk-stat-value">{{ totalExecutions.toLocaleString() }}</div>
          <div class="sk-stat-label">Total Runs</div>
        </div>
        <div class="sk-stat-card">
          <div class="sk-stat-value">{{ totalLines.toLocaleString() }}</div>
          <div class="sk-stat-label">Lines of Code</div>
        </div>
      </div>

      <!-- Search/filter (when not editing) -->
      <div v-if="skills.length > 0 && !editing" class="mb-4">
        <input v-model="search" type="text" class="hm-input sk-search" placeholder="Search skills by name or description..." />
      </div>

      <!-- Loading skeleton -->
      <div v-if="loading && skills.length === 0" class="space-y-3">
        <div class="grid grid-cols-2 md:grid-cols-4 gap-3">
          <div v-for="n in 4" :key="n" class="hm-card text-center">
            <div class="skeleton skeleton-stat"></div>
            <div class="skeleton skeleton-text" style="width:60%;margin:0.25rem auto 0;"></div>
          </div>
        </div>
        <div v-for="n in 3" :key="n + 4" class="hm-card"><div class="skeleton skeleton-row"></div><div class="skeleton skeleton-text mt-2" style="width:70%"></div></div>
      </div>

      <!-- Error state -->
      <div v-else-if="error" class="hm-card border-red-900 error-state" role="alert">
        <span class="error-icon" aria-hidden="true">\u26A0</span>
        <p class="text-red-400">{{ error }}</p>
        <button @click="fetchSkills" class="btn btn-ghost text-xs">Retry</button>
      </div>

      <!-- Empty state -->
      <div v-else-if="skills.length === 0 && !editing" class="hm-card empty-state">
        <span class="empty-state-icon">\u{1F9E9}</span>
        <span class="empty-state-text">No skills loaded</span>
        <span class="empty-state-hint">Click "New Skill" to create a custom tool</span>
      </div>

      <!-- Skill cards -->
      <div v-else-if="!editing">
        <div class="sk-card-grid">
          <div v-for="s in displayedSkills" :key="s.name" class="sk-card" :class="{ 'sk-card-tested': testResults[s.name] }">
            <!-- Card header -->
            <div class="sk-card-header">
              <div class="sk-card-title-row">
                <span class="sk-card-icon">\u{1F9E9}</span>
                <span class="sk-card-name">{{ s.name }}</span>
                <span v-if="s.execution_count > 0" class="sk-card-runs">{{ s.execution_count.toLocaleString() }} runs</span>
              </div>
              <div class="sk-card-actions">
                <button @click.stop="testSkill(s.name)"
                        class="sk-action-btn sk-action-test"
                        :disabled="testing === s.name"
                        :title="testing === s.name ? 'Testing...' : 'Run test'">
                  {{ testing === s.name ? '\u23F3' : '\u25B6' }}
                </button>
                <button @click.stop="toggleCode(s.name)"
                        class="sk-action-btn sk-action-code"
                        :title="showCode[s.name] ? 'Hide code' : 'View code'">
                  {{ showCode[s.name] ? '\u{1F4D6}' : '\u{1F4C4}' }}
                </button>
                <button @click.stop="editSkill(s)" class="sk-action-btn sk-action-edit" title="Edit">\u270E</button>
                <button @click.stop="confirmDelete(s.name)" class="sk-action-btn sk-action-delete" title="Delete">\u2715</button>
              </div>
            </div>

            <!-- Card body -->
            <div class="sk-card-body">
              <div class="sk-card-desc">{{ s.description || 'No description' }}</div>
              <div class="sk-card-meta">
                <span class="sk-card-date">Loaded: {{ formatDate(s.loaded_at) }}</span>
                <span v-if="s.code" class="sk-card-lines">{{ countLines(s.code) }} lines</span>
              </div>
            </div>

            <!-- Test result -->
            <div v-if="testResults[s.name]" class="sk-test-result"
                 :class="testResults[s.name].is_error ? 'sk-test-fail' : 'sk-test-pass'">
              <div class="sk-test-label">
                {{ testResults[s.name].is_error ? '\u2718 Test Failed' : '\u2714 Test Passed' }}
              </div>
              <div class="sk-test-output">{{ truncate(testResults[s.name].result, 500) }}</div>
            </div>

            <!-- Code preview with line numbers -->
            <div v-if="showCode[s.name] && s.code" class="sk-code-container">
              <div class="sk-code-header">
                <span class="sk-code-filename">{{ s.name }}.py</span>
                <button @click.stop="copyCode(s.code)" class="sk-code-copy" title="Copy code">
                  {{ copied === s.name ? '\u2714' : '\u{1F4CB}' }}
                </button>
              </div>
              <div class="sk-code-wrap">
                <pre class="sk-line-numbers">{{ getLineNumbers(s.code) }}</pre>
                <pre class="sk-code-block"><code v-html="highlight(s.code)"></code></pre>
              </div>
            </div>
          </div>
        </div>

        <!-- Empty search -->
        <div v-if="displayedSkills.length === 0 && search" class="hm-card empty-state">
          <span class="empty-state-icon">\u{1F50D}</span>
          <span class="empty-state-text">No skills match "{{ search }}"</span>
          <span class="empty-state-hint">Try a different search term</span>
        </div>
      </div>

      <!-- Create/Edit form with enhanced editor -->
      <div v-if="editing" class="sk-editor-panel">
        <div class="sk-editor-header">
          <h2 class="sk-editor-title">
            {{ editMode === 'create' ? 'Create Skill' : 'Edit Skill: ' + editName }}
          </h2>
          <button @click="cancelEdit" class="btn btn-ghost text-xs">Cancel</button>
        </div>

        <div v-if="editMode === 'create'" class="mb-3">
          <label class="sk-field-label">Name</label>
          <input v-model="editName" type="text" class="hm-input" placeholder="my_skill"
                 style="max-width:300px" />
          <div class="sk-field-hint">Lowercase, alphanumeric + underscores, starts with letter</div>
        </div>

        <div class="mb-3">
          <label class="sk-field-label">Code</label>
          <div class="sk-editor-wrap">
            <div class="sk-editor-gutter">{{ editorLineNums }}</div>
            <textarea v-model="editCode" class="sk-editor-textarea" rows="24"
                      @keydown="handleEditorKey"
                      @scroll="syncScroll"
                      ref="editorRef"
                      placeholder="# Skill code here...&#10;&#10;SKILL_DEFINITION = {&#10;    'name': 'my_skill',&#10;    'description': 'What this skill does',&#10;    'input_schema': {&#10;        'type': 'object',&#10;        'properties': {},&#10;    },&#10;}&#10;&#10;async def execute(tool_input, context):&#10;    return 'result'"></textarea>
          </div>
          <div class="sk-editor-status">
            <span class="sk-editor-line-count">{{ editLineCount }} lines</span>
            <span class="sk-editor-char-count">{{ editCode.length.toLocaleString() }} chars</span>
          </div>
        </div>

        <!-- Validation preview -->
        <div v-if="editCode && editValidation" class="sk-validation-box"
             :class="editValidation.valid ? 'sk-validation-ok' : 'sk-validation-err'">
          <span>{{ editValidation.valid ? '\u2714 Valid Python structure' : '\u26A0 ' + editValidation.message }}</span>
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
      <div v-if="deleteTarget" class="modal-overlay" @click.self="deleteTarget = null" role="dialog" aria-modal="true" aria-labelledby="skill-delete-title">
        <div class="modal-content">
          <h3 id="skill-delete-title" class="text-lg font-semibold mb-2">Delete Skill</h3>
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
    const search = ref('');
    const copied = ref(null);

    // Editor state
    const editing = ref(false);
    const editMode = ref('create');
    const editName = ref('');
    const editCode = ref('');
    const editError = ref(null);
    const editSuccess = ref(null);
    const saving = ref(false);
    const editorRef = ref(null);

    // Delete state
    const deleteTarget = ref(null);
    const deleting = ref(false);

    // Computed
    const enabledCount = computed(() => skills.value.length);
    const totalExecutions = computed(() => skills.value.reduce((sum, s) => sum + (s.execution_count || 0), 0));
    const totalLines = computed(() => skills.value.reduce((sum, s) => sum + countLines(s.code), 0));

    const displayedSkills = computed(() => {
      if (!search.value) return skills.value;
      const q = search.value.toLowerCase();
      return skills.value.filter(s =>
        s.name.toLowerCase().includes(q) || (s.description || '').toLowerCase().includes(q)
      );
    });

    const editLineCount = computed(() => {
      if (!editCode.value) return 0;
      return editCode.value.split('\n').length;
    });

    const editorLineNums = computed(() => {
      const count = Math.max(editLineCount.value, 1);
      return Array.from({ length: count }, (_, i) => i + 1).join('\n');
    });

    /** Basic structural validation of skill code */
    const editValidation = computed(() => {
      const code = editCode.value.trim();
      if (!code) return null;
      if (!code.includes('SKILL_DEFINITION')) {
        return { valid: false, message: 'Missing SKILL_DEFINITION dict' };
      }
      if (!code.includes('async def execute')) {
        return { valid: false, message: 'Missing async def execute function' };
      }
      return { valid: true, message: '' };
    });

    function highlight(code) {
      return highlightPython(code);
    }

    function truncate(text, max) {
      if (!text) return '';
      return text.length > max ? text.slice(0, max) + '...' : text;
    }

    function formatDate(iso) {
      if (!iso) return '\u2014';
      try {
        const d = new Date(iso);
        return d.toLocaleString();
      } catch { return iso; }
    }

    function countLines(code) {
      if (!code) return 0;
      return code.split('\n').length;
    }

    function getLineNumbers(code) {
      return lineNumbers(code);
    }

    function toggleCode(name) {
      showCode.value = { ...showCode.value, [name]: !showCode.value[name] };
    }

    async function copyCode(code) {
      try {
        await navigator.clipboard.writeText(code);
        const skill = skills.value.find(s => s.code === code);
        if (skill) {
          copied.value = skill.name;
          setTimeout(() => { copied.value = null; }, 2000);
        }
      } catch { /* clipboard not available */ }
    }

    /** Handle Tab key in editor for indentation */
    function handleEditorKey(e) {
      if (e.key === 'Tab') {
        e.preventDefault();
        const ta = e.target;
        const start = ta.selectionStart;
        const end = ta.selectionEnd;
        editCode.value = editCode.value.substring(0, start) + '    ' + editCode.value.substring(end);
        nextTick(() => {
          ta.selectionStart = ta.selectionEnd = start + 4;
        });
      }
    }

    /** Sync scroll between gutter and textarea */
    function syncScroll(e) {
      const gutter = e.target.previousElementSibling;
      if (gutter) gutter.scrollTop = e.target.scrollTop;
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
      skills, loading, error, showCode, testResults, testing, search, copied,
      editing, editMode, editName, editCode, editError, editSuccess, saving,
      editorRef,
      deleteTarget, deleting,
      enabledCount, totalExecutions, totalLines, displayedSkills,
      editLineCount, editorLineNums, editValidation,
      highlight, truncate, formatDate, countLines, getLineNumbers,
      toggleCode, copyCode, handleEditorKey, syncScroll,
      fetchSkills, testSkill, showCreate, editSkill, cancelEdit, saveSkill,
      confirmDelete, doDelete,
    };
  },
};
