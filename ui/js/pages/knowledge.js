/**
 * Loki Management UI — Knowledge Page
 * Browse/search/ingest/delete knowledge documents
 */
import { api } from '../api.js';

const { ref, onMounted } = Vue;

export default {
  template: `
    <div class="p-6">
      <div class="flex items-center justify-between mb-4">
        <h1 class="text-xl font-semibold">Knowledge</h1>
        <div class="flex gap-2">
          <button @click="showIngest = !showIngest" class="btn btn-primary text-xs">
            {{ showIngest ? 'Cancel' : 'Ingest Document' }}
          </button>
          <button @click="fetchSources" class="btn btn-ghost text-xs" :disabled="loading">
            {{ loading ? 'Loading...' : 'Refresh' }}
          </button>
        </div>
      </div>

      <!-- Search bar -->
      <div class="mb-4 flex gap-2">
        <input v-model="searchQuery" type="text" class="loki-input"
               placeholder="Search knowledge base..."
               @keyup.enter="doSearch" />
        <button @click="doSearch" class="btn btn-primary text-xs whitespace-nowrap" :disabled="searching">
          {{ searching ? 'Searching...' : 'Search' }}
        </button>
        <button v-if="searchResults" @click="clearSearch" class="btn btn-ghost text-xs">Clear</button>
      </div>

      <!-- Search results -->
      <div v-if="searchResults" class="mb-6">
        <div class="text-sm font-medium text-gray-400 mb-2">
          Search Results <span class="badge badge-info">{{ searchResults.length }}</span>
        </div>
        <div v-if="searchResults.length === 0" class="loki-card">
          <p class="text-gray-400 text-sm">No results for "{{ lastQuery }}"</p>
        </div>
        <div v-else class="space-y-2">
          <div v-for="(r, i) in searchResults" :key="i" class="loki-card">
            <div class="flex items-center gap-2 mb-1">
              <span class="badge badge-info">{{ r.source || 'unknown' }}</span>
              <span v-if="r.score" class="text-gray-500 text-xs font-mono">score: {{ r.score.toFixed(3) }}</span>
            </div>
            <div class="text-sm text-gray-300 whitespace-pre-wrap break-words">{{ truncate(r.content || r.text || '', 500) }}</div>
          </div>
        </div>
      </div>

      <!-- Ingest form -->
      <div v-if="showIngest" class="loki-card mb-4">
        <h2 class="text-sm font-medium mb-3">Ingest Document</h2>
        <div class="mb-3">
          <label class="text-gray-400 text-xs block mb-1">Source Name</label>
          <input v-model="ingestSource" type="text" class="loki-input" placeholder="e.g. project-docs, api-reference" />
        </div>
        <div class="mb-3">
          <label class="text-gray-400 text-xs block mb-1">Content</label>
          <textarea v-model="ingestContent" class="loki-input" rows="8"
                    placeholder="Paste document content here..."></textarea>
        </div>
        <div v-if="ingestError" class="mb-3 text-red-400 text-sm">{{ ingestError }}</div>
        <div v-if="ingestSuccess" class="mb-3 text-green-400 text-sm">{{ ingestSuccess }}</div>
        <button @click="doIngest" class="btn btn-primary text-xs" :disabled="ingesting">
          {{ ingesting ? 'Ingesting...' : 'Ingest' }}
        </button>
      </div>

      <!-- Sources list -->
      <div v-if="loading && sources.length === 0" class="space-y-2">
        <div v-for="n in 3" :key="n" class="skeleton skeleton-row"></div>
      </div>
      <div v-else-if="error" class="loki-card border-red-900 error-state">
        <p class="text-red-400">{{ error }}</p>
        <button @click="fetchSources" class="btn btn-ghost text-xs">Retry</button>
      </div>
      <div v-else-if="sources.length === 0 && !showIngest" class="loki-card">
        <p class="text-gray-400">No documents ingested. Click "Ingest Document" to add one.</p>
      </div>
      <div v-else-if="sources.length > 0">
        <div class="text-sm font-medium text-gray-400 mb-2">
          Ingested Sources <span class="badge badge-info">{{ sources.length }}</span>
        </div>
        <table class="loki-table">
          <thead>
            <tr>
              <th>Source</th>
              <th>Chunks</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="s in sources" :key="s.source || s.name || s">
              <td class="font-mono text-sm">{{ s.source || s.name || s }}</td>
              <td>
                <span class="badge badge-info">{{ s.chunk_count || s.chunks || '—' }}</span>
              </td>
              <td>
                <button @click="confirmDelete(s.source || s.name || s)" class="btn btn-danger text-xs">Delete</button>
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <!-- Delete confirmation -->
      <div v-if="deleteTarget" class="modal-overlay" @click.self="deleteTarget = null">
        <div class="modal-content">
          <h3 class="text-lg font-semibold mb-2">Delete Source</h3>
          <p class="text-gray-400 text-sm mb-4">
            Delete all chunks for <span class="font-mono font-semibold text-gray-200">{{ deleteTarget }}</span>? This cannot be undone.
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
    const sources = ref([]);
    const loading = ref(true);
    const error = ref(null);

    // Search
    const searchQuery = ref('');
    const searchResults = ref(null);
    const searching = ref(false);
    const lastQuery = ref('');

    // Ingest
    const showIngest = ref(false);
    const ingestSource = ref('');
    const ingestContent = ref('');
    const ingestError = ref(null);
    const ingestSuccess = ref(null);
    const ingesting = ref(false);

    // Delete
    const deleteTarget = ref(null);
    const deleting = ref(false);

    function truncate(text, max) {
      if (!text) return '';
      return text.length > max ? text.slice(0, max) + '...' : text;
    }

    async function fetchSources() {
      loading.value = true;
      error.value = null;
      try {
        const data = await api.get('/api/knowledge');
        // API may return array of objects or strings
        sources.value = Array.isArray(data) ? data : [];
      } catch (e) {
        error.value = e.message;
      }
      loading.value = false;
    }

    async function doSearch() {
      const q = searchQuery.value.trim();
      if (!q) return;
      searching.value = true;
      lastQuery.value = q;
      try {
        const results = await api.get(`/api/knowledge/search?q=${encodeURIComponent(q)}`);
        searchResults.value = Array.isArray(results) ? results : [];
      } catch (e) {
        searchResults.value = [];
      }
      searching.value = false;
    }

    function clearSearch() {
      searchResults.value = null;
      searchQuery.value = '';
    }

    async function doIngest() {
      ingestError.value = null;
      ingestSuccess.value = null;
      const source = ingestSource.value.trim();
      const content = ingestContent.value.trim();
      if (!source) { ingestError.value = 'Source name is required'; return; }
      if (!content) { ingestError.value = 'Content is required'; return; }

      ingesting.value = true;
      try {
        const result = await api.post('/api/knowledge', { source, content });
        ingestSuccess.value = `Ingested ${result.chunks || 0} chunks from "${source}"`;
        ingestSource.value = '';
        ingestContent.value = '';
        await fetchSources();
        setTimeout(() => { showIngest.value = false; ingestSuccess.value = null; }, 1500);
      } catch (e) {
        ingestError.value = e.message;
      }
      ingesting.value = false;
    }

    function confirmDelete(source) {
      deleteTarget.value = source;
    }

    async function doDelete() {
      if (!deleteTarget.value) return;
      deleting.value = true;
      try {
        await api.del(`/api/knowledge/${encodeURIComponent(deleteTarget.value)}`);
        await fetchSources();
      } catch { /* ignore */ }
      deleting.value = false;
      deleteTarget.value = null;
    }

    onMounted(() => { fetchSources(); });

    return {
      sources, loading, error,
      searchQuery, searchResults, searching, lastQuery,
      showIngest, ingestSource, ingestContent, ingestError, ingestSuccess, ingesting,
      deleteTarget, deleting,
      truncate, fetchSources, doSearch, clearSearch, doIngest, confirmDelete, doDelete,
    };
  },
};
