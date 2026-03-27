/**
 * Heimdall Management UI — Knowledge Page
 * Browse/search/ingest/delete knowledge documents with previews and search highlighting
 */
import { api } from '../api.js';

const { ref, onMounted } = Vue;

function escapeHtml(text) {
  return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function highlightTerms(text, query) {
  if (!text || !query) return escapeHtml(text);
  const escaped = escapeHtml(text);
  const terms = query.trim().split(/\s+/).filter(Boolean);
  if (!terms.length) return escaped;
  const pattern = terms.map(t => t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|');
  try {
    return escaped.replace(new RegExp(`(${pattern})`, 'gi'),
      '<mark class="knowledge-highlight">$1</mark>');
  } catch { return escaped; }
}

export default {
  template: `
    <div class="p-6 page-fade-in">
      <div class="flex items-center justify-between mb-4 flex-wrap gap-2">
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
        <input v-model="searchQuery" type="text" class="hm-input"
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
          <span class="text-gray-500 text-xs ml-2">for "{{ lastQuery }}"</span>
        </div>
        <div v-if="searchError" class="hm-card border-red-900">
          <p class="text-red-400 text-sm">Search error: {{ searchError }}</p>
        </div>
        <div v-else-if="searchResults.length === 0" class="hm-card empty-state">
          <span class="empty-state-icon">\u{1F50D}</span>
          <span class="empty-state-text">No results for "{{ lastQuery }}"</span>
          <span class="empty-state-hint">Try different search terms or ingest more documents</span>
        </div>
        <div v-else class="space-y-2">
          <div v-for="(r, i) in searchResults" :key="i" class="hm-card">
            <div class="flex items-center gap-2 mb-1">
              <span class="badge badge-info">{{ r.source || 'unknown' }}</span>
              <span v-if="r.score" class="text-gray-500 text-xs font-mono">score: {{ r.score.toFixed(3) }}</span>
              <span v-if="r.chunk_index !== undefined" class="text-gray-600 text-xs">chunk #{{ r.chunk_index }}</span>
            </div>
            <div class="text-sm text-gray-300 whitespace-pre-wrap break-words"
                 v-html="highlightTerms(truncate(r.content || r.text || '', 500), lastQuery)"></div>
          </div>
        </div>
      </div>

      <!-- Ingest form -->
      <div v-if="showIngest" class="hm-card mb-4">
        <h2 class="text-sm font-medium mb-3">Ingest Document</h2>
        <div class="mb-3">
          <label class="text-gray-400 text-xs block mb-1">Source Name</label>
          <input v-model="ingestSource" type="text" class="hm-input" placeholder="e.g. project-docs, api-reference" />
        </div>
        <div class="mb-3">
          <label class="text-gray-400 text-xs block mb-1">Content</label>
          <textarea v-model="ingestContent" class="hm-input" rows="8"
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
      <div v-else-if="error" class="hm-card border-red-900 error-state">
        <span class="error-icon">\u26A0</span>
        <p class="text-red-400">{{ error }}</p>
        <button @click="fetchSources" class="btn btn-ghost text-xs">Retry</button>
      </div>
      <div v-else-if="sources.length === 0 && !showIngest" class="hm-card empty-state">
        <span class="empty-state-icon">\u{1F4DA}</span>
        <span class="empty-state-text">No documents ingested</span>
        <span class="empty-state-hint">Click "Ingest Document" to add knowledge for Heimdall to reference</span>
      </div>
      <div v-else-if="sources.length > 0">
        <div class="text-sm font-medium text-gray-400 mb-2">
          Ingested Sources <span class="badge badge-info">{{ sources.length }}</span>
        </div>
        <div class="space-y-2">
          <div v-for="s in sources" :key="s.source || s.name || s" class="hm-card">
            <div class="flex items-center justify-between mb-1">
              <div class="flex items-center gap-2">
                <span class="font-mono text-sm font-semibold">{{ s.source || s.name || s }}</span>
                <span class="badge badge-info">{{ s.chunks || '—' }} chunks</span>
                <span v-if="s.uploader" class="badge badge-warning text-xs">{{ s.uploader }}</span>
              </div>
              <div class="flex gap-1">
                <button @click="doReingest(s.source || s.name || s)"
                        class="btn btn-ghost text-xs"
                        :disabled="reingesting === (s.source || s.name || s)">
                  {{ reingesting === (s.source || s.name || s) ? 'Re-ingesting...' : 'Re-ingest' }}
                </button>
                <button @click="confirmDelete(s.source || s.name || s)" class="btn btn-danger text-xs">Delete</button>
              </div>
            </div>
            <div v-if="s.ingested_at" class="text-gray-600 text-xs mb-1">
              Ingested: {{ formatDate(s.ingested_at) }}
            </div>
            <div v-if="s.preview" class="text-gray-400 text-sm mt-1 whitespace-pre-wrap break-words knowledge-preview">{{ s.preview }}</div>
            <div v-if="reingestResult && reingestResult.source === (s.source || s.name || s)"
                 class="mt-2 text-xs"
                 :class="reingestResult.error ? 'text-red-400' : 'text-green-400'">
              {{ reingestResult.message }}
            </div>
          </div>
        </div>
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
    const searchError = ref(null);

    // Ingest
    const showIngest = ref(false);
    const ingestSource = ref('');
    const ingestContent = ref('');
    const ingestError = ref(null);
    const ingestSuccess = ref(null);
    const ingesting = ref(false);

    // Re-ingest
    const reingesting = ref(null);
    const reingestResult = ref(null);
    let reingestTimer = null;

    // Delete
    const deleteTarget = ref(null);
    const deleting = ref(false);

    function truncate(text, max) {
      if (!text) return '';
      return text.length > max ? text.slice(0, max) + '...' : text;
    }

    function formatDate(iso) {
      if (!iso) return '';
      try {
        const d = new Date(iso);
        return d.toLocaleString();
      } catch { return iso; }
    }

    async function fetchSources() {
      loading.value = true;
      error.value = null;
      try {
        const data = await api.get('/api/knowledge');
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
      searchError.value = null;
      lastQuery.value = q;
      try {
        const results = await api.get(`/api/knowledge/search?q=${encodeURIComponent(q)}`);
        searchResults.value = Array.isArray(results) ? results : [];
      } catch (e) {
        searchResults.value = [];
        searchError.value = e.message || 'Search failed';
      }
      searching.value = false;
    }

    function clearSearch() {
      searchResults.value = null;
      searchQuery.value = '';
      searchError.value = null;
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

    async function doReingest(source) {
      reingesting.value = source;
      reingestResult.value = null;
      if (reingestTimer) { clearTimeout(reingestTimer); reingestTimer = null; }
      try {
        const result = await api.post(`/api/knowledge/${encodeURIComponent(source)}/reingest`);
        reingestResult.value = {
          source,
          error: false,
          message: `Re-ingested ${result.chunks || 0} chunks`,
        };
        await fetchSources();
        reingestTimer = setTimeout(() => { reingestResult.value = null; reingestTimer = null; }, 3000);
      } catch (e) {
        reingestResult.value = {
          source,
          error: true,
          message: e.message,
        };
      }
      reingesting.value = null;
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
      searchQuery, searchResults, searching, lastQuery, searchError,
      showIngest, ingestSource, ingestContent, ingestError, ingestSuccess, ingesting,
      reingesting, reingestResult,
      deleteTarget, deleting,
      truncate, formatDate, highlightTerms, fetchSources, doSearch, clearSearch,
      doIngest, doReingest, confirmDelete, doDelete,
    };
  },
};
