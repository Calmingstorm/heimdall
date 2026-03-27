/**
 * Heimdall Management UI — Knowledge Page
 * Visual chunk browser with tree view, search highlighting, ingest/delete
 */
import { api } from '../api.js';

const { ref, computed, onMounted } = Vue;

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

      <!-- Stats bar -->
      <div v-if="!loading && sources.length > 0" class="kb-stats-bar">
        <div class="kb-stat">
          <span class="kb-stat-value">{{ sources.length }}</span>
          <span class="kb-stat-label">Sources</span>
        </div>
        <div class="kb-stat">
          <span class="kb-stat-value">{{ totalChunks }}</span>
          <span class="kb-stat-label">Chunks</span>
        </div>
        <div class="kb-stat">
          <span class="kb-stat-value">{{ uploaderCount }}</span>
          <span class="kb-stat-label">Uploaders</span>
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
          <div v-for="(r, i) in searchResults" :key="i" class="hm-card kb-search-result">
            <div class="flex items-center gap-2 mb-1">
              <span class="badge badge-info">{{ r.source || 'unknown' }}</span>
              <span v-if="r.score" class="kb-score-badge">{{ r.score.toFixed(3) }}</span>
              <span v-if="r.chunk_index !== undefined" class="text-gray-600 text-xs">chunk #{{ r.chunk_index }}</span>
            </div>
            <div class="text-sm text-gray-300 whitespace-pre-wrap break-words"
                 v-html="highlightTerms(truncate(r.content || r.text || '', 500), lastQuery)"></div>
          </div>
        </div>
      </div>

      <!-- Ingest form -->
      <div v-if="showIngest" class="hm-card mb-4 kb-ingest-form">
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

      <!-- Sources tree view -->
      <div v-if="loading && sources.length === 0" class="space-y-2" role="status" aria-label="Loading sources">
        <div v-for="n in 3" :key="n" class="skeleton skeleton-row"></div>
      </div>
      <div v-else-if="error" class="hm-card border-red-900 error-state" role="alert">
        <span class="error-icon" aria-hidden="true">\u26A0</span>
        <p class="text-red-400">{{ error }}</p>
        <button @click="fetchSources" class="btn btn-ghost text-xs">Retry</button>
      </div>
      <div v-else-if="sources.length === 0 && !showIngest" class="hm-card empty-state">
        <span class="empty-state-icon">\u{1F4DA}</span>
        <span class="empty-state-text">No documents ingested</span>
        <span class="empty-state-hint">Click "Ingest Document" to add knowledge for Heimdall to reference</span>
      </div>
      <div v-else-if="sources.length > 0" class="kb-tree">
        <div class="text-sm font-medium text-gray-400 mb-2">
          Sources <span class="badge badge-info">{{ sources.length }}</span>
        </div>
        <div class="kb-tree-list">
          <div v-for="s in sources" :key="s.source || s.name || s" class="kb-tree-node">
            <!-- Source header (tree branch) -->
            <div class="kb-tree-header" @click="toggleSource(s.source || s.name || s)"
                 role="button" tabindex="0" @keydown.enter="toggleSource(s.source || s.name || s)" @keydown.space.prevent="toggleSource(s.source || s.name || s)"
                 :aria-expanded="!!expanded[s.source || s.name || s]">
              <span class="kb-tree-arrow" :class="{ 'kb-tree-arrow-open': expanded[s.source || s.name || s] }" aria-hidden="true">
                \u25B6
              </span>
              <span class="kb-tree-icon">\u{1F4C4}</span>
              <span class="kb-tree-name">{{ s.source || s.name || s }}</span>
              <span class="badge badge-info text-xs">{{ s.chunks || 0 }} chunks</span>
              <span v-if="s.uploader" class="badge badge-warning text-xs">{{ s.uploader }}</span>
              <div class="kb-tree-actions">
                <button @click.stop="doReingest(s.source || s.name || s)"
                        class="btn btn-ghost text-xs"
                        :disabled="reingesting === (s.source || s.name || s)">
                  {{ reingesting === (s.source || s.name || s) ? 'Re-ingesting...' : 'Re-ingest' }}
                </button>
                <button @click.stop="confirmDelete(s.source || s.name || s)" class="btn btn-danger text-xs">Delete</button>
              </div>
            </div>

            <!-- Source metadata -->
            <div v-if="s.ingested_at && !expanded[s.source || s.name || s]" class="kb-tree-meta">
              Ingested: {{ formatDate(s.ingested_at) }}
            </div>
            <div v-if="s.preview && !expanded[s.source || s.name || s]" class="kb-tree-preview">{{ s.preview }}</div>

            <!-- Re-ingest result -->
            <div v-if="reingestResult && reingestResult.source === (s.source || s.name || s)"
                 class="kb-tree-meta"
                 :class="reingestResult.error ? 'text-red-400' : 'text-green-400'">
              {{ reingestResult.message }}
            </div>

            <!-- Chunk browser (expanded) -->
            <div v-if="expanded[s.source || s.name || s]" class="kb-chunk-browser">
              <div v-if="loadingChunks === (s.source || s.name || s)" class="kb-chunk-loading">
                <div class="spinner" style="width:14px;height:14px;border-width:2px;"></div> Loading chunks...
              </div>
              <div v-else-if="sourceChunks[s.source || s.name || s]" class="kb-chunk-list">
                <div class="kb-chunk-header">
                  <span class="text-gray-400 text-xs">{{ sourceChunks[s.source || s.name || s].length }} chunks</span>
                  <span class="text-gray-600 text-xs">Ingested: {{ formatDate(s.ingested_at) }}</span>
                </div>
                <div v-for="chunk in sourceChunks[s.source || s.name || s]" :key="chunk.chunk_id"
                     class="kb-chunk-item" :class="{ 'kb-chunk-selected': selectedChunk === chunk.chunk_id }"
                     @click="selectedChunk = selectedChunk === chunk.chunk_id ? null : chunk.chunk_id">
                  <div class="kb-chunk-item-header">
                    <span class="kb-chunk-index">#{{ chunk.chunk_index }}</span>
                    <span class="kb-chunk-chars">{{ chunk.char_count }} chars</span>
                    <div class="kb-chunk-bar">
                      <div class="kb-chunk-bar-fill" :style="{ width: chunkBarWidth(chunk, s.source || s.name || s) + '%' }"></div>
                    </div>
                  </div>
                  <div v-if="selectedChunk === chunk.chunk_id" class="kb-chunk-content">{{ chunk.content }}</div>
                  <div v-else class="kb-chunk-preview">{{ truncate(chunk.content, 120) }}</div>
                </div>
              </div>
              <div v-else class="kb-chunk-empty text-gray-500 text-xs">No chunks found</div>
            </div>
          </div>
        </div>
      </div>

      <!-- Delete confirmation -->
      <div v-if="deleteTarget" class="modal-overlay" @click.self="deleteTarget = null" role="dialog" aria-modal="true" aria-labelledby="kb-delete-title">
        <div class="modal-content">
          <h3 id="kb-delete-title" class="text-lg font-semibold mb-2">Delete Source</h3>
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

    // Tree / chunk browser
    const expanded = ref({});
    const sourceChunks = ref({});
    const loadingChunks = ref(null);
    const selectedChunk = ref(null);

    // Computed stats
    const totalChunks = computed(() => sources.value.reduce((sum, s) => sum + (s.chunks || 0), 0));
    const uploaderCount = computed(() => {
      const set = new Set(sources.value.map(s => s.uploader).filter(Boolean));
      return set.size;
    });

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

    function chunkBarWidth(chunk, source) {
      const chunks = sourceChunks.value[source];
      if (!chunks || chunks.length === 0) return 0;
      const maxChars = Math.max(...chunks.map(c => c.char_count || 0));
      if (maxChars === 0) return 0;
      return Math.round((chunk.char_count / maxChars) * 100);
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

    async function toggleSource(source) {
      if (expanded.value[source]) {
        expanded.value[source] = false;
        selectedChunk.value = null;
        return;
      }
      expanded.value[source] = true;
      if (sourceChunks.value[source]) return;

      loadingChunks.value = source;
      try {
        const chunks = await api.get(`/api/knowledge/${encodeURIComponent(source)}/chunks`);
        sourceChunks.value[source] = Array.isArray(chunks) ? chunks : [];
      } catch {
        sourceChunks.value[source] = [];
      }
      loadingChunks.value = null;
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
        sourceChunks.value = {};
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
        delete sourceChunks.value[source];
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
        delete sourceChunks.value[deleteTarget.value];
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
      expanded, sourceChunks, loadingChunks, selectedChunk,
      totalChunks, uploaderCount,
      truncate, formatDate, highlightTerms, chunkBarWidth,
      fetchSources, toggleSource, doSearch, clearSearch,
      doIngest, doReingest, confirmDelete, doDelete,
    };
  },
};
