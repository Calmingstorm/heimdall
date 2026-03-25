/**
 * Loki Management UI — Schedules Page
 * View/create/delete scheduled tasks (cron, one-time, webhook)
 */
import { api } from '../api.js';

const { ref, computed, onMounted } = Vue;

export default {
  template: `
    <div class="p-6">
      <div class="flex items-center justify-between mb-4">
        <h1 class="text-xl font-semibold">Schedules</h1>
        <div class="flex gap-2">
          <button @click="showCreate = !showCreate" class="btn btn-primary text-xs">
            {{ showCreate ? 'Cancel' : 'New Schedule' }}
          </button>
          <button @click="fetchSchedules" class="btn btn-ghost text-xs" :disabled="loading">
            {{ loading ? 'Loading...' : 'Refresh' }}
          </button>
        </div>
      </div>

      <!-- Create form -->
      <div v-if="showCreate" class="loki-card mb-4">
        <h2 class="text-sm font-medium mb-3">Create Schedule</h2>

        <div class="mb-3">
          <label class="text-gray-400 text-xs block mb-1">Description</label>
          <input v-model="form.description" type="text" class="loki-input"
                 placeholder="e.g. Daily disk check" />
        </div>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-3 mb-3">
          <div>
            <label class="text-gray-400 text-xs block mb-1">Action Type</label>
            <select v-model="form.action" class="loki-input">
              <option value="reminder">Reminder</option>
              <option value="check">Check (tool call)</option>
              <option value="workflow">Workflow (multi-step)</option>
              <option value="digest">Digest</option>
            </select>
          </div>
          <div>
            <label class="text-gray-400 text-xs block mb-1">Channel ID</label>
            <input v-model="form.channel_id" type="text" class="loki-input"
                   placeholder="Discord channel ID" />
          </div>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-3 mb-3">
          <div>
            <label class="text-gray-400 text-xs block mb-1">Cron Expression</label>
            <input v-model="form.cron" type="text" class="loki-input"
                   placeholder="e.g. 0 */6 * * * (every 6h)" />
          </div>
          <div>
            <label class="text-gray-400 text-xs block mb-1">One-Time (ISO datetime)</label>
            <input v-model="form.run_at" type="text" class="loki-input"
                   placeholder="e.g. 2026-04-01T09:00:00" />
          </div>
        </div>

        <div v-if="form.action === 'reminder'" class="mb-3">
          <label class="text-gray-400 text-xs block mb-1">Message</label>
          <input v-model="form.message" type="text" class="loki-input"
                 placeholder="Reminder message..." />
        </div>

        <div v-if="form.action === 'check'" class="grid grid-cols-1 md:grid-cols-2 gap-3 mb-3">
          <div>
            <label class="text-gray-400 text-xs block mb-1">Tool Name</label>
            <input v-model="form.tool_name" type="text" class="loki-input"
                   placeholder="e.g. check_disk" />
          </div>
          <div>
            <label class="text-gray-400 text-xs block mb-1">Tool Input (JSON)</label>
            <input v-model="form.tool_input_str" type="text" class="loki-input"
                   placeholder='e.g. {"host":"server1"}' />
          </div>
        </div>

        <div v-if="createError" class="mb-3 text-red-400 text-sm">{{ createError }}</div>
        <div v-if="createSuccess" class="mb-3 text-green-400 text-sm">{{ createSuccess }}</div>

        <button @click="doCreate" class="btn btn-primary text-xs" :disabled="creating">
          {{ creating ? 'Creating...' : 'Create' }}
        </button>
      </div>

      <!-- Schedule list -->
      <div v-if="loading && schedules.length === 0" class="space-y-2">
        <div v-for="n in 4" :key="n" class="skeleton skeleton-row"></div>
      </div>
      <div v-else-if="error" class="loki-card border-red-900 error-state">
        <p class="text-red-400">{{ error }}</p>
        <button @click="fetchSchedules" class="btn btn-ghost text-xs">Retry</button>
      </div>
      <div v-else-if="schedules.length === 0 && !showCreate" class="loki-card">
        <p class="text-gray-400">No scheduled tasks. Click "New Schedule" to create one.</p>
      </div>
      <div v-else-if="schedules.length > 0">
        <!-- Summary cards -->
        <div class="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
          <div class="loki-card text-center">
            <div class="text-2xl font-bold">{{ schedules.length }}</div>
            <div class="text-gray-400 text-xs">Total</div>
          </div>
          <div class="loki-card text-center">
            <div class="text-2xl font-bold">{{ cronCount }}</div>
            <div class="text-gray-400 text-xs">Recurring</div>
          </div>
          <div class="loki-card text-center">
            <div class="text-2xl font-bold">{{ oneTimeCount }}</div>
            <div class="text-gray-400 text-xs">One-Time</div>
          </div>
          <div class="loki-card text-center">
            <div class="text-2xl font-bold">{{ webhookCount }}</div>
            <div class="text-gray-400 text-xs">Webhook</div>
          </div>
        </div>

        <table class="loki-table">
          <thead>
            <tr>
              <th>Description</th>
              <th>Type</th>
              <th>Action</th>
              <th>Schedule</th>
              <th>Last Run</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="s in schedules" :key="s.id">
              <td class="text-sm">{{ s.description }}</td>
              <td>
                <span v-if="s.trigger" class="badge badge-warning">webhook</span>
                <span v-else-if="s.one_time" class="badge badge-info">one-time</span>
                <span v-else class="badge badge-success">cron</span>
              </td>
              <td class="font-mono text-xs text-gray-400">{{ s.action }}</td>
              <td class="text-sm text-gray-400 font-mono">
                <span v-if="s.cron">{{ s.cron }}</span>
                <span v-else-if="s.run_at">{{ formatIso(s.run_at) }}</span>
                <span v-else-if="s.trigger">{{ s.trigger.type || 'webhook' }}</span>
                <span v-else>-</span>
              </td>
              <td class="text-sm text-gray-400">{{ s.last_run ? formatIso(s.last_run) : 'never' }}</td>
              <td>
                <button @click="confirmDelete(s.id)" class="btn btn-danger text-xs">Delete</button>
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <!-- Delete confirmation -->
      <div v-if="deleteTarget" class="modal-overlay" @click.self="deleteTarget = null">
        <div class="modal-content">
          <h3 class="text-lg font-semibold mb-2">Delete Schedule</h3>
          <p class="text-gray-400 text-sm mb-4">
            Delete this scheduled task? This cannot be undone.
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
    const schedules = ref([]);
    const loading = ref(true);
    const error = ref(null);

    // Create form
    const showCreate = ref(false);
    const form = ref({
      description: '',
      action: 'reminder',
      channel_id: '',
      cron: '',
      run_at: '',
      message: '',
      tool_name: '',
      tool_input_str: '',
    });
    const creating = ref(false);
    const createError = ref(null);
    const createSuccess = ref(null);

    // Delete
    const deleteTarget = ref(null);
    const deleting = ref(false);

    const cronCount = computed(() => schedules.value.filter(s => s.cron && !s.one_time).length);
    const oneTimeCount = computed(() => schedules.value.filter(s => s.one_time).length);
    const webhookCount = computed(() => schedules.value.filter(s => s.trigger).length);

    function formatIso(iso) {
      if (!iso) return '';
      try {
        const d = new Date(iso);
        return d.toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
      } catch { return iso; }
    }

    async function fetchSchedules() {
      loading.value = true;
      error.value = null;
      try {
        schedules.value = await api.get('/api/schedules');
      } catch (e) {
        error.value = e.message;
      }
      loading.value = false;
    }

    async function doCreate() {
      createError.value = null;
      createSuccess.value = null;
      const f = form.value;
      if (!f.description.trim()) { createError.value = 'Description is required'; return; }
      if (!f.channel_id.trim()) { createError.value = 'Channel ID is required'; return; }
      if (!f.cron.trim() && !f.run_at.trim()) { createError.value = 'Cron expression or run_at time is required'; return; }

      const payload = {
        description: f.description.trim(),
        action: f.action,
        channel_id: f.channel_id.trim(),
      };
      if (f.cron.trim()) payload.cron = f.cron.trim();
      if (f.run_at.trim()) payload.run_at = f.run_at.trim();
      if (f.action === 'reminder' && f.message.trim()) payload.message = f.message.trim();
      if (f.action === 'check') {
        if (f.tool_name.trim()) payload.tool_name = f.tool_name.trim();
        if (f.tool_input_str.trim()) {
          try {
            payload.tool_input = JSON.parse(f.tool_input_str.trim());
          } catch {
            createError.value = 'Tool input must be valid JSON';
            return;
          }
        }
      }

      creating.value = true;
      try {
        await api.post('/api/schedules', payload);
        createSuccess.value = 'Schedule created';
        // Reset form
        form.value = {
          description: '', action: 'reminder', channel_id: '',
          cron: '', run_at: '', message: '', tool_name: '', tool_input_str: '',
        };
        await fetchSchedules();
        setTimeout(() => { showCreate.value = false; createSuccess.value = null; }, 800);
      } catch (e) {
        createError.value = e.message;
      }
      creating.value = false;
    }

    function confirmDelete(id) {
      deleteTarget.value = id;
    }

    async function doDelete() {
      if (!deleteTarget.value) return;
      deleting.value = true;
      try {
        await api.del(`/api/schedules/${encodeURIComponent(deleteTarget.value)}`);
        await fetchSchedules();
      } catch { /* ignore */ }
      deleting.value = false;
      deleteTarget.value = null;
    }

    onMounted(() => { fetchSchedules(); });

    return {
      schedules, loading, error,
      showCreate, form, creating, createError, createSuccess,
      deleteTarget, deleting,
      cronCount, oneTimeCount, webhookCount,
      formatIso, fetchSchedules, doCreate, confirmDelete, doDelete,
    };
  },
};
