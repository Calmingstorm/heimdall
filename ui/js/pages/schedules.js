/**
 * Heimdall Management UI — Schedules Page
 * View/create/delete scheduled tasks (cron, one-time, webhook)
 */
import { api } from '../api.js';

const { ref, computed, onMounted } = Vue;

export default {
  template: `
    <div class="p-6 page-fade-in">
      <div class="flex items-center justify-between mb-4 flex-wrap gap-2">
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
      <div v-if="showCreate" class="hm-card mb-4">
        <h2 class="text-sm font-medium mb-3">Create Schedule</h2>

        <div class="mb-3">
          <label class="text-gray-400 text-xs block mb-1">Description</label>
          <input v-model="form.description" type="text" class="hm-input"
                 placeholder="e.g. Daily disk check" />
        </div>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-3 mb-3">
          <div>
            <label class="text-gray-400 text-xs block mb-1">Action Type</label>
            <select v-model="form.action" class="hm-input">
              <option value="reminder">Reminder</option>
              <option value="check">Check (tool call)</option>
              <option value="workflow">Workflow (multi-step)</option>
              <option value="digest">Digest</option>
            </select>
          </div>
          <div>
            <label class="text-gray-400 text-xs block mb-1">Channel ID</label>
            <input v-model="form.channel_id" type="text" class="hm-input"
                   placeholder="Discord channel ID" />
          </div>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-3 mb-3">
          <div>
            <label class="text-gray-400 text-xs block mb-1">Cron Expression</label>
            <div class="flex gap-2">
              <input v-model="form.cron" type="text" class="hm-input"
                     placeholder="e.g. 0 */6 * * *" @input="onCronInput" />
              <button @click="validateCron" class="btn btn-ghost text-xs whitespace-nowrap"
                      :disabled="!form.cron.trim() || validatingCron">
                {{ validatingCron ? '...' : 'Validate' }}
              </button>
            </div>
            <!-- Cron helper -->
            <div v-if="cronResult" class="mt-2 text-xs">
              <div v-if="cronResult.valid" class="text-green-400">
                Valid. Next runs:
                <div v-for="(run, i) in cronResult.next_runs" :key="i" class="text-gray-400 ml-2">
                  {{ formatIso(run) }} ({{ formatFuture(run) }})
                </div>
              </div>
              <div v-else class="text-red-400">{{ cronResult.error }}</div>
            </div>
            <!-- Quick cron presets -->
            <div class="flex flex-wrap gap-1 mt-2">
              <button v-for="p in cronPresets" :key="p.expr"
                      @click="form.cron = p.expr; onCronInput()"
                      class="cron-preset-btn">
                {{ p.label }}
              </button>
            </div>
          </div>
          <div>
            <label class="text-gray-400 text-xs block mb-1">One-Time (ISO datetime)</label>
            <input v-model="form.run_at" type="text" class="hm-input"
                   placeholder="e.g. 2026-04-01T09:00:00" />
          </div>
        </div>

        <div v-if="form.action === 'reminder'" class="mb-3">
          <label class="text-gray-400 text-xs block mb-1">Message</label>
          <input v-model="form.message" type="text" class="hm-input"
                 placeholder="Reminder message..." />
        </div>

        <div v-if="form.action === 'check'" class="grid grid-cols-1 md:grid-cols-2 gap-3 mb-3">
          <div>
            <label class="text-gray-400 text-xs block mb-1">Tool Name</label>
            <input v-model="form.tool_name" type="text" class="hm-input"
                   placeholder="e.g. run_command" />
          </div>
          <div>
            <label class="text-gray-400 text-xs block mb-1">Tool Input (JSON)</label>
            <input v-model="form.tool_input_str" type="text" class="hm-input"
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
      <div v-else-if="error" class="hm-card border-red-900 error-state" role="alert">
        <span class="error-icon" aria-hidden="true">\u26A0</span>
        <p class="text-red-400">{{ error }}</p>
        <button @click="fetchSchedules" class="btn btn-ghost text-xs">Retry</button>
      </div>
      <div v-else-if="schedules.length === 0 && !showCreate" class="hm-card empty-state">
        <span class="empty-state-icon">\u{23F0}</span>
        <span class="empty-state-text">No scheduled tasks</span>
        <span class="empty-state-hint">Click "New Schedule" to set up automated checks or reminders</span>
      </div>
      <div v-else-if="schedules.length > 0">
        <!-- Summary cards -->
        <div class="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
          <div class="hm-card text-center">
            <div class="text-2xl font-bold">{{ schedules.length }}</div>
            <div class="text-gray-400 text-xs">Total</div>
          </div>
          <div class="hm-card text-center">
            <div class="text-2xl font-bold">{{ cronCount }}</div>
            <div class="text-gray-400 text-xs">Recurring</div>
          </div>
          <div class="hm-card text-center">
            <div class="text-2xl font-bold">{{ oneTimeCount }}</div>
            <div class="text-gray-400 text-xs">One-Time</div>
          </div>
          <div class="hm-card text-center">
            <div class="text-2xl font-bold">{{ webhookCount }}</div>
            <div class="text-gray-400 text-xs">Webhook</div>
          </div>
        </div>

        <div class="table-responsive">
        <table class="hm-table">
          <thead>
            <tr>
              <th>Description</th>
              <th>Type</th>
              <th class="mobile-hide">Action</th>
              <th class="mobile-hide">Schedule</th>
              <th class="mobile-hide">Next Run</th>
              <th class="mobile-hide">Last Run</th>
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
              <td class="font-mono text-xs text-gray-400 mobile-hide">{{ s.action }}</td>
              <td class="text-sm text-gray-400 font-mono mobile-hide">
                <span v-if="s.cron">{{ s.cron }}</span>
                <span v-else-if="s.run_at">{{ formatIso(s.run_at) }}</span>
                <span v-else-if="s.trigger">{{ s.trigger.type || 'webhook' }}</span>
                <span v-else>-</span>
              </td>
              <td class="text-sm mobile-hide">
                <span v-if="s.next_run" class="text-indigo-300" :title="formatIso(s.next_run)">
                  {{ formatFuture(s.next_run) }}
                </span>
                <span v-else class="text-gray-600">-</span>
              </td>
              <td class="text-sm text-gray-400 mobile-hide">{{ s.last_run ? formatAge(s.last_run) : 'never' }}</td>
              <td class="whitespace-nowrap">
                <div class="flex gap-1">
                  <button @click="doRunNow(s.id)" class="btn btn-ghost text-xs"
                          :disabled="runningId === s.id"
                          title="Trigger this schedule immediately">
                    {{ runningId === s.id ? 'Running...' : 'Run Now' }}
                  </button>
                  <button @click="confirmDelete(s.id)" class="btn btn-danger text-xs">Delete</button>
                </div>
              </td>
            </tr>
          </tbody>
        </table>
        </div>
      </div>

      <!-- Toast -->
      <div v-if="toast" class="toast" :class="toast.type === 'success' ? 'toast-success' : 'toast-error'">
        {{ toast.message }}
      </div>

      <!-- Delete confirmation -->
      <div v-if="deleteTarget" class="modal-overlay" @click.self="deleteTarget = null" role="dialog" aria-modal="true" aria-labelledby="sched-delete-title">
        <div class="modal-content">
          <h3 id="sched-delete-title" class="text-lg font-semibold mb-2">Delete Schedule</h3>
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
    const toast = ref(null);

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

    // Cron validation
    const cronResult = ref(null);
    const validatingCron = ref(false);
    const cronPresets = [
      { label: 'Every hour', expr: '0 * * * *' },
      { label: 'Every 6h', expr: '0 */6 * * *' },
      { label: 'Daily 9am', expr: '0 9 * * *' },
      { label: 'Weekly Mon', expr: '0 9 * * 1' },
      { label: 'Every 30m', expr: '*/30 * * * *' },
    ];

    // Run now
    const runningId = ref(null);

    // Delete
    const deleteTarget = ref(null);
    const deleting = ref(false);

    const cronCount = computed(() => schedules.value.filter(s => s.cron && !s.one_time).length);
    const oneTimeCount = computed(() => schedules.value.filter(s => s.one_time).length);
    const webhookCount = computed(() => schedules.value.filter(s => s.trigger).length);

    function showToast(message, type = 'success') {
      toast.value = { message, type };
      setTimeout(() => { toast.value = null; }, 3000);
    }

    function formatIso(iso) {
      if (!iso) return '';
      try {
        const d = new Date(iso);
        return d.toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
      } catch { return iso; }
    }

    function formatAge(ts) {
      if (!ts) return '-';
      const now = Date.now();
      const t = new Date(ts).getTime();
      const diff = (now - t) / 1000;
      if (diff < 60) return 'just now';
      if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
      if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
      return `${Math.floor(diff / 86400)}d ago`;
    }

    function formatFuture(ts) {
      if (!ts) return '-';
      const now = Date.now();
      const t = new Date(ts).getTime();
      const diff = (t - now) / 1000;
      if (diff < 0) return 'overdue';
      if (diff < 60) return 'in < 1 min';
      if (diff < 3600) return `in ${Math.floor(diff / 60)} min`;
      if (diff < 86400) {
        const h = Math.floor(diff / 3600);
        const m = Math.floor((diff % 3600) / 60);
        return m > 0 ? `in ${h}h ${m}m` : `in ${h}h`;
      }
      const d = Math.floor(diff / 86400);
      return `in ${d} day${d !== 1 ? 's' : ''}`;
    }

    function onCronInput() {
      cronResult.value = null;
    }

    async function validateCron() {
      const expr = form.value.cron.trim();
      if (!expr) return;
      validatingCron.value = true;
      try {
        cronResult.value = await api.post('/api/schedules/validate-cron', { expression: expr });
      } catch (e) {
        cronResult.value = { valid: false, error: e.message };
      }
      validatingCron.value = false;
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
        cronResult.value = null;
        await fetchSchedules();
        setTimeout(() => { showCreate.value = false; createSuccess.value = null; }, 800);
      } catch (e) {
        createError.value = e.message;
      }
      creating.value = false;
    }

    async function doRunNow(scheduleId) {
      runningId.value = scheduleId;
      try {
        await api.post(`/api/schedules/${encodeURIComponent(scheduleId)}/run`);
        showToast('Schedule triggered');
        await fetchSchedules();
      } catch (e) {
        showToast(e.message || 'Failed to trigger', 'error');
      }
      runningId.value = null;
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
      schedules, loading, error, toast,
      showCreate, form, creating, createError, createSuccess,
      cronResult, validatingCron, cronPresets,
      runningId,
      deleteTarget, deleting,
      cronCount, oneTimeCount, webhookCount,
      showToast, formatIso, formatAge, formatFuture,
      onCronInput, validateCron,
      fetchSchedules, doCreate, doRunNow, confirmDelete, doDelete,
    };
  },
};
