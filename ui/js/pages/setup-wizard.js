/**
 * Heimdall Management UI — First-Boot Setup Wizard
 *
 * Multi-step form shown when Heimdall detects it hasn't been configured yet.
 * No auth required (runs before auth is configured).
 */
const { ref, computed, reactive, onMounted } = Vue;

export default {
  template: `
    <div class="min-h-screen flex items-center justify-center p-4 page-fade-in" role="main">
      <div class="w-full max-w-lg">
        <!-- Header -->
        <div class="text-center mb-6">
          <h1 class="text-2xl font-bold mb-1">Heimdall Setup</h1>
          <p class="text-gray-400 text-sm">Configure your bot in a few steps</p>
        </div>

        <!-- Progress bar -->
        <div class="flex items-center gap-1 mb-6">
          <template v-for="(s, i) in steps" :key="i">
            <div class="flex-1 h-1 rounded-full transition-colors"
                 :class="i <= currentStep ? 'bg-blue-500' : 'bg-gray-700'" />
          </template>
        </div>

        <!-- Step indicator -->
        <p class="text-xs text-gray-500 mb-4">Step {{ currentStep + 1 }} of {{ steps.length }} &mdash; {{ steps[currentStep] }}</p>

        <!-- Error banner -->
        <div v-if="error" class="mb-4 p-3 rounded bg-red-900/40 border border-red-700 text-red-300 text-sm" role="alert">
          {{ error }}
        </div>

        <!-- Step 1: Discord Token -->
        <div v-if="currentStep === 0" class="hm-card">
          <h2 class="text-lg font-semibold mb-3">Discord Bot Token</h2>
          <p class="text-gray-400 text-sm mb-4">
            Create a bot at
            <a href="https://discord.com/developers/applications" target="_blank" rel="noopener"
               class="text-blue-400 hover:underline">discord.com/developers</a>.
            Enable MESSAGE CONTENT and SERVER MEMBERS intents.
          </p>
          <label for="discord-token" class="block text-sm font-medium mb-1">Bot Token</label>
          <input id="discord-token" v-model="form.discord_token" type="password"
                 class="hm-input mb-1 w-full" placeholder="Paste your bot token"
                 @keyup.enter="nextStep" />
          <p v-if="tokenHint" class="text-xs mb-3" :class="tokenHint.ok ? 'text-green-400' : 'text-amber-400'">
            {{ tokenHint.text }}
          </p>
        </div>

        <!-- Step 2: Remote Hosts -->
        <div v-if="currentStep === 1" class="hm-card">
          <h2 class="text-lg font-semibold mb-3">Remote Hosts</h2>
          <p class="text-gray-400 text-sm mb-4">
            Add servers Heimdall can manage via SSH. You can skip this and add hosts later in config.yml.
          </p>
          <div v-for="(host, i) in form.hostList" :key="i" class="mb-3 p-3 rounded bg-gray-800/50 border border-gray-700">
            <div class="flex items-center gap-2 mb-2">
              <input v-model="host.name" placeholder="Name (e.g. myserver)" class="hm-input flex-1 text-sm" />
              <button @click="removeHost(i)" class="text-red-400 hover:text-red-300 text-xs px-2">Remove</button>
            </div>
            <div class="flex gap-2">
              <input v-model="host.address" placeholder="IP or hostname" class="hm-input flex-1 text-sm" />
              <input v-model="host.ssh_user" placeholder="SSH user" class="hm-input w-28 text-sm" />
            </div>
          </div>
          <button @click="addHost" class="btn btn-ghost text-sm">+ Add Host</button>
        </div>

        <!-- Step 3: Features -->
        <div v-if="currentStep === 2" class="hm-card">
          <h2 class="text-lg font-semibold mb-3">Optional Features</h2>
          <p class="text-gray-400 text-sm mb-4">Enable extra capabilities. All can be changed later.</p>
          <label class="flex items-center gap-3 mb-3 cursor-pointer">
            <input type="checkbox" v-model="form.features.browser" class="rounded" />
            <div>
              <span class="text-sm font-medium">Browser Automation</span>
              <p class="text-xs text-gray-500">Screenshot pages, interact with web apps</p>
            </div>
          </label>
          <label class="flex items-center gap-3 mb-3 cursor-pointer">
            <input type="checkbox" v-model="form.features.voice" class="rounded" />
            <div>
              <span class="text-sm font-medium">Voice Channel Support</span>
              <p class="text-xs text-gray-500">Join voice channels, text-to-speech</p>
            </div>
          </label>
          <label class="flex items-center gap-3 cursor-pointer">
            <input type="checkbox" v-model="form.features.comfyui" class="rounded" />
            <div>
              <span class="text-sm font-medium">ComfyUI Image Generation</span>
              <p class="text-xs text-gray-500">Generate images via ComfyUI API</p>
            </div>
          </label>
        </div>

        <!-- Step 4: Web UI Token -->
        <div v-if="currentStep === 3" class="hm-card">
          <h2 class="text-lg font-semibold mb-3">Web UI Security</h2>
          <p class="text-gray-400 text-sm mb-4">
            Protect this management interface with an API token.
          </p>
          <div class="flex items-center gap-2 mb-3">
            <label class="flex items-center gap-2 cursor-pointer">
              <input type="radio" v-model="form.tokenMode" value="generate" />
              <span class="text-sm">Generate random token</span>
            </label>
            <label class="flex items-center gap-2 cursor-pointer ml-4">
              <input type="radio" v-model="form.tokenMode" value="custom" />
              <span class="text-sm">Custom token</span>
            </label>
            <label class="flex items-center gap-2 cursor-pointer ml-4">
              <input type="radio" v-model="form.tokenMode" value="none" />
              <span class="text-sm">No auth</span>
            </label>
          </div>
          <div v-if="form.tokenMode === 'custom'">
            <input v-model="form.customToken" type="text" class="hm-input w-full text-sm"
                   placeholder="Enter your API token" />
          </div>
          <div v-if="form.tokenMode === 'generate'" class="p-3 rounded bg-gray-800/50 border border-gray-700">
            <p class="text-xs text-gray-400 mb-1">A random token will be generated. Save it after setup!</p>
          </div>
          <div v-if="form.tokenMode === 'none'" class="p-3 rounded bg-amber-900/30 border border-amber-700">
            <p class="text-xs text-amber-300">Warning: The web UI will be accessible without authentication.</p>
          </div>
        </div>

        <!-- Step 5: Review & Apply -->
        <div v-if="currentStep === 4" class="hm-card">
          <h2 class="text-lg font-semibold mb-3">Review & Apply</h2>
          <div class="space-y-2 text-sm">
            <div class="flex justify-between">
              <span class="text-gray-400">Discord Token</span>
              <span class="text-green-400">Configured</span>
            </div>
            <div class="flex justify-between">
              <span class="text-gray-400">Remote Hosts</span>
              <span>{{ validHosts.length || 'None' }}</span>
            </div>
            <div class="flex justify-between">
              <span class="text-gray-400">Features</span>
              <span>{{ enabledFeatures || 'None' }}</span>
            </div>
            <div class="flex justify-between">
              <span class="text-gray-400">Web UI Auth</span>
              <span>{{ form.tokenMode === 'none' ? 'Disabled' : 'Enabled' }}</span>
            </div>
          </div>
        </div>

        <!-- Success state -->
        <div v-if="setupDone" class="hm-card text-center">
          <div class="text-4xl mb-3">&#x2705;</div>
          <h2 class="text-lg font-semibold mb-2">Setup Complete!</h2>
          <p class="text-gray-400 text-sm mb-3">
            Configuration saved. Heimdall is restarting with the new settings.
          </p>
          <div v-if="generatedToken" class="p-3 rounded bg-gray-800 border border-gray-600 mb-3 text-left">
            <p class="text-xs text-gray-400 mb-1">Your Web UI API Token (save this!):</p>
            <code class="text-sm text-green-400 break-all select-all">{{ generatedToken }}</code>
          </div>
          <p class="text-xs text-gray-500">
            Reloading in {{ reloadCountdown }}s...
          </p>
        </div>

        <!-- Navigation buttons -->
        <div v-if="!setupDone" class="flex justify-between mt-4">
          <button v-if="currentStep > 0" @click="prevStep" class="btn btn-ghost">Back</button>
          <div v-else></div>
          <button v-if="currentStep < steps.length - 1" @click="nextStep"
                  class="btn btn-primary" :disabled="!canProceed">
            Next
          </button>
          <button v-else @click="submit" class="btn btn-primary" :disabled="submitting">
            <span v-if="submitting" class="inline-flex items-center gap-2">
              <svg class="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"/><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"/></svg>
              Applying...
            </span>
            <span v-else>Apply Configuration</span>
          </button>
        </div>
      </div>
    </div>`,

  setup() {
    const steps = ['Discord Token', 'Remote Hosts', 'Features', 'Web UI Security', 'Review'];
    const currentStep = ref(0);
    const error = ref(null);
    const submitting = ref(false);
    const setupDone = ref(false);
    const generatedToken = ref('');
    const reloadCountdown = ref(10);

    const form = reactive({
      discord_token: '',
      hostList: [],
      features: { browser: false, voice: false, comfyui: false },
      tokenMode: 'generate',
      customToken: '',
    });

    const tokenHint = computed(() => {
      const t = form.discord_token.trim();
      if (!t) return null;
      const parts = t.split('.');
      if (parts.length !== 3 || parts.some(p => !p)) {
        return { ok: false, text: 'Token should have 3 dot-separated parts' };
      }
      return { ok: true, text: 'Token format looks valid' };
    });

    const validHosts = computed(() =>
      form.hostList.filter(h => h.name.trim() && h.address.trim())
    );

    const enabledFeatures = computed(() => {
      const names = [];
      if (form.features.browser) names.push('Browser');
      if (form.features.voice) names.push('Voice');
      if (form.features.comfyui) names.push('ComfyUI');
      return names.join(', ');
    });

    const canProceed = computed(() => {
      if (currentStep.value === 0) {
        return tokenHint.value?.ok === true;
      }
      return true;
    });

    function addHost() {
      form.hostList.push({ name: '', address: '', ssh_user: 'root' });
    }

    function removeHost(i) {
      form.hostList.splice(i, 1);
    }

    function nextStep() {
      error.value = null;
      if (currentStep.value === 0 && !tokenHint.value?.ok) {
        error.value = 'Please enter a valid Discord bot token.';
        return;
      }
      if (currentStep.value < steps.length - 1) {
        currentStep.value++;
      }
    }

    function prevStep() {
      error.value = null;
      if (currentStep.value > 0) currentStep.value--;
    }

    function _generateToken(len = 32) {
      const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
      const arr = new Uint8Array(len);
      crypto.getRandomValues(arr);
      return Array.from(arr, b => chars[b % chars.length]).join('');
    }

    async function submit() {
      error.value = null;
      submitting.value = true;

      // Build hosts dict
      const hosts = {};
      for (const h of validHosts.value) {
        hosts[h.name.trim()] = {
          address: h.address.trim(),
          ssh_user: h.ssh_user.trim() || 'root',
        };
      }

      // Determine web API token
      let webToken = '';
      if (form.tokenMode === 'generate') {
        webToken = _generateToken();
        generatedToken.value = webToken;
      } else if (form.tokenMode === 'custom') {
        webToken = form.customToken.trim();
      }

      const payload = {
        discord_token: form.discord_token.trim(),
        hosts,
        features: form.features,
        web_api_token: webToken,
        timezone: 'UTC',
      };

      try {
        const resp = await fetch('/api/setup/complete', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
        const data = await resp.json().catch(() => null);
        if (!resp.ok) {
          error.value = data?.error || `Setup failed (HTTP ${resp.status})`;
          submitting.value = false;
          return;
        }
        setupDone.value = true;
        // Auto-reload after countdown
        const timer = setInterval(() => {
          reloadCountdown.value--;
          if (reloadCountdown.value <= 0) {
            clearInterval(timer);
            window.location.reload();
          }
        }, 1000);
      } catch (e) {
        error.value = e.message || 'Network error';
      } finally {
        submitting.value = false;
      }
    }

    return {
      steps, currentStep, error, submitting, setupDone, generatedToken,
      reloadCountdown, form, tokenHint, validHosts, enabledFeatures,
      canProceed, addHost, removeHost, nextStep, prevStep, submit,
    };
  },
};
