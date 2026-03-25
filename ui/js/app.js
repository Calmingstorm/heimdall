/**
 * Loki Management UI — Main App
 * Vue 3 + Vue Router (CDN globals) + Tailwind CSS
 */
import { api, ws } from './api.js';
import DashboardPage from './pages/dashboard.js';
import SessionsPage from './pages/sessions.js';
import ToolsPage from './pages/tools.js';
import SkillsPage from './pages/skills.js';
import KnowledgePage from './pages/knowledge.js';
import SchedulesPage from './pages/schedules.js';
import LoopsPage from './pages/loops.js';
import ProcessesPage from './pages/processes.js';
import ConfigPage from './pages/config.js';
import LogsPage from './pages/logs.js';
import AuditPage from './pages/audit.js';
import MemoryPage from './pages/memory.js';

const { createApp, ref, computed, onMounted, onUnmounted, watch, nextTick } = Vue;
const { createRouter, createWebHashHistory } = VueRouter;

// All page components imported above

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------
const routes = [
  { path: '/',           redirect: '/dashboard' },
  { path: '/dashboard',  component: DashboardPage,  meta: { label: 'Dashboard',  icon: '\u{1F4CA}' } },
  { path: '/sessions',   component: SessionsPage,   meta: { label: 'Sessions',   icon: '\u{1F4AC}' } },
  { path: '/tools',      component: ToolsPage,      meta: { label: 'Tools',      icon: '\u{1F527}' } },
  { path: '/skills',     component: SkillsPage,     meta: { label: 'Skills',     icon: '\u{1F9E9}' } },
  { path: '/knowledge',  component: KnowledgePage,  meta: { label: 'Knowledge',  icon: '\u{1F4DA}' } },
  { path: '/schedules',  component: SchedulesPage,  meta: { label: 'Schedules',  icon: '\u{23F0}' } },
  { path: '/loops',      component: LoopsPage,      meta: { label: 'Loops',      icon: '\u{1F504}' } },
  { path: '/processes',  component: ProcessesPage,  meta: { label: 'Processes',  icon: '\u{2699}\u{FE0F}' } },
  { path: '/audit',      component: AuditPage,      meta: { label: 'Audit',      icon: '\u{1F4DD}' } },
  { path: '/config',     component: ConfigPage,     meta: { label: 'Config',     icon: '\u{2699}\u{FE0F}' } },
  { path: '/logs',       component: LogsPage,       meta: { label: 'Logs',       icon: '\u{1F4C4}' } },
  { path: '/memory',     component: MemoryPage,     meta: { label: 'Memory',     icon: '\u{1F9E0}' } },
];

const router = createRouter({
  history: createWebHashHistory(),
  routes,
});

// ---------------------------------------------------------------------------
// Login component
// ---------------------------------------------------------------------------
const LoginScreen = {
  template: `
    <div class="min-h-screen flex items-center justify-center">
      <div class="loki-card w-full max-w-sm">
        <h1 class="text-xl font-semibold mb-1 text-center">Loki</h1>
        <p class="text-gray-400 text-sm text-center mb-4">Management Interface</p>
        <div v-if="error" class="mb-3 text-red-400 text-sm text-center">{{ error }}</div>
        <form @submit.prevent="login">
          <input
            v-model="token"
            type="password"
            placeholder="API Token"
            class="loki-input mb-3"
            autofocus
          />
          <button type="submit" class="btn btn-primary w-full justify-center" :disabled="busy">
            <span v-if="busy" class="spinner" style="width:14px;height:14px;border-width:2px;"></span>
            {{ busy ? 'Connecting...' : 'Connect' }}
          </button>
        </form>
      </div>
    </div>`,
  props: ['onLogin'],
  setup(props) {
    const token = ref('');
    const error = ref(null);
    const busy = ref(false);

    async function login() {
      busy.value = true;
      error.value = null;
      api.setToken(token.value);
      const check = await api.check();
      busy.value = false;
      if (check.ok) {
        props.onLogin();
      } else if (check.needsAuth) {
        error.value = 'Invalid token';
        api.setToken('');
      } else {
        error.value = check.error || 'Cannot reach server';
      }
    }
    return { token, error, busy, login };
  },
};

// ---------------------------------------------------------------------------
// Root App
// ---------------------------------------------------------------------------
const App = {
  template: `
    <div v-if="authState === 'checking'" class="min-h-screen flex items-center justify-center">
      <div class="spinner"></div>
    </div>
    <login-screen v-else-if="authState === 'login'" :on-login="onLogin" />
    <div v-else class="flex min-h-screen">
      <!-- Sidebar -->
      <aside class="loki-sidebar" :class="{ collapsed: sidebarCollapsed, 'mobile-open': mobileOpen }">
        <div class="flex items-center gap-2 px-3 py-3 border-b border-gray-800">
          <button @click="toggleSidebar" class="btn-ghost p-1 rounded" title="Toggle sidebar">
            <span style="font-size:1.1rem;">{{ sidebarCollapsed ? '\u{25B6}' : '\u{2630}' }}</span>
          </button>
          <span class="sidebar-header-text font-semibold text-sm tracking-wide">LOKI</span>
        </div>
        <nav class="flex-1 py-2 overflow-y-auto">
          <router-link
            v-for="r in navRoutes"
            :key="r.path"
            :to="r.path"
            class="nav-item"
            active-class="active"
            @click="mobileOpen = false"
          >
            <span class="nav-icon">{{ r.meta.icon }}</span>
            <span class="nav-label">{{ r.meta.label }}</span>
          </router-link>
        </nav>
        <div class="px-3 py-2 border-t border-gray-800 text-xs text-gray-500 sidebar-header-text">
          <div class="flex items-center gap-1.5 mb-1">
            <span class="status-dot" :class="wsConnected ? 'online' : 'offline'"></span>
            {{ wsConnected ? 'Live' : 'Disconnected' }}
          </div>
          <div class="text-gray-600" style="font-size:0.625rem;">
            <kbd class="px-1 py-0.5 bg-gray-800 rounded">/</kbd> search
            <kbd class="px-1 py-0.5 bg-gray-800 rounded ml-1">Esc</kbd> close
          </div>
        </div>
      </aside>

      <!-- Mobile overlay -->
      <div v-if="mobileOpen" class="fixed inset-0 bg-black/50 z-30 md:hidden" @click="mobileOpen = false"></div>

      <!-- Main content -->
      <div class="loki-main">
        <div class="loki-topbar">
          <button class="btn-ghost p-1 rounded md:hidden" @click="mobileOpen = !mobileOpen">
            <span style="font-size:1.1rem;">\u{2630}</span>
          </button>
          <div class="flex items-center gap-2">
            <span class="status-dot" :class="botStatus"></span>
            <span class="text-sm font-medium">Loki</span>
          </div>
          <span v-if="botUptime" class="text-xs text-gray-500">{{ botUptime }}</span>
          <div class="flex-1"></div>
          <button @click="logout" class="btn btn-ghost text-xs">Logout</button>
        </div>
        <router-view />
      </div>
    </div>`,
  setup() {
    const authState = ref('checking'); // 'checking' | 'login' | 'ready'
    const sidebarCollapsed = ref(false);
    const mobileOpen = ref(false);
    const wsConnected = ref(false);
    const botStatus = ref('starting');
    const botUptime = ref('');

    const navRoutes = routes.filter(r => r.meta);

    // Global keyboard shortcuts
    function onKeydown(e) {
      // Esc: close mobile sidebar, or modals (modals handle their own Esc via @click.self)
      if (e.key === 'Escape') {
        if (mobileOpen.value) { mobileOpen.value = false; e.preventDefault(); return; }
      }
      // / : focus first search input on page (unless already in an input)
      if (e.key === '/' && !['INPUT', 'TEXTAREA', 'SELECT'].includes(e.target.tagName)) {
        e.preventDefault();
        const input = document.querySelector('.loki-main input[type="text"], .loki-main .loki-input:not(textarea):not(select)');
        if (input) input.focus();
      }
    }

    // Check auth on mount
    onMounted(async () => {
      document.addEventListener('keydown', onKeydown);
      const check = await api.check();
      if (check.ok) {
        authState.value = 'ready';
        startLive();
      } else if (check.needsAuth) {
        authState.value = 'login';
      } else {
        // Server unreachable — try without auth
        authState.value = 'ready';
        startLive();
      }
    });

    function onLogin() {
      authState.value = 'ready';
      startLive();
    }

    function logout() {
      api.setToken('');
      ws.disconnect();
      authState.value = 'login';
    }

    function toggleSidebar() {
      sidebarCollapsed.value = !sidebarCollapsed.value;
    }

    // Live updates
    let statusInterval = null;

    function startLive() {
      ws.onStatusChange = (connected) => { wsConnected.value = connected; };
      ws.connect();
      fetchStatus();
      statusInterval = setInterval(fetchStatus, 15000);
    }

    async function fetchStatus() {
      try {
        const s = await api.get('/api/status');
        botStatus.value = s.status === 'online' ? 'online' : 'starting';
        const sec = s.uptime_seconds || 0;
        const h = Math.floor(sec / 3600);
        const m = Math.floor((sec % 3600) / 60);
        botUptime.value = `${h}h ${m}m uptime`;
      } catch {
        botStatus.value = 'offline';
        botUptime.value = '';
      }
    }

    onUnmounted(() => {
      if (statusInterval) clearInterval(statusInterval);
      ws.disconnect();
      document.removeEventListener('keydown', onKeydown);
    });

    return {
      authState, sidebarCollapsed, mobileOpen, wsConnected,
      botStatus, botUptime, navRoutes,
      onLogin, logout, toggleSidebar,
    };
  },
};

// ---------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------
const app = createApp(App);
app.component('login-screen', LoginScreen);
app.use(router);
app.mount('#app');
