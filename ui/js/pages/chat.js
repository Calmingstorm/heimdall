/**
 * Heimdall Management UI — Chat Page
 * Polished messaging interface with bubbles, timestamps, tool cards,
 * markdown code-copy, inline images, and typing indicator.
 */
import { api, ws } from '../api.js';

const { ref, computed, onMounted, onUnmounted, nextTick, watch } = Vue;

// Configure marked for safe rendering
const markedOpts = { breaks: true, gfm: true };

function renderMarkdown(text) {
  if (!text) return '';
  try {
    if (typeof marked !== 'undefined' && marked.parse) {
      return marked.parse(text, markedOpts);
    }
  } catch { /* fall through */ }
  return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/\n/g, '<br>');
}

/** Format a timestamp for display */
function formatTime(ts) {
  const d = new Date(ts);
  const h = d.getHours().toString().padStart(2, '0');
  const m = d.getMinutes().toString().padStart(2, '0');
  return `${h}:${m}`;
}

/** Map tool names to category icons */
const TOOL_ICONS = {
  run_command: '\u2318', ssh_command: '\u2318', run_script: '\u2318',
  read_file: '\uD83D\uDCC4', write_file: '\u270F\uFE0F', list_directory: '\uD83D\uDCC2',
  search_knowledge: '\uD83D\uDD0D', ingest_document: '\uD83D\uDCDA',
  generate_image: '\uD83C\uDFA8', analyze_image: '\uD83D\uDDBC\uFE0F',
  analyze_pdf: '\uD83D\uDCC3', browser_screenshot: '\uD83C\uDF10',
  manage_process: '\u2699\uFE0F', check_service: '\uD83D\uDCCA',
};

function getToolIcon(name) {
  if (TOOL_ICONS[name]) return TOOL_ICONS[name];
  if (name.startsWith('incus_')) return '\uD83D\uDCE6';
  if (name.startsWith('systemd_')) return '\u2699\uFE0F';
  if (name.startsWith('prometheus_')) return '\uD83D\uDCCA';
  return '\uD83D\uDD27';
}

/** Detect image URLs in text for inline display */
const IMG_URL_RE = /https?:\/\/\S+\.(?:png|jpg|jpeg|gif|webp|svg)(?:\?\S*)?/gi;

function extractImageUrls(text) {
  if (!text) return [];
  const matches = text.match(IMG_URL_RE);
  return matches ? [...new Set(matches)] : [];
}

export default {
  template: `
    <div class="chat-container page-fade-in" role="region" aria-label="Chat">
      <!-- Message list -->
      <div class="chat-messages" ref="messagesEl" role="log" aria-live="polite" aria-label="Messages">
        <!-- Empty state -->
        <div v-if="messages.length === 0" class="chat-empty">
          <div class="chat-welcome">
            <div class="chat-welcome-icon">
              <svg viewBox="0 0 24 24" width="48" height="48" fill="none" stroke="currentColor" stroke-width="1.5">
                <circle cx="12" cy="12" r="3"/>
                <path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/>
                <circle cx="12" cy="12" r="9" stroke-dasharray="4 3"/>
              </svg>
            </div>
            <div class="chat-welcome-title">Heimdall is watching</div>
            <div class="chat-welcome-subtitle">Ask anything. Run commands. Manage infrastructure.</div>
            <div class="chat-suggestions">
              <button v-for="s in suggestions" :key="s" class="chat-suggestion" @click="useSuggestion(s)">{{ s }}</button>
            </div>
          </div>
        </div>

        <!-- Messages -->
        <template v-for="(msg, i) in messages" :key="msg.id">
          <!-- Date separator -->
          <div v-if="showDateSeparator(i)" class="chat-date-sep">
            <span>{{ formatDate(msg.timestamp) }}</span>
          </div>

          <div class="chat-message" :class="'chat-' + msg.role">
            <!-- Avatar -->
            <div class="chat-avatar" :class="'chat-avatar-' + msg.role">
              <span v-if="msg.role === 'bot'" class="chat-avatar-eye">
                <svg viewBox="0 0 20 20" width="16" height="16" fill="currentColor">
                  <path d="M10 3C5 3 1.73 7.11 1 10c.73 2.89 4 7 9 7s8.27-4.11 9-7c-.73-2.89-4-7-9-7zm0 12a5 5 0 110-10 5 5 0 010 10zm0-8a3 3 0 100 6 3 3 0 000-6z"/>
                </svg>
              </span>
              <span v-else class="chat-avatar-user">
                <svg viewBox="0 0 20 20" width="14" height="14" fill="currentColor">
                  <path d="M10 10a4 4 0 100-8 4 4 0 000 8zm-7 8a7 7 0 0114 0H3z"/>
                </svg>
              </span>
            </div>

            <!-- Bubble -->
            <div class="chat-bubble-wrap">
              <!-- User message -->
              <div v-if="msg.role === 'user'" class="chat-bubble chat-bubble-user">
                <div class="chat-bubble-text">{{ msg.content }}</div>
              </div>

              <!-- Bot message -->
              <div v-else class="chat-bubble chat-bubble-bot">
                <div class="chat-bubble-header">
                  <span class="chat-bubble-label">Heimdall</span>
                  <span v-if="msg.is_error" class="chat-error-indicator">error</span>
                </div>

                <!-- Tool cards -->
                <div v-if="msg.tools_used && msg.tools_used.length > 0" class="chat-tool-cards">
                  <button class="chat-tools-toggle" @click="msg._showTools = !msg._showTools"
                          :aria-expanded="msg._showTools" aria-label="Toggle tool details">
                    <span class="chat-tools-toggle-icon" aria-hidden="true">{{ msg._showTools ? '\u25BC' : '\u25B6' }}</span>
                    <span class="chat-tools-toggle-count">{{ msg.tools_used.length }}</span>
                    <span>tool{{ msg.tools_used.length > 1 ? 's' : '' }} executed</span>
                  </button>
                  <div v-if="msg._showTools" class="chat-tool-list">
                    <div v-for="t in msg.tools_used" :key="t" class="chat-tool-card">
                      <span class="chat-tool-icon">{{ getToolIcon(t) }}</span>
                      <span class="chat-tool-name">{{ t }}</span>
                    </div>
                  </div>
                </div>

                <!-- Markdown body -->
                <div class="chat-bubble-text chat-markdown" v-html="msg.html"></div>

                <!-- Inline images -->
                <div v-if="msg.images && msg.images.length > 0" class="chat-images">
                  <div v-for="(url, j) in msg.images" :key="j" class="chat-image-thumb">
                    <img :src="url" :alt="'Image ' + (j+1)" loading="lazy" @click="openImage(url)" @error="onImageError($event)"/>
                  </div>
                </div>
              </div>

              <!-- Timestamp -->
              <div class="chat-timestamp">{{ formatTime(msg.timestamp) }}</div>
            </div>
          </div>
        </template>

        <!-- Typing indicator -->
        <div v-if="sending" class="chat-message chat-bot" role="status" aria-label="Heimdall is responding">
          <div class="chat-avatar chat-avatar-bot">
            <span class="chat-avatar-eye chat-avatar-pulse">
              <svg viewBox="0 0 20 20" width="16" height="16" fill="currentColor" aria-hidden="true">
                <path d="M10 3C5 3 1.73 7.11 1 10c.73 2.89 4 7 9 7s8.27-4.11 9-7c-.73-2.89-4-7-9-7zm0 12a5 5 0 110-10 5 5 0 010 10zm0-8a3 3 0 100 6 3 3 0 000-6z"/>
              </svg>
            </span>
          </div>
          <div class="chat-bubble-wrap">
            <div class="chat-bubble chat-bubble-bot chat-bubble-typing">
              <div class="chat-typing" aria-hidden="true">
                <span></span><span></span><span></span>
              </div>
              <span class="chat-typing-text">{{ typingText }}</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Input area -->
      <div class="chat-input-area" role="form" aria-label="Send message">
        <div class="chat-input-row">
          <label for="chat-message-input" class="sr-only">Message</label>
          <textarea
            id="chat-message-input"
            ref="inputEl"
            v-model="input"
            class="chat-input"
            placeholder="Message Heimdall..."
            rows="1"
            :disabled="sending"
            @keydown.enter.exact.prevent="send"
            @input="autoResize"
          ></textarea>
          <button class="btn btn-primary chat-send-btn" :disabled="!canSend" @click="send" aria-label="Send message">
            <span v-if="sending" class="spinner" style="width:14px;height:14px;border-width:2px;" aria-hidden="true"></span>
            <svg v-else viewBox="0 0 20 20" width="16" height="16" fill="currentColor" class="chat-send-icon" aria-hidden="true">
              <path d="M2.94 5.34l6.22 2.6L2.94 5.34zM9.16 12.06l-6.22 2.6 1.36-5.2 4.86 2.6zM18.44 10L2.12 2.4l2.06 7.6-2.06 7.6L18.44 10z"/>
            </svg>
          </button>
        </div>
        <div class="chat-input-hint">
          <span class="text-gray-600 text-xs">Enter to send &middot; Shift+Enter for newline</span>
          <span class="chat-connection-status" :class="ws.connected ? 'chat-ws-on' : 'chat-ws-off'">
            <span class="chat-status-dot"></span>
            {{ wsStatus }}
          </span>
        </div>
      </div>
    </div>`,

  setup() {
    const messages = ref([]);
    const input = ref('');
    const sending = ref(false);
    const messagesEl = ref(null);
    const inputEl = ref(null);
    const typingElapsed = ref(0);
    let typingTimer = null;
    let sentViaWs = false;
    let msgIdCounter = 0;

    const suggestions = [
      'Check system health',
      'List running services',
      'Show disk usage',
      'What can you do?',
    ];

    const canSend = computed(() => input.value.trim().length > 0 && !sending.value);
    const wsStatus = computed(() => ws.connected ? 'Connected' : 'REST fallback');

    const typingPhrases = [
      'Watching across all realms...',
      'Processing...',
      'Consulting the bifrost...',
      'Observing...',
    ];
    const typingText = computed(() => {
      const idx = Math.floor(typingElapsed.value / 4) % typingPhrases.length;
      const secs = typingElapsed.value;
      return secs > 3 ? `${typingPhrases[idx]} (${secs}s)` : typingPhrases[0];
    });

    function scrollToBottom() {
      nextTick(() => {
        if (messagesEl.value) {
          messagesEl.value.scrollTop = messagesEl.value.scrollHeight;
        }
      });
    }

    function autoResize() {
      if (!inputEl.value) return;
      const el = inputEl.value;
      el.style.height = 'auto';
      el.style.height = Math.min(el.scrollHeight, 120) + 'px';
    }

    function addMessage(role, content, extras = {}) {
      const msg = {
        id: ++msgIdCounter,
        role,
        content,
        timestamp: Date.now(),
        html: role === 'bot' ? renderMarkdown(content) : '',
        tools_used: extras.tools_used || [],
        is_error: extras.is_error || false,
        images: role === 'bot' ? extractImageUrls(content) : [],
        _showTools: false,
      };
      messages.value.push(msg);
      scrollToBottom();
      // Attach copy buttons after render
      if (role === 'bot') {
        nextTick(() => attachCopyButtons());
      }
      return msg;
    }

    /** Attach copy-to-clipboard buttons on code blocks */
    function attachCopyButtons() {
      if (!messagesEl.value) return;
      const blocks = messagesEl.value.querySelectorAll('.chat-markdown pre:not([data-copy])');
      blocks.forEach(pre => {
        pre.setAttribute('data-copy', 'true');
        pre.style.position = 'relative';
        const btn = document.createElement('button');
        btn.className = 'chat-code-copy';
        btn.textContent = 'Copy';
        btn.addEventListener('click', () => {
          const code = pre.querySelector('code');
          const text = code ? code.textContent : pre.textContent;
          navigator.clipboard.writeText(text).then(() => {
            btn.textContent = 'Copied!';
            setTimeout(() => { btn.textContent = 'Copy'; }, 1500);
          }).catch(() => {});
        });
        pre.appendChild(btn);
      });
    }

    function showDateSeparator(i) {
      if (i === 0) return true;
      const prev = messages.value[i - 1];
      const curr = messages.value[i];
      const pDay = new Date(prev.timestamp).toDateString();
      const cDay = new Date(curr.timestamp).toDateString();
      return pDay !== cDay;
    }

    function formatDate(ts) {
      const d = new Date(ts);
      const today = new Date();
      if (d.toDateString() === today.toDateString()) return 'Today';
      const yesterday = new Date(today);
      yesterday.setDate(yesterday.getDate() - 1);
      if (d.toDateString() === yesterday.toDateString()) return 'Yesterday';
      return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
    }

    function useSuggestion(text) {
      input.value = text;
      nextTick(() => send());
    }

    function openImage(url) {
      window.open(url, '_blank', 'noopener');
    }

    function onImageError(event) {
      event.target.style.display = 'none';
    }

    function startTypingTimer() {
      typingElapsed.value = 0;
      typingTimer = setInterval(() => {
        typingElapsed.value++;
      }, 1000);
    }

    function stopTypingTimer() {
      if (typingTimer) { clearInterval(typingTimer); typingTimer = null; }
      typingElapsed.value = 0;
    }

    // WebSocket chat handler
    function onChatMessage(data) {
      if (!sending.value) return;
      sending.value = false;
      sentViaWs = false;
      stopTypingTimer();
      if (data.type === 'chat_response') {
        addMessage('bot', data.content, {
          tools_used: data.tools_used || [],
          is_error: data.is_error || false,
        });
      } else if (data.type === 'chat_error') {
        addMessage('bot', data.error || 'Unknown error', { is_error: true });
      }
      nextTick(() => inputEl.value?.focus());
    }

    async function sendViaRest(content) {
      try {
        const result = await api.post('/api/chat', {
          content,
          channel_id: 'web-default',
        });
        addMessage('bot', result.response, {
          tools_used: result.tools_used || [],
          is_error: result.is_error || false,
        });
      } catch (e) {
        addMessage('bot', e.message || 'Failed to send message', { is_error: true });
      }
    }

    async function send() {
      const content = input.value.trim();
      if (!content || sending.value) return;

      addMessage('user', content);
      input.value = '';
      sending.value = true;
      sentViaWs = false;
      startTypingTimer();

      if (inputEl.value) inputEl.value.style.height = 'auto';

      if (ws.connected) {
        const sent = ws.sendChat(content, { channelId: 'web-default' });
        if (sent) {
          sentViaWs = true;
          startWsTimeout();
        } else {
          await sendViaRest(content);
          sending.value = false;
          stopTypingTimer();
        }
      } else {
        await sendViaRest(content);
        sending.value = false;
        stopTypingTimer();
      }

      nextTick(() => inputEl.value?.focus());
    }

    // Timeout for WS responses
    let wsTimeout = null;
    watch(sending, (val) => {
      if (!val) {
        if (wsTimeout) { clearTimeout(wsTimeout); wsTimeout = null; }
      }
    });

    function startWsTimeout() {
      wsTimeout = setTimeout(() => {
        if (sending.value) {
          sending.value = false;
          sentViaWs = false;
          stopTypingTimer();
          addMessage('bot', 'Response timed out. Try again.', { is_error: true });
        }
      }, 120000);
    }

    onMounted(() => {
      ws.subscribe('chat', onChatMessage);
      nextTick(() => inputEl.value?.focus());
    });

    onUnmounted(() => {
      ws.unsubscribe('chat', onChatMessage);
      if (wsTimeout) clearTimeout(wsTimeout);
      stopTypingTimer();
    });

    return {
      messages, input, sending, messagesEl, inputEl,
      canSend, wsStatus, typingText, suggestions,
      send, autoResize, formatTime, formatDate,
      showDateSeparator, useSuggestion, openImage,
      onImageError, getToolIcon,
    };
  },
};
