/**
 * Heimdall Management UI — Chat Page
 * WebSocket-based chat with REST fallback, markdown rendering, tool call display
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
  // Fallback: escape HTML and convert newlines
  return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/\n/g, '<br>');
}

export default {
  template: `
    <div class="chat-container page-fade-in">
      <!-- Message list -->
      <div class="chat-messages" ref="messagesEl">
        <div v-if="messages.length === 0" class="chat-empty">
          <div class="empty-state">
            <span class="empty-state-icon">\u{1F4AD}</span>
            <span class="empty-state-text">No messages yet</span>
            <span class="empty-state-hint">Say something to Heimdall to start a conversation</span>
          </div>
        </div>
        <div v-for="(msg, i) in messages" :key="i" class="chat-message" :class="'chat-' + msg.role">
          <!-- User message -->
          <div v-if="msg.role === 'user'" class="chat-bubble chat-bubble-user">
            <div class="chat-bubble-text">{{ msg.content }}</div>
          </div>
          <!-- Heimdall response -->
          <div v-else class="chat-bubble chat-bubble-bot">
            <div class="chat-bubble-label">Heimdall</div>
            <div v-if="msg.tools_used && msg.tools_used.length > 0" class="chat-tools">
              <button class="chat-tools-toggle" @click="msg._showTools = !msg._showTools">
                <span class="text-xs">{{ msg._showTools ? '\u25BC' : '\u25B6' }}</span>
                <span>{{ msg.tools_used.length }} tool{{ msg.tools_used.length > 1 ? 's' : '' }} used</span>
              </button>
              <div v-if="msg._showTools" class="chat-tools-list">
                <span v-for="t in msg.tools_used" :key="t" class="badge badge-info">{{ t }}</span>
              </div>
            </div>
            <div class="chat-bubble-text chat-markdown" v-html="msg.html"></div>
            <div v-if="msg.is_error" class="chat-error-badge">
              <span class="badge badge-danger">error</span>
            </div>
          </div>
        </div>
        <!-- Typing indicator -->
        <div v-if="sending" class="chat-message chat-bot">
          <div class="chat-bubble chat-bubble-bot">
            <div class="chat-bubble-label">Heimdall</div>
            <div class="chat-typing">
              <span></span><span></span><span></span>
            </div>
          </div>
        </div>
      </div>

      <!-- Input area -->
      <div class="chat-input-area">
        <div class="chat-input-row">
          <textarea
            ref="inputEl"
            v-model="input"
            class="chat-input"
            placeholder="Message Heimdall..."
            rows="1"
            :disabled="sending"
            @keydown.enter.exact.prevent="send"
            @input="autoResize"
          ></textarea>
          <button class="btn btn-primary chat-send-btn" :disabled="!canSend" @click="send">
            <span v-if="sending" class="spinner" style="width:14px;height:14px;border-width:2px;"></span>
            <span v-else>Send</span>
          </button>
        </div>
        <div class="chat-input-hint">
          <span class="text-gray-600 text-xs">Enter to send &middot; Shift+Enter for newline</span>
          <span class="text-gray-600 text-xs">{{ wsStatus }}</span>
        </div>
      </div>
    </div>`,

  setup() {
    const messages = ref([]);
    const input = ref('');
    const sending = ref(false);
    const messagesEl = ref(null);
    const inputEl = ref(null);
    let sentViaWs = false; // Track if current send used WebSocket (for timeout)

    const canSend = computed(() => input.value.trim().length > 0 && !sending.value);
    const wsStatus = computed(() => ws.connected ? 'WebSocket' : 'REST');

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
        role,
        content,
        html: role === 'bot' ? renderMarkdown(content) : '',
        tools_used: extras.tools_used || [],
        is_error: extras.is_error || false,
        _showTools: false,
      };
      messages.value.push(msg);
      scrollToBottom();
      return msg;
    }

    // WebSocket chat handler
    function onChatMessage(data) {
      if (!sending.value) return; // Ignore stale responses
      sending.value = false;
      sentViaWs = false;
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

      // Reset textarea height
      if (inputEl.value) inputEl.value.style.height = 'auto';

      // Try WebSocket first, fall back to REST
      if (ws.connected) {
        const sent = ws.sendChat(content, { channelId: 'web-default' });
        if (sent) {
          sentViaWs = true;
          startWsTimeout();
          // Response will come via onChatMessage callback
        } else {
          await sendViaRest(content);
          sending.value = false;
        }
      } else {
        await sendViaRest(content);
        sending.value = false;
      }

      nextTick(() => inputEl.value?.focus());
    }

    // Timeout for WS responses — fall back after 120s
    let wsTimeout = null;
    watch(sending, (val) => {
      if (!val) {
        // Clear timeout when sending completes (REST or WS response arrived)
        if (wsTimeout) { clearTimeout(wsTimeout); wsTimeout = null; }
      }
    });
    // Set timeout after WS send (called from send() after sentViaWs is set)
    function startWsTimeout() {
      wsTimeout = setTimeout(() => {
        if (sending.value) {
          sending.value = false;
          sentViaWs = false;
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
    });

    return {
      messages, input, sending, messagesEl, inputEl,
      canSend, wsStatus,
      send, autoResize,
    };
  },
};
