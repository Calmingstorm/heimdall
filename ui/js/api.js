/**
 * Heimdall Management UI — API Client + WebSocket Manager
 *
 * Usage:
 *   import { api, ws } from './api.js';
 *   const status = await api.get('/api/status');
 *   ws.connect(); ws.subscribe('events', handler);
 */

class HeimdallAPI {
  constructor() {
    this._token = sessionStorage.getItem('heimdall_token') || '';
    this._sessionTimeout = 0;
    this._lastActivity = Date.now();
    this._activityTimer = null;
    this.onSessionExpired = null; // callback when session times out
  }

  get token() { return this._token; }
  get sessionTimeout() { return this._sessionTimeout; }

  setToken(token, timeoutSeconds = 0) {
    this._token = token;
    this._sessionTimeout = timeoutSeconds;
    this._lastActivity = Date.now();
    if (token) {
      sessionStorage.setItem('heimdall_token', token);
      if (timeoutSeconds > 0) {
        sessionStorage.setItem('heimdall_session_timeout', String(timeoutSeconds));
      }
      this._startActivityMonitor();
    } else {
      sessionStorage.removeItem('heimdall_token');
      sessionStorage.removeItem('heimdall_session_timeout');
      this._stopActivityMonitor();
    }
  }

  _startActivityMonitor() {
    this._stopActivityMonitor();
    if (this._sessionTimeout <= 0) return;
    this._activityTimer = setInterval(() => {
      const elapsed = (Date.now() - this._lastActivity) / 1000;
      if (elapsed >= this._sessionTimeout) {
        this._stopActivityMonitor();
        if (this.onSessionExpired) this.onSessionExpired();
      }
    }, 10000); // Check every 10s
  }

  _stopActivityMonitor() {
    if (this._activityTimer) {
      clearInterval(this._activityTimer);
      this._activityTimer = null;
    }
  }

  _headers(extra = {}) {
    const h = { 'Content-Type': 'application/json', ...extra };
    if (this._token) h['Authorization'] = `Bearer ${this._token}`;
    return h;
  }

  async _request(method, path, body = null) {
    this._lastActivity = Date.now();
    const opts = { method, headers: this._headers() };
    if (body !== null) opts.body = JSON.stringify(body);
    const resp = await fetch(path, opts);
    if (resp.status === 401) {
      throw new AuthError('Unauthorized');
    }
    const data = await resp.json().catch(() => null);
    if (!resp.ok) {
      const msg = data?.error || `HTTP ${resp.status}`;
      throw new ApiError(msg, resp.status, data);
    }
    return data;
  }

  get(path) { return this._request('GET', path); }
  post(path, data) { return this._request('POST', path, data); }
  put(path, data) { return this._request('PUT', path, data); }
  del(path) { return this._request('DELETE', path); }

  /** Authenticate with the server. Returns session info or throws. */
  async login(token) {
    const resp = await fetch('/api/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ token }),
    });
    const data = await resp.json().catch(() => null);
    if (!resp.ok) {
      throw new AuthError(data?.error || 'Login failed');
    }
    this.setToken(data.session_id, data.timeout_seconds || 0);
    return data;
  }

  /** Logout — invalidate the server-side session. */
  async logout() {
    try {
      await this.post('/api/auth/logout', {});
    } catch { /* ignore errors during logout */ }
    this.setToken('');
  }

  /** Check if the server is reachable and auth is valid. */
  async check() {
    try {
      await this.get('/api/status');
      return { ok: true, needsAuth: false };
    } catch (e) {
      if (e instanceof AuthError) return { ok: false, needsAuth: true };
      return { ok: false, needsAuth: false, error: e.message };
    }
  }
}

class AuthError extends Error {
  constructor(msg) { super(msg); this.name = 'AuthError'; }
}

class ApiError extends Error {
  constructor(msg, status, data) {
    super(msg);
    this.name = 'ApiError';
    this.status = status;
    this.data = data;
  }
}

class HeimdallWebSocket {
  constructor(api) {
    this._api = api;
    this._ws = null;
    this._handlers = { logs: [], events: [], chat: [] };
    this._reconnectDelay = 1000;
    this._maxReconnectDelay = 30000;
    this._shouldConnect = false;
    this._subscriptions = new Set();
    this._reconnectAttempt = 0;
    this._lastPongTime = 0;
    this._pingInterval = null;
    this._latency = -1;
    // state: 'disconnected' | 'connecting' | 'connected' | 'reconnecting'
    this._state = 'disconnected';
    this.onStatusChange = null; // callback(connected: boolean)
    this.onStateChange = null;  // callback(state: string, detail: object)
  }

  get connected() { return this._ws?.readyState === WebSocket.OPEN; }

  /** Current connection state with detail. */
  get state() { return this._state; }
  get reconnectAttempt() { return this._reconnectAttempt; }
  get latency() { return this._latency; }

  connect() {
    this._shouldConnect = true;
    this._setState('connecting');
    this._open();
  }

  disconnect() {
    this._shouldConnect = false;
    this._reconnectAttempt = 0;
    this._latency = -1;
    this._stopPing();
    if (this._ws) {
      this._ws.close();
      this._ws = null;
    }
    this._setState('disconnected');
  }

  _setState(state) {
    if (this._state === state) return;
    this._state = state;
    if (this.onStateChange) {
      this.onStateChange(state, {
        attempt: this._reconnectAttempt,
        latency: this._latency,
      });
    }
  }

  _startPing() {
    this._stopPing();
    this._pingInterval = setInterval(() => {
      if (this.connected) {
        try {
          this._ws.send(JSON.stringify({ type: 'ping', ts: Date.now() }));
        } catch { /* ignore */ }
      }
    }, 15000);
  }

  _stopPing() {
    if (this._pingInterval) {
      clearInterval(this._pingInterval);
      this._pingInterval = null;
    }
  }

  subscribe(channel, handler) {
    if (!this._handlers[channel]) this._handlers[channel] = [];
    this._handlers[channel].push(handler);
    // Only pub/sub channels need server-side subscription (not chat — it's request/response)
    if (channel !== 'chat') {
      this._subscriptions.add(channel);
      if (this.connected) {
        this._ws.send(JSON.stringify({ subscribe: channel }));
      }
    }
  }

  unsubscribe(channel, handler) {
    const arr = this._handlers[channel];
    if (arr) {
      const idx = arr.indexOf(handler);
      if (idx >= 0) arr.splice(idx, 1);
      if (arr.length === 0) {
        if (channel !== 'chat') {
          this._subscriptions.delete(channel);
          if (this.connected) {
            this._ws.send(JSON.stringify({ unsubscribe: channel }));
          }
        }
      }
    }
  }

  /** Send a chat message via WebSocket. Returns true if sent. */
  sendChat(content, { channelId, userId, username } = {}) {
    if (!this.connected) return false;
    this._ws.send(JSON.stringify({
      type: 'chat',
      content,
      channel_id: channelId || 'web-default',
      user_id: userId || undefined,
      username: username || undefined,
    }));
    return true;
  }

  _open() {
    if (this._ws) return;
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    let url = `${proto}//${location.host}/api/ws`;
    if (this._api.token) url += `?token=${encodeURIComponent(this._api.token)}`;
    this._ws = new WebSocket(url);

    this._ws.onopen = () => {
      this._reconnectDelay = 1000;
      this._reconnectAttempt = 0;
      // Re-subscribe to channels
      for (const ch of this._subscriptions) {
        this._ws.send(JSON.stringify({ subscribe: ch }));
      }
      this._startPing();
      this._setState('connected');
      if (this.onStatusChange) this.onStatusChange(true);
    };

    this._ws.onmessage = (evt) => {
      let data;
      try { data = JSON.parse(evt.data); } catch { return; }
      const type = data.type;
      if (type === 'pong') {
        if (data.ts) {
          this._latency = Date.now() - data.ts;
          this._lastPongTime = Date.now();
        }
        return;
      }
      if (type === 'log') {
        for (const h of this._handlers.logs || []) h(data);
      } else if (type === 'event') {
        for (const h of this._handlers.events || []) h(data);
      } else if (type === 'chat_response' || type === 'chat_error') {
        for (const h of this._handlers.chat || []) h(data);
      }
      // subscribed/unsubscribed confirmations are silently consumed
    };

    this._ws.onclose = () => {
      this._ws = null;
      this._stopPing();
      this._latency = -1;
      if (this.onStatusChange) this.onStatusChange(false);
      if (this._shouldConnect) {
        this._reconnectAttempt++;
        this._setState('reconnecting');
        setTimeout(() => this._open(), this._reconnectDelay);
        this._reconnectDelay = Math.min(this._reconnectDelay * 2, this._maxReconnectDelay);
      } else {
        this._setState('disconnected');
      }
    };

    this._ws.onerror = () => {
      // onclose will fire after onerror, handled there
    };
  }
}

// Singleton instances
export const api = new HeimdallAPI();
export const ws = new HeimdallWebSocket(api);
export { AuthError, ApiError };
