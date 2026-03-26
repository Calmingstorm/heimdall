/**
 * Loki Management UI — API Client + WebSocket Manager
 *
 * Usage:
 *   import { api, ws } from './api.js';
 *   const status = await api.get('/api/status');
 *   ws.connect(); ws.subscribe('events', handler);
 */

class LokiAPI {
  constructor() {
    this._token = sessionStorage.getItem('loki_token') || '';
  }

  get token() { return this._token; }

  setToken(token) {
    this._token = token;
    if (token) {
      sessionStorage.setItem('loki_token', token);
    } else {
      sessionStorage.removeItem('loki_token');
    }
  }

  _headers(extra = {}) {
    const h = { 'Content-Type': 'application/json', ...extra };
    if (this._token) h['Authorization'] = `Bearer ${this._token}`;
    return h;
  }

  async _request(method, path, body = null) {
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

class LokiWebSocket {
  constructor(api) {
    this._api = api;
    this._ws = null;
    this._handlers = { logs: [], events: [], chat: [] };
    this._reconnectDelay = 1000;
    this._maxReconnectDelay = 30000;
    this._shouldConnect = false;
    this._subscriptions = new Set();
    this.onStatusChange = null; // callback(connected: boolean)
  }

  get connected() { return this._ws?.readyState === WebSocket.OPEN; }

  connect() {
    this._shouldConnect = true;
    this._open();
  }

  disconnect() {
    this._shouldConnect = false;
    if (this._ws) {
      this._ws.close();
      this._ws = null;
    }
  }

  subscribe(channel, handler) {
    if (!this._handlers[channel]) this._handlers[channel] = [];
    this._handlers[channel].push(handler);
    this._subscriptions.add(channel);
    if (this.connected) {
      this._ws.send(JSON.stringify({ subscribe: channel }));
    }
  }

  unsubscribe(channel, handler) {
    const arr = this._handlers[channel];
    if (arr) {
      const idx = arr.indexOf(handler);
      if (idx >= 0) arr.splice(idx, 1);
      if (arr.length === 0) {
        this._subscriptions.delete(channel);
        if (this.connected) {
          this._ws.send(JSON.stringify({ unsubscribe: channel }));
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
      // Re-subscribe to channels
      for (const ch of this._subscriptions) {
        this._ws.send(JSON.stringify({ subscribe: ch }));
      }
      if (this.onStatusChange) this.onStatusChange(true);
    };

    this._ws.onmessage = (evt) => {
      let data;
      try { data = JSON.parse(evt.data); } catch { return; }
      const type = data.type;
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
      if (this.onStatusChange) this.onStatusChange(false);
      if (this._shouldConnect) {
        setTimeout(() => this._open(), this._reconnectDelay);
        this._reconnectDelay = Math.min(this._reconnectDelay * 2, this._maxReconnectDelay);
      }
    };

    this._ws.onerror = () => {
      // onclose will fire after onerror, handled there
    };
  }
}

// Singleton instances
export const api = new LokiAPI();
export const ws = new LokiWebSocket(api);
export { AuthError, ApiError };
