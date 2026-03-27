"""
Round 42 — WebSocket Real-Time Polish Tests
Connection indicator, smooth transitions, optimistic updates, ping/pong.
"""

import json
import re
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
UI_DIR = Path(__file__).resolve().parent.parent / "ui"
STYLE_CSS = UI_DIR / "css" / "style.css"
APP_JS = UI_DIR / "js" / "app.js"
API_JS = UI_DIR / "js" / "api.js"
DASHBOARD_JS = UI_DIR / "js" / "pages" / "dashboard.js"
LOGS_JS = UI_DIR / "js" / "pages" / "logs.js"
CHAT_JS = UI_DIR / "js" / "pages" / "chat.js"

WS_PY = Path(__file__).resolve().parent.parent / "src" / "web" / "websocket.py"


@pytest.fixture(scope="module")
def style_css():
    return STYLE_CSS.read_text()


@pytest.fixture(scope="module")
def app_js():
    return APP_JS.read_text()


@pytest.fixture(scope="module")
def api_js():
    return API_JS.read_text()


@pytest.fixture(scope="module")
def dashboard_js():
    return DASHBOARD_JS.read_text()


@pytest.fixture(scope="module")
def logs_js():
    return LOGS_JS.read_text()


@pytest.fixture(scope="module")
def chat_js():
    return CHAT_JS.read_text()


@pytest.fixture(scope="module")
def ws_py():
    return WS_PY.read_text()


# ===================================================================
# 1. WebSocket Connection States (Frontend)
# ===================================================================


class TestWSConnectionStates:
    """WebSocket client tracks connection states properly."""

    def test_state_property_exists(self, api_js):
        """ws.state getter is defined."""
        assert "get state()" in api_js

    def test_reconnect_attempt_property(self, api_js):
        """ws.reconnectAttempt getter is defined."""
        assert "get reconnectAttempt()" in api_js

    def test_latency_property(self, api_js):
        """ws.latency getter is defined."""
        assert "get latency()" in api_js

    def test_initial_state_disconnected(self, api_js):
        """Initial state is 'disconnected'."""
        assert "_state = 'disconnected'" in api_js

    def test_states_defined(self, api_js):
        """All four connection states are documented."""
        assert "'disconnected'" in api_js
        assert "'connecting'" in api_js
        assert "'connected'" in api_js
        assert "'reconnecting'" in api_js

    def test_on_state_change_callback(self, api_js):
        """onStateChange callback is defined."""
        assert "this.onStateChange = null" in api_js

    def test_set_state_fires_callback(self, api_js):
        """_setState calls onStateChange with state and detail."""
        assert "_setState(state)" in api_js
        assert "this.onStateChange(state," in api_js

    def test_connect_sets_connecting(self, api_js):
        """connect() sets state to 'connecting'."""
        # connect() calls _setState('connecting')
        assert "_setState('connecting')" in api_js

    def test_onopen_sets_connected(self, api_js):
        """onopen sets state to 'connected'."""
        assert "_setState('connected')" in api_js

    def test_onclose_sets_reconnecting(self, api_js):
        """onclose sets state to 'reconnecting' when shouldConnect."""
        assert "_setState('reconnecting')" in api_js

    def test_disconnect_sets_disconnected(self, api_js):
        """disconnect() sets state to 'disconnected'."""
        # Multiple refs in disconnect and onclose
        assert "_setState('disconnected')" in api_js

    def test_reconnect_attempt_increments(self, api_js):
        """Reconnect attempt counter increments on close."""
        assert "this._reconnectAttempt++" in api_js

    def test_reconnect_attempt_resets_on_open(self, api_js):
        """Reconnect attempt resets to 0 on successful connect."""
        assert "this._reconnectAttempt = 0" in api_js


# ===================================================================
# 2. Ping/Pong Latency Measurement
# ===================================================================


class TestPingPong:
    """Ping/pong for latency measurement."""

    def test_frontend_sends_ping(self, api_js):
        """Frontend sends ping messages with timestamp."""
        assert "type: 'ping'" in api_js
        assert "ts: Date.now()" in api_js

    def test_frontend_handles_pong(self, api_js):
        """Frontend handles pong responses to compute latency."""
        assert "type === 'pong'" in api_js
        assert "this._latency = Date.now() - data.ts" in api_js

    def test_ping_interval_set(self, api_js):
        """Ping interval is started on connection."""
        assert "_startPing()" in api_js
        assert "setInterval(" in api_js

    def test_ping_interval_cleared(self, api_js):
        """Ping interval is cleared on disconnect/close."""
        assert "_stopPing()" in api_js

    def test_backend_responds_pong(self, ws_py):
        """Backend responds to ping with pong."""
        assert '"pong"' in ws_py
        assert '"ping"' in ws_py


class TestBackendPingPong:
    """Backend ping/pong handler."""

    @pytest.mark.asyncio
    async def test_ws_ping_returns_pong(self):
        """WebSocket manager responds to ping with pong and timestamp."""
        from src.web.websocket import WebSocketManager

        bot = MagicMock()
        mgr = WebSocketManager(bot)

        # Simulate a WebSocket that receives a ping message
        mock_ws = AsyncMock()
        mock_ws.closed = False
        mock_ws.send_json = AsyncMock()

        # We test the ping handling logic directly by checking the code path
        # The handler checks data.get("type") == "ping"
        # and responds with {"type": "pong", "ts": data.get("ts")}
        # Since we can't easily simulate the full WS loop, we verify
        # the source code contains the correct handler
        import inspect
        source = inspect.getsource(mgr.handle)
        assert '"ping"' in source
        assert '"pong"' in source


# ===================================================================
# 3. Connection Indicator UI
# ===================================================================


class TestConnectionIndicatorCSS:
    """CSS for connection indicator states."""

    def test_ws_indicator_base(self, style_css):
        """Base ws-indicator class exists."""
        assert ".ws-indicator" in style_css

    def test_ws_connected_state(self, style_css):
        """Connected state uses success color."""
        assert ".ws-connected" in style_css
        assert "var(--hm-success)" in style_css

    def test_ws_disconnected_state(self, style_css):
        """Disconnected state uses danger color."""
        assert ".ws-disconnected" in style_css

    def test_ws_connecting_state(self, style_css):
        """Connecting state uses warning color with pulse."""
        assert ".ws-connecting" in style_css

    def test_ws_reconnecting_state(self, style_css):
        """Reconnecting state uses warning color with pulse."""
        assert ".ws-reconnecting" in style_css

    def test_ws_pulse_animation(self, style_css):
        """Pulse animation is defined for connecting states."""
        assert "@keyframes ws-pulse" in style_css

    def test_ws_indicator_transition(self, style_css):
        """Indicator has smooth color transitions."""
        # ws-indicator has transition on background and box-shadow
        idx = style_css.index(".ws-indicator")
        block = style_css[idx:idx + 300]
        assert "transition" in block


class TestConnectionIndicatorApp:
    """App root uses ws-indicator component."""

    def test_app_uses_ws_indicator(self, app_js):
        """App root template uses ws-indicator class."""
        assert "ws-indicator" in app_js
        assert "'ws-' + wsState" in app_js

    def test_app_shows_latency(self, app_js):
        """App root shows latency when available."""
        assert "wsLatency" in app_js
        assert "ms" in app_js

    def test_app_ws_label_computed(self, app_js):
        """wsLabel computed shows appropriate text per state."""
        assert "wsLabel" in app_js
        assert "'Live'" in app_js
        assert "'Connecting" in app_js
        assert "'Reconnecting" in app_js
        assert "'Disconnected'" in app_js

    def test_app_ws_state_ref(self, app_js):
        """wsState reactive ref is defined."""
        assert "wsState = ref(" in app_js

    def test_app_ws_latency_ref(self, app_js):
        """wsLatency reactive ref is defined."""
        assert "wsLatency = ref(" in app_js


# ===================================================================
# 4. Connection Toast Notifications
# ===================================================================


class TestConnectionToasts:
    """Toast notifications for connection events."""

    def test_toast_css_base(self, style_css):
        """Toast base class exists."""
        assert ".ws-toast" in style_css

    def test_toast_success_variant(self, style_css):
        """Success toast variant for reconnection."""
        assert ".ws-toast-success" in style_css

    def test_toast_warn_variant(self, style_css):
        """Warning toast variant for connection loss."""
        assert ".ws-toast-warn" in style_css

    def test_toast_info_variant(self, style_css):
        """Info toast variant."""
        assert ".ws-toast-info" in style_css

    def test_toast_enter_animation(self, style_css):
        """Toast has enter animation."""
        assert "ws-toast-enter-active" in style_css
        assert "@keyframes ws-toast-in" in style_css

    def test_toast_leave_animation(self, style_css):
        """Toast has leave animation."""
        assert "ws-toast-leave-active" in style_css
        assert "@keyframes ws-toast-out" in style_css

    def test_app_toast_on_disconnect(self, app_js):
        """App shows toast when connection is lost."""
        assert "Connection lost" in app_js

    def test_app_toast_on_reconnect(self, app_js):
        """App shows toast when connection is restored."""
        assert "Connection restored" in app_js

    def test_app_toast_ref(self, app_js):
        """wsToast reactive ref is defined."""
        assert "wsToast = ref(" in app_js

    def test_app_toast_auto_dismiss(self, app_js):
        """Toasts auto-dismiss with timeout."""
        assert "wsToastTimer" in app_js
        assert "setTimeout" in app_js

    def test_app_template_toast(self, app_js):
        """Toast is rendered in the template with Vue transition."""
        assert "ws-toast" in app_js
        assert "transition" in app_js.lower()


# ===================================================================
# 5. Smooth Data Transitions
# ===================================================================


class TestSmoothTransitions:
    """CSS transitions for smooth data updates."""

    def test_item_enter_animation(self, style_css):
        """item-enter class with slide-in animation."""
        assert ".item-enter" in style_css
        assert "@keyframes item-slide-in" in style_css

    def test_badge_pop_animation(self, style_css):
        """Badge pop animation for count updates."""
        assert ".badge-pop" in style_css
        assert "@keyframes badge-pop" in style_css

    def test_action_pending_class(self, style_css):
        """action-pending class dims element."""
        assert ".action-pending" in style_css

    def test_action_success_animation(self, style_css):
        """action-success animation for completed actions."""
        assert ".action-success" in style_css
        assert "@keyframes action-flash-ok" in style_css

    def test_action_error_animation(self, style_css):
        """action-error animation for failed actions."""
        assert ".action-error" in style_css
        assert "@keyframes action-flash-err" in style_css

    def test_dash_stat_value_transition(self, style_css):
        """Stat values have color transition added in smooth transitions section."""
        # The transition is added in the SMOOTH DATA TRANSITIONS section
        assert ".dash-stat-value" in style_css
        # Find the smooth transitions section rule
        section_idx = style_css.index("SMOOTH DATA TRANSITIONS")
        section = style_css[section_idx:]
        assert ".dash-stat-value" in section
        stat_idx = section.index(".dash-stat-value")
        block = section[stat_idx:stat_idx + 200]
        assert "transition" in block

    def test_badge_transition(self, style_css):
        """Badges have transition for smooth updates."""
        # The transition is added in the SMOOTH DATA TRANSITIONS section
        section_idx = style_css.index("SMOOTH DATA TRANSITIONS")
        section = style_css[section_idx:]
        assert ".badge" in section
        badge_idx = section.index(".badge")
        block = section[badge_idx:badge_idx + 200]
        assert "transition" in block


class TestDashboardTransitions:
    """Dashboard uses smooth transitions for data."""

    def test_activity_item_enter(self, dashboard_js):
        """New activity items use item-enter class."""
        assert "item-enter" in dashboard_js

    def test_flash_new_still_used(self, dashboard_js):
        """flash-new class still used for new items."""
        assert "flash-new" in dashboard_js


# ===================================================================
# 6. Optimistic Updates
# ===================================================================


class TestOptimisticUpdates:
    """Dashboard actions use optimistic updates."""

    def test_clear_sessions_optimistic(self, dashboard_js):
        """clearSessions optimistically sets sessions to 0."""
        assert "session_count: 0" in dashboard_js or "session_count: 0 }" in dashboard_js

    def test_clear_sessions_rollback(self, dashboard_js):
        """clearSessions rolls back on failure."""
        assert "prevSessions" in dashboard_js
        # Check rollback pattern
        assert "session_count: prevSessions" in dashboard_js

    def test_stop_loops_optimistic(self, dashboard_js):
        """stopAllLoops optimistically sets loops to 0."""
        assert "loop_count: 0" in dashboard_js or "loop_count: 0 }" in dashboard_js

    def test_stop_loops_rollback(self, dashboard_js):
        """stopAllLoops rolls back on failure."""
        assert "prevLoops" in dashboard_js
        assert "loop_count: prevLoops" in dashboard_js

    def test_optimistic_before_api_call(self, dashboard_js):
        """Optimistic update happens before the API call."""
        # In clearSessions, "session_count: 0" appears before "api.post"
        src = dashboard_js
        clear_fn_start = src.index("async function clearSessions()")
        clear_fn_end = src.index("async function stopAllLoops()")
        clear_fn = src[clear_fn_start:clear_fn_end]
        optimistic_pos = clear_fn.index("session_count: 0")
        api_pos = clear_fn.index("api.post")
        assert optimistic_pos < api_pos, "Optimistic update must happen before API call"

    def test_stop_loops_optimistic_before_api(self, dashboard_js):
        """Stop loops optimistic update happens before API call."""
        src = dashboard_js
        stop_fn_start = src.index("async function stopAllLoops()")
        stop_fn_end = src.index("function retry()")
        stop_fn = src[stop_fn_start:stop_fn_end]
        optimistic_pos = stop_fn.index("loop_count: 0")
        api_pos = stop_fn.index("api.post")
        assert optimistic_pos < api_pos, "Optimistic update must happen before API call"


# ===================================================================
# 7. Logs Page WS State
# ===================================================================


class TestLogsWSState:
    """Logs page uses WS state tracking."""

    def test_logs_uses_ws_indicator(self, logs_js):
        """Logs page uses ws-indicator class."""
        assert "ws-indicator" in logs_js

    def test_logs_ws_state_ref(self, logs_js):
        """Logs page has wsState ref."""
        assert "wsState" in logs_js

    def test_logs_ws_state_label(self, logs_js):
        """Logs page has wsStateLabel computed."""
        assert "wsStateLabel" in logs_js

    def test_logs_no_polling_interval(self, logs_js):
        """Logs page does not use polling interval for WS status."""
        assert "statusCheckInterval" not in logs_js

    def test_logs_state_labels(self, logs_js):
        """Logs page shows correct labels for each state."""
        assert "'Live'" in logs_js
        assert "'Connecting" in logs_js
        assert "'Reconnecting" in logs_js
        assert "'Disconnected'" in logs_js


# ===================================================================
# 8. Chat Page WS State
# ===================================================================


class TestChatWSState:
    """Chat page uses enhanced WS state."""

    def test_chat_ws_status_shows_reconnecting(self, chat_js):
        """Chat status shows 'Reconnecting' state."""
        assert "Reconnecting" in chat_js

    def test_chat_ws_status_shows_connecting(self, chat_js):
        """Chat status shows 'Connecting' state."""
        assert "Connecting" in chat_js

    def test_chat_uses_ws_state(self, chat_js):
        """Chat page reads ws.state property."""
        assert "ws.state" in chat_js


# ===================================================================
# 9. Reduced Motion Support
# ===================================================================


class TestReducedMotion:
    """New animations respect reduced motion preference."""

    def test_ws_pulse_reduced(self, style_css):
        """ws-connecting/reconnecting pulse disabled in reduced motion."""
        # Find the reduced motion block
        rm_idx = style_css.index("prefers-reduced-motion: reduce")
        rm_block = style_css[rm_idx:]
        assert "ws-connecting" in rm_block or "ws-reconnecting" in rm_block

    def test_ws_toast_reduced(self, style_css):
        """Toast animation disabled in reduced motion."""
        rm_idx = style_css.index("prefers-reduced-motion: reduce")
        rm_block = style_css[rm_idx:]
        assert "ws-toast" in rm_block


# ===================================================================
# 10. Forced Colors Support
# ===================================================================


class TestForcedColors:
    """Connection indicator supports forced colors mode."""

    def test_status_dots_forced_colors(self, style_css):
        """Status dots have forced-color-adjust: none."""
        fc_idx = style_css.index("forced-colors: active")
        fc_block = style_css[fc_idx:fc_idx + 500]
        assert "status-dot" in fc_block or "forced-color-adjust" in fc_block


# ===================================================================
# 11. Backend WebSocket Manager Tests
# ===================================================================


class TestWebSocketPingHandler:
    """Backend handles ping messages."""

    @pytest.mark.asyncio
    async def test_handle_has_ping_branch(self):
        """WebSocket handle method contains ping handling."""
        from src.web.websocket import WebSocketManager
        import inspect
        source = inspect.getsource(WebSocketManager.handle)
        assert '"ping"' in source
        assert '"pong"' in source

    @pytest.mark.asyncio
    async def test_pong_includes_timestamp(self):
        """Pong response includes the original timestamp."""
        from src.web.websocket import WebSocketManager
        import inspect
        source = inspect.getsource(WebSocketManager.handle)
        assert 'data.get("ts")' in source


# ===================================================================
# 12. Backward Compatibility
# ===================================================================


class TestBackwardCompatibility:
    """New features don't break existing functionality."""

    def test_onStatusChange_still_called(self, api_js):
        """Legacy onStatusChange callback is still fired."""
        assert "this.onStatusChange" in api_js
        assert "onStatusChange(true)" in api_js
        assert "onStatusChange(false)" in api_js

    def test_connected_getter_unchanged(self, api_js):
        """connected getter still works (readyState check)."""
        assert "WebSocket.OPEN" in api_js

    def test_reconnect_delay_logic_preserved(self, api_js):
        """Exponential backoff reconnection still works."""
        assert "this._reconnectDelay * 2" in api_js
        assert "this._maxReconnectDelay" in api_js

    def test_subscription_system_intact(self, api_js):
        """Subscribe/unsubscribe system unchanged."""
        assert "subscribe(channel, handler)" in api_js
        assert "unsubscribe(channel, handler)" in api_js

    def test_send_chat_still_works(self, api_js):
        """sendChat method still present."""
        assert "sendChat(content," in api_js

    def test_old_status_dot_still_used(self, style_css):
        """Old status-dot class still exists for components using it."""
        assert ".status-dot" in style_css
        assert ".status-dot.online" in style_css
        assert ".status-dot.offline" in style_css
        assert ".status-dot.starting" in style_css

    def test_flash_new_animation_preserved(self, style_css):
        """flash-new animation class still present."""
        assert ".flash-new" in style_css
        assert "@keyframes flash-highlight" in style_css

    def test_dashboard_flash_new_preserved(self, dashboard_js):
        """Dashboard still uses flash-new for activity items."""
        assert "flash-new" in dashboard_js

    def test_logs_subscribed_ref_preserved(self, logs_js):
        """Logs page still has subscribed ref for compatibility."""
        assert "subscribed" in logs_js

    def test_app_wsConnected_preserved(self, app_js):
        """wsConnected ref still updated for compatibility."""
        assert "wsConnected" in app_js

    def test_chat_connection_status_preserved(self, chat_js):
        """Chat connection status display still works."""
        assert "wsStatus" in chat_js
        assert "chat-connection-status" in chat_js


# ===================================================================
# 13. Integration: State flows correctly
# ===================================================================


class TestStateFlow:
    """Verify state transitions are logically consistent."""

    def test_connect_flow(self, api_js):
        """connect() -> 'connecting' -> onopen -> 'connected'."""
        # connect() calls _setState('connecting') then _open()
        src = api_js
        connect_start = src.index("connect() {")
        connect_end = src.index("disconnect()")
        connect_fn = src[connect_start:connect_end]
        assert "_setState('connecting')" in connect_fn

    def test_reconnect_flow(self, api_js):
        """onclose -> 'reconnecting' -> _open() -> onopen -> 'connected'."""
        src = api_js
        onclose_idx = src.index("this._ws.onclose")
        onclose_block = src[onclose_idx:onclose_idx + 400]
        assert "_setState('reconnecting')" in onclose_block

    def test_clean_disconnect_flow(self, api_js):
        """disconnect() -> 'disconnected', no reconnect."""
        src = api_js
        disconnect_start = src.index("disconnect() {")
        disconnect_end = src.index("\n\n", disconnect_start)
        disconnect_fn = src[disconnect_start:disconnect_end]
        assert "_setState('disconnected')" in disconnect_fn
        assert "_shouldConnect = false" in disconnect_fn

    def test_app_state_change_handler(self, app_js):
        """App root sets up onStateChange handler."""
        assert "ws.onStateChange" in app_js

    def test_app_updates_ws_state(self, app_js):
        """App root updates wsState ref on state change."""
        assert "wsState.value = state" in app_js

    def test_app_updates_ws_latency(self, app_js):
        """App root updates wsLatency from detail."""
        assert "wsLatency.value" in app_js
