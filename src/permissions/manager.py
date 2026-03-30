from __future__ import annotations

import json
from pathlib import Path

from ..logging import get_logger

log = get_logger("permissions")

VALID_TIERS = ("admin", "user", "guest")

# Tools available to the "user" tier — read-only monitoring and search.
# Everything else is admin-only.
USER_TIER_TOOLS = frozenset({
    "run_command",
    "search_history",
    "search_knowledge",
    "web_search",
    "fetch_url",
    "list_schedules",
    "list_tasks",
    "list_skills",
    "list_knowledge",
    "manage_list",
    "parse_time",
})


class PermissionManager:
    """Manages per-user permission tiers with config defaults and runtime overrides."""

    def __init__(
        self,
        config_tiers: dict[str, str],
        default_tier: str = "user",
        overrides_path: str = "./data/permissions.json",
    ) -> None:
        self._config_tiers = dict(config_tiers)
        self._default_tier = default_tier if default_tier in VALID_TIERS else "user"
        self._overrides_path = Path(overrides_path)
        self._overrides: dict[str, str] = {}
        self._load_overrides()

    def _load_overrides(self) -> None:
        if self._overrides_path.exists():
            try:
                data = json.loads(self._overrides_path.read_text())
                if isinstance(data, dict):
                    self._overrides = {
                        k: v for k, v in data.items() if v in VALID_TIERS
                    }
            except (json.JSONDecodeError, OSError) as e:
                log.warning("Failed to load permission overrides: %s", e)

    def _save_overrides(self) -> None:
        self._overrides_path.parent.mkdir(parents=True, exist_ok=True)
        self._overrides_path.write_text(json.dumps(self._overrides, indent=2))

    def get_tier(self, user_id: str) -> str:
        """Get the permission tier for a user. Runtime overrides take precedence."""
        if user_id in self._overrides:
            return self._overrides[user_id]
        return self._config_tiers.get(user_id, self._default_tier)

    def set_tier(self, user_id: str, tier: str) -> None:
        """Set a user's permission tier (persisted as runtime override)."""
        if tier not in VALID_TIERS:
            raise ValueError(f"Invalid tier '{tier}'. Must be one of: {', '.join(VALID_TIERS)}")
        self._overrides[user_id] = tier
        self._save_overrides()
        log.info("Permission tier for user %s set to %s", user_id, tier)

    def filter_tools(self, user_id: str, tools: list[dict]) -> list[dict] | None:
        """Filter tool list based on user's tier.

        Returns None for guest tier (no tools at all).
        Returns full list for admin, filtered list for user tier.
        """
        tier = self.get_tier(user_id)
        if tier == "admin":
            return tools
        if tier == "guest":
            return None
        # User tier: only allowlisted read-only tools
        return [t for t in tools if t["name"] in USER_TIER_TOOLS]

    def allowed_tool_names(self, user_id: str) -> set[str] | None:
        """Return the set of tool names this user can access.

        Returns None for admin (all tools allowed) or an empty set for guest.
        """
        tier = self.get_tier(user_id)
        if tier == "admin":
            return None  # no restriction
        if tier == "guest":
            return set()
        return set(USER_TIER_TOOLS)

    def is_admin(self, user_id: str) -> bool:
        return self.get_tier(user_id) == "admin"

    def is_guest(self, user_id: str) -> bool:
        return self.get_tier(user_id) == "guest"
