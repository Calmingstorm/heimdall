from __future__ import annotations

import asyncio

import discord

from ..logging import get_logger

log = get_logger("approval")


class ApprovalView(discord.ui.View):
    """Button-based approval view for tool execution."""

    def __init__(self, allowed_users: list[str], timeout: int = 60) -> None:
        super().__init__(timeout=timeout)
        self._allowed_users = allowed_users
        self._event = asyncio.Event()
        self._approved: bool = False

    @discord.ui.button(label="Approve", style=discord.ButtonStyle.green, emoji="\u2705")
    async def approve_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        if str(interaction.user.id) not in self._allowed_users:
            await interaction.response.send_message("You are not authorized to approve this action.", ephemeral=True)
            return
        self._approved = True
        self._event.set()
        self.stop()
        await interaction.response.defer()

    @discord.ui.button(label="Deny", style=discord.ButtonStyle.red, emoji="\u274c")
    async def deny_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        if str(interaction.user.id) not in self._allowed_users:
            await interaction.response.send_message("You are not authorized to deny this action.", ephemeral=True)
            return
        self._approved = False
        self._event.set()
        self.stop()
        await interaction.response.defer()

    async def wait_for_result(self) -> bool:
        """Wait for a button press or timeout. Returns True if approved."""
        try:
            await asyncio.wait_for(self._event.wait(), timeout=self.timeout)
        except asyncio.TimeoutError:
            self._approved = False
        return self._approved

    async def on_timeout(self) -> None:
        self._event.set()

    def disable_all(self) -> None:
        for item in self.children:
            if isinstance(item, discord.ui.Button):
                item.disabled = True


async def request_approval(
    bot: discord.Client,
    channel: discord.abc.Messageable,
    tool_name: str,
    tool_input: dict,
    allowed_users: list[str],
    timeout: int = 60,
) -> bool:
    """Send an approval embed with buttons and wait for a response from an allowed user."""
    max_val_len = 200
    lines = []
    for k, v in tool_input.items():
        v_str = str(v)
        if len(v_str) > max_val_len:
            v_str = v_str[:max_val_len] + f"... ({len(str(v))} chars)"
        lines.append(f"**{k}:** `{v_str}`")
    params = "\n".join(lines)

    desc = f"The following action requires your approval:\n\n{params}"
    if len(desc) > 4000:
        desc = desc[:4000] + "\n\n*(truncated)*"

    embed = discord.Embed(
        title=f"Approval Required: {tool_name}",
        description=desc,
        color=discord.Color.orange(),
    )
    embed.set_footer(text=f"Click a button within {timeout}s to respond")

    view = ApprovalView(allowed_users=allowed_users, timeout=timeout)
    msg = await channel.send(embed=embed, view=view)

    approved = await view.wait_for_result()

    # Update embed and disable buttons
    view.disable_all()
    if approved:
        embed.color = discord.Color.green()
        embed.set_footer(text="Approved")
        log.info("Action %s approved", tool_name)
    else:
        embed.color = discord.Color.red()
        footer = "Denied" if view._event.is_set() and not approved else "Timed out \u2014 action cancelled"
        embed.set_footer(text=footer)
        log.info("Action %s %s", tool_name, "denied" if view._event.is_set() else "timed out")

    try:
        await msg.edit(embed=embed, view=view)
    except discord.HTTPException:
        pass

    return approved
