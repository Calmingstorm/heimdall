from __future__ import annotations

import asyncio
import signal
import sys
from pathlib import Path

from .config import load_config
from .discord import LokiBot
from .health import HealthServer
from .logging import setup_logging, get_logger


def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yml"

    if not Path(config_path).exists():
        print(f"Config file not found: {config_path}")
        sys.exit(1)

    config = load_config(config_path)
    setup_logging(level=config.logging.level, log_dir=config.logging.directory)
    log = get_logger("main")

    log.info("Starting Loki")

    health = HealthServer(webhook_config=config.webhook)
    bot = LokiBot(config)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def run() -> None:
        await health.start()

        # Wire webhook send_message callback to Discord bot
        async def _webhook_send(channel_id: str, text: str) -> None:
            channel = bot.get_channel(int(channel_id))
            if channel:
                await channel.send(text)
            else:
                log.warning("Webhook: channel %s not found", channel_id)

        health.set_send_message(_webhook_send)
        health.set_trigger_callback(bot.scheduler.fire_triggers)

        def handle_signal() -> None:
            log.info("Shutdown signal received")
            loop.create_task(shutdown())

        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, handle_signal)

        try:
            health.set_ready(True)
            log.info("Connecting to Discord...")
            await bot.start(config.discord.token)
        except Exception as e:
            log.error("Fatal error: %s", e, exc_info=True)
            await shutdown()

    async def shutdown() -> None:
        log.info("Shutting down...")
        if bot.voice_manager:
            await bot.voice_manager.shutdown()
        if bot.browser_manager:
            await bot.browser_manager.shutdown()
        await bot.scheduler.stop()
        bot.sessions.save_all()
        await bot.classifier.close()
        await bot.close()
        await health.stop()
        loop.stop()

    try:
        loop.run_until_complete(run())
    except (KeyboardInterrupt, SystemExit):
        loop.run_until_complete(shutdown())
    finally:
        loop.close()
        log.info("Loki stopped")


if __name__ == "__main__":
    main()
