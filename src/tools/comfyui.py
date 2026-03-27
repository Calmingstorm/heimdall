"""ComfyUI HTTP client for text-to-image generation."""

from __future__ import annotations

import asyncio
import json
import uuid

import aiohttp

from ..logging import get_logger

log = get_logger("comfyui")

# Default txt2img workflow template.
# The {prompt}, {negative}, {width}, {height} placeholders are filled at runtime.
_DEFAULT_WORKFLOW = {
    "3": {
        "class_type": "KSampler",
        "inputs": {
            "seed": 0,
            "steps": 20,
            "cfg": 7.0,
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": 1.0,
            "model": ["4", 0],
            "positive": ["6", 0],
            "negative": ["7", 0],
            "latent_image": ["5", 0],
        },
    },
    "4": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": "realisticVisionV60B1_v60B1VAE.safetensors"},
    },
    "5": {
        "class_type": "EmptyLatentImage",
        "inputs": {"width": 1024, "height": 1024, "batch_size": 1},
    },
    "6": {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": "", "clip": ["4", 1]},
    },
    "7": {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": "", "clip": ["4", 1]},
    },
    "8": {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
    },
    "9": {
        "class_type": "SaveImage",
        "inputs": {"filename_prefix": "heimdall", "images": ["8", 0]},
    },
}


class ComfyUIClient:
    """Async HTTP client for ComfyUI image generation."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    async def generate(
        self,
        prompt: str,
        negative: str = "",
        width: int = 1024,
        height: int = 1024,
        model: str = "",
    ) -> bytes | None:
        """Generate an image from a text prompt.

        Returns PNG image bytes on success, None on failure.
        """
        import copy
        import random

        workflow = copy.deepcopy(_DEFAULT_WORKFLOW)

        # Use specified model or auto-detect from available checkpoints
        ckpt = model if model else workflow["4"]["inputs"]["ckpt_name"]
        resolved = await self._resolve_checkpoint(ckpt)
        if not resolved:
            log.warning("No checkpoints available on ComfyUI")
            return None
        workflow["4"]["inputs"]["ckpt_name"] = resolved

        workflow["5"]["inputs"]["width"] = width
        workflow["5"]["inputs"]["height"] = height
        workflow["6"]["inputs"]["text"] = prompt
        workflow["7"]["inputs"]["text"] = negative
        workflow["3"]["inputs"]["seed"] = random.randint(0, 2**32 - 1)

        client_id = uuid.uuid4().hex[:8]
        payload = {"prompt": workflow, "client_id": client_id}

        timeout = aiohttp.ClientTimeout(total=120)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                # Queue the prompt
                async with session.post(
                    f"{self.base_url}/prompt", json=payload
                ) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        log.warning("ComfyUI /prompt failed (%s): %s", resp.status, body[:200])
                        return None
                    data = await resp.json()
                    prompt_id = data.get("prompt_id")
                    if not prompt_id:
                        log.warning("ComfyUI returned no prompt_id")
                        return None
                    # Validate prompt_id to prevent path traversal in URL
                    if not isinstance(prompt_id, str) or not all(
                        c.isalnum() or c in "-_" for c in prompt_id
                    ) or len(prompt_id) > 100:
                        log.warning("ComfyUI returned suspicious prompt_id: %s", str(prompt_id)[:50])
                        return None

                # Poll /history until complete
                image_filename = await self._poll_history(session, prompt_id)
                if not image_filename:
                    return None

                # Download the image
                async with session.get(
                    f"{self.base_url}/view",
                    params={"filename": image_filename},
                ) as resp:
                    if resp.status != 200:
                        log.warning("ComfyUI /view failed (%s)", resp.status)
                        return None
                    return await resp.read()

        except asyncio.TimeoutError:
            log.warning("ComfyUI generation timed out (120s)")
            return None
        except aiohttp.ClientError as e:
            log.warning("ComfyUI connection error: %s", e)
            return None
        except Exception as e:
            log.warning("ComfyUI unexpected error: %s", e)
            return None

    async def _resolve_checkpoint(self, preferred: str) -> str | None:
        """Check if preferred checkpoint exists, fall back to first available."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/object_info/CheckpointLoaderSimple"
                ) as resp:
                    if resp.status != 200:
                        return preferred  # Can't check, try anyway
                    data = await resp.json()
                    available = data["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0]
                    if preferred in available:
                        return preferred
                    if available:
                        chosen = available[0]
                        log.info("Checkpoint '%s' not found, using '%s'", preferred, chosen)
                        return chosen
                    return None
        except Exception:
            return preferred  # Can't check, try anyway

    async def _poll_history(
        self, session: aiohttp.ClientSession, prompt_id: str
    ) -> str | None:
        """Poll /history/{prompt_id} until the job completes. Returns filename."""
        for _ in range(120):  # 120 * 1s = 2 min max
            await asyncio.sleep(1)
            try:
                async with session.get(
                    f"{self.base_url}/history/{prompt_id}"
                ) as resp:
                    if resp.status != 200:
                        continue
                    data = await resp.json()

                entry = data.get(prompt_id)
                if not entry:
                    continue

                outputs = entry.get("outputs", {})
                for node_id, node_output in outputs.items():
                    images = node_output.get("images", [])
                    if images:
                        return images[0].get("filename")

            except Exception:
                continue

        log.warning("ComfyUI polling timed out for prompt %s", prompt_id)
        return None
