"""Claude Code headless-mode LLM backend.

Shells out to the local ``claude`` CLI in print mode (``claude -p``) so
generations are billed to the Claude subscription the user is logged into
instead of an API key. Each call runs in an empty temporary working
directory with all tools, MCP servers, and setting sources disabled, so
the model receives exactly the system message and conversation -- the
same context an API call would get.

Select it with a ``claude-code/`` model prefix, e.g.::

    llm:
      models:
        - name: "claude-code/claude-sonnet-5"
"""

import asyncio
import json
import logging
import os
import shutil
import tempfile
from typing import Any, Dict, List

from skydiscover.config import LLMModelConfig
from skydiscover.llm.base import LLMInterface, LLMResponse

logger = logging.getLogger("skydiscover.llm")

# Flags that strip every context channel Claude Code normally injects
# (cwd files, git status, CLAUDE.md, settings, hooks, tools, MCP servers).
_ISOLATION_ARGS = [
    "--tools", "",
    "--setting-sources", "",
    "--strict-mcp-config",
    "--mcp-config", '{"mcpServers":{}}',
    "--no-session-persistence",
]


class ClaudeCodeLLM(LLMInterface):
    """LLM backend that drives headless Claude Code on subscription auth."""

    def __init__(self, model_cfg: LLMModelConfig):
        self.model = model_cfg.name
        self.timeout = model_cfg.timeout or 600
        self.retries = model_cfg.retries if model_cfg.retries is not None else 3
        self.retry_delay = model_cfg.retry_delay if model_cfg.retry_delay is not None else 5
        self.reasoning_effort = model_cfg.reasoning_effort

        self.cli_path = shutil.which("claude")
        if self.cli_path is None:
            raise RuntimeError(
                "claude-code models require the `claude` CLI on PATH. "
                "Install Claude Code and log in (`claude /login`), or set "
                "CLAUDE_CODE_OAUTH_TOKEN from `claude setup-token`."
            )

        # Force subscription auth: an API key in the environment would
        # silently switch the CLI to metered API billing.
        self.env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        self.env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
        self.env["DISABLE_AUTOUPDATER"] = "1"

        if not hasattr(logger, "_initialized_models"):
            logger._initialized_models = set()
        if self.model not in logger._initialized_models:
            logger.info(f"Claude Code (subscription) LLM: {self.model}")
            logger._initialized_models.add(self.model)

    async def generate(
        self, system_message: str, messages: List[Dict[str, Any]], **kwargs
    ) -> LLMResponse:
        if kwargs.get("image_output"):
            raise ValueError("claude-code backend does not support image generation")

        prompt = self._flatten_messages(messages)
        timeout = kwargs.get("timeout", self.timeout)
        attempt = 0
        while True:
            try:
                text = await asyncio.wait_for(
                    self._call_cli(system_message or "", prompt), timeout=timeout
                )
                return LLMResponse(text=text)
            except Exception as e:
                if attempt >= self.retries:
                    raise
                logger.warning(f"Error attempt {attempt + 1}/{self.retries + 1}: {e}, retrying...")
                attempt += 1
                await asyncio.sleep(self.retry_delay)

    @staticmethod
    def _flatten_messages(messages: List[Dict[str, Any]]) -> str:
        """Fold a chat history into a single prompt string."""
        if len(messages) == 1:
            return str(messages[0].get("content", ""))
        return "\n\n".join(f"[{m.get('role', 'user')}]\n{m.get('content', '')}" for m in messages)

    async def _call_cli(self, system_message: str, prompt: str) -> str:
        cmd = [
            self.cli_path,
            "-p",
            "--output-format", "json",
            "--model", self.model,
            "--system-prompt", system_message,
            *_ISOLATION_ARGS,
        ]
        if self.reasoning_effort is not None:
            cmd += ["--effort", self.reasoning_effort]

        # Empty cwd so no project files or git state can leak into context.
        tmpdir = tempfile.mkdtemp(prefix="skydiscover-claude-")
        proc = None
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=tmpdir,
                env=self.env,
            )
            stdout, stderr = await proc.communicate(prompt.encode())
        except asyncio.CancelledError:
            if proc is not None and proc.returncode is None:
                proc.kill()
                await proc.wait()
            raise
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        if proc.returncode != 0:
            raise RuntimeError(
                f"claude CLI exited with code {proc.returncode}: "
                f"{stderr.decode(errors='replace').strip()[:500] or stdout.decode(errors='replace').strip()[:500]}"
            )

        try:
            data = json.loads(stdout.decode())
        except json.JSONDecodeError as e:
            raise RuntimeError(f"claude CLI returned invalid JSON: {e}") from e

        if data.get("is_error"):
            raise RuntimeError(f"claude CLI error: {str(data.get('result', ''))[:500]}")

        usage = data.get("usage", {})
        logger.debug(
            f"Claude Code usage: in={usage.get('input_tokens')} out={usage.get('output_tokens')} "
            f"cache_read={usage.get('cache_read_input_tokens')}"
        )
        return data.get("result", "")
