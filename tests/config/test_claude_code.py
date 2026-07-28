"""Tests for the claude-code (headless Claude Code CLI) backend wiring."""

import asyncio
import json
import os
import stat

import pytest

from skydiscover.config import LLMConfig, LLMModelConfig
from skydiscover.llm.claude_code import ClaudeCodeLLM


class TestClaudeCodeConfigResolution:
    def test_prefix_resolves_to_claude_code_client(self):
        cfg = LLMConfig(models=[LLMModelConfig(name="claude-code/claude-sonnet-5")])
        model = cfg.models[0]
        assert model.name == "claude-sonnet-5"
        assert model.init_client is ClaudeCodeLLM

    def test_bare_claude_prefix_still_uses_anthropic_api(self):
        cfg = LLMConfig(models=[LLMModelConfig(name="claude-sonnet-5")])
        model = cfg.models[0]
        assert model.init_client is None
        assert "api.anthropic.com" in model.api_base

    def test_explicit_init_client_not_overridden(self):
        sentinel = object()
        cfg = LLMConfig(
            models=[LLMModelConfig(name="claude-code/claude-sonnet-5", init_client=sentinel)]
        )
        assert cfg.models[0].init_client is sentinel


class TestFlattenMessages:
    def test_single_message_passes_content_through(self):
        assert ClaudeCodeLLM._flatten_messages([{"role": "user", "content": "hi"}]) == "hi"

    def test_multi_message_includes_role_markers(self):
        out = ClaudeCodeLLM._flatten_messages(
            [
                {"role": "user", "content": "a"},
                {"role": "assistant", "content": "b"},
                {"role": "user", "content": "c"},
            ]
        )
        assert out == "[user]\na\n\n[assistant]\nb\n\n[user]\nc"


@pytest.fixture
def fake_cli(tmp_path):
    """A stand-in `claude` executable that echoes a canned JSON result."""
    script = tmp_path / "claude"
    response = {"is_error": False, "result": "mutated code", "usage": {}}
    script.write_text(f"#!/bin/sh\ncat > /dev/null\necho '{json.dumps(response)}'\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script


class TestGenerate:
    def _make_llm(self, cli_path):
        os.environ["PATH"] = f"{os.path.dirname(cli_path)}:{os.environ['PATH']}"
        return ClaudeCodeLLM(LLMModelConfig(name="claude-sonnet-5", timeout=30, retries=0))

    def test_generate_returns_cli_result(self, fake_cli):
        llm = self._make_llm(str(fake_cli))
        llm.cli_path = str(fake_cli)
        resp = asyncio.run(llm.generate("sys", [{"role": "user", "content": "mutate"}]))
        assert resp.text == "mutated code"

    def test_image_output_rejected(self, fake_cli):
        llm = self._make_llm(str(fake_cli))
        with pytest.raises(ValueError):
            asyncio.run(llm.generate("sys", [], image_output=True))

    def test_cli_error_raises(self, fake_cli):
        fake_cli.write_text('#!/bin/sh\ncat > /dev/null\necho \'{"is_error": true, "result": "limit reached"}\'\n')
        llm = self._make_llm(str(fake_cli))
        llm.cli_path = str(fake_cli)
        with pytest.raises(RuntimeError, match="limit reached"):
            asyncio.run(llm.generate("sys", [{"role": "user", "content": "x"}]))
