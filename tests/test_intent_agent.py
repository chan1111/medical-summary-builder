"""Tests for the LLM-powered IntentAgent."""

from __future__ import annotations

import json
from unittest.mock import call, patch

import pytest

from medical_summary_builder.agents.intent_agent import (
    IntentAgent,
    _call_llm,
    _try_parse_decision,
)
from medical_summary_builder.pipeline import PipelineContext


# ---------------------------------------------------------------------------
# _try_parse_decision unit tests
# ---------------------------------------------------------------------------

class TestTryParseDecision:
    def test_returns_none_for_plain_text(self):
        assert _try_parse_decision("Would you like to use the default template?") is None

    def test_returns_none_for_json_without_done(self):
        assert _try_parse_decision('{"columns": "Date, Provider"}') is None

    def test_returns_none_for_done_false(self):
        assert _try_parse_decision('{"done": false}') is None

    def test_parses_use_template_true(self):
        result = _try_parse_decision('{"done": true, "use_template": true}')
        assert result == {"done": True, "use_template": True}

    def test_parses_custom_columns(self):
        payload = '{"done": true, "use_template": false, "columns": "Date, Facility, Ref"}'
        result = _try_parse_decision(payload)
        assert result is not None
        assert result["use_template"] is False
        assert result["columns"] == "Date, Facility, Ref"

    def test_ignores_surrounding_whitespace(self):
        payload = '  {"done": true, "use_template": true}  '
        result = _try_parse_decision(payload)
        assert result is not None

    def test_returns_none_for_empty_string(self):
        assert _try_parse_decision("") is None

    def test_returns_none_for_invalid_json(self):
        assert _try_parse_decision("{bad json}") is None


# ---------------------------------------------------------------------------
# _call_llm unit tests
# ---------------------------------------------------------------------------

class TestCallLlm:
    def test_raises_when_token_missing(self, monkeypatch):
        monkeypatch.delenv("AI_BUILDER_TOKEN", raising=False)
        with pytest.raises(EnvironmentError, match="AI_BUILDER_TOKEN"):
            _call_llm("sys", [{"role": "user", "content": "hi"}], "grok-4-fast")

    def test_returns_stripped_content(self, monkeypatch):
        monkeypatch.setenv("AI_BUILDER_TOKEN", "test-token")
        from unittest.mock import MagicMock
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = "  hello  "
        with patch("medical_summary_builder.agents.intent_agent.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.create.return_value = mock_resp
            result = _call_llm("sys", [{"role": "user", "content": "hi"}], "grok-4-fast")
        assert result == "hello"

    def test_passes_system_and_history_as_messages(self, monkeypatch):
        monkeypatch.setenv("AI_BUILDER_TOKEN", "test-token")
        from unittest.mock import MagicMock
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = "reply"
        captured = {}
        with patch("medical_summary_builder.agents.intent_agent.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.create.side_effect = (
                lambda **kw: (captured.update({"messages": kw["messages"]}) or mock_resp)
            )
            _call_llm("SYSTEM", [{"role": "user", "content": "USER MSG"}], "grok-4-fast")
        assert captured["messages"][0] == {"role": "system", "content": "SYSTEM"}
        assert captured["messages"][1] == {"role": "user", "content": "USER MSG"}


# ---------------------------------------------------------------------------
# IntentAgent — CLI preset (no-op path)
# ---------------------------------------------------------------------------

class TestIntentAgentPreset:
    def test_skips_llm_when_layout_already_set(self, base_context: PipelineContext, monkeypatch):
        base_context.layout_instruction = "Date, Provider, Reason, Ref"
        with patch("medical_summary_builder.agents.intent_agent._call_llm") as mock_llm:
            IntentAgent().run(base_context)
        mock_llm.assert_not_called()

    def test_returns_same_context_when_preset(self, base_context: PipelineContext):
        base_context.layout_instruction = "Date, Facility, Summary"
        result = IntentAgent().run(base_context)
        assert result is base_context
        assert result.layout_instruction == "Date, Facility, Summary"


# ---------------------------------------------------------------------------
# IntentAgent — LLM decides to use default template
# ---------------------------------------------------------------------------

class TestIntentAgentUsesTemplate:
    def test_sets_layout_instruction_to_none(self, base_context: PipelineContext, monkeypatch):
        monkeypatch.setenv("AI_BUILDER_TOKEN", "test-token")
        decision = json.dumps({"done": True, "use_template": True})
        with (
            patch("medical_summary_builder.agents.intent_agent.Prompt.ask", return_value="use default"),
            patch("medical_summary_builder.agents.intent_agent._call_llm", return_value=decision),
        ):
            result = IntentAgent().run(base_context)
        assert result.layout_instruction is None

    def test_single_turn_conversation(self, base_context: PipelineContext, monkeypatch):
        monkeypatch.setenv("AI_BUILDER_TOKEN", "test-token")
        decision = json.dumps({"done": True, "use_template": True})
        with (
            patch("medical_summary_builder.agents.intent_agent.Prompt.ask", return_value="default please"),
            patch("medical_summary_builder.agents.intent_agent._call_llm", return_value=decision) as mock_llm,
        ):
            IntentAgent().run(base_context)
        assert mock_llm.call_count == 1


# ---------------------------------------------------------------------------
# IntentAgent — LLM decides to use custom layout
# ---------------------------------------------------------------------------

class TestIntentAgentCustomLayout:
    def test_sets_custom_columns(self, base_context: PipelineContext, monkeypatch):
        monkeypatch.setenv("AI_BUILDER_TOKEN", "test-token")
        decision = json.dumps({
            "done": True,
            "use_template": False,
            "columns": "Date, Facility, Physician, Summary, Ref",
        })
        with (
            patch("medical_summary_builder.agents.intent_agent.Prompt.ask", return_value="5 columns: date facility physician summary ref"),
            patch("medical_summary_builder.agents.intent_agent._call_llm", return_value=decision),
        ):
            result = IntentAgent().run(base_context)
        assert result.layout_instruction == "Date, Facility, Physician, Summary, Ref"

    def test_layout_instruction_comes_from_llm(self, base_context: PipelineContext, monkeypatch):
        """layout_instruction should reflect the columns the LLM extracted, not raw user input."""
        monkeypatch.setenv("AI_BUILDER_TOKEN", "test-token")
        decision = json.dumps({
            "done": True,
            "use_template": False,
            "columns": "日期, 醫院, 摘要",
        })
        with (
            patch("medical_summary_builder.agents.intent_agent.Prompt.ask", return_value="我要三欄：日期醫院摘要"),
            patch("medical_summary_builder.agents.intent_agent._call_llm", return_value=decision),
        ):
            result = IntentAgent().run(base_context)
        assert result.layout_instruction == "日期, 醫院, 摘要"


# ---------------------------------------------------------------------------
# IntentAgent — multi-turn conversation
# ---------------------------------------------------------------------------

class TestIntentAgentMultiTurn:
    def test_llm_asks_followup_then_decides(self, base_context: PipelineContext, monkeypatch):
        """LLM first returns a clarifying question, then a decision on the second turn."""
        monkeypatch.setenv("AI_BUILDER_TOKEN", "test-token")
        decision = json.dumps({"done": True, "use_template": True})

        llm_replies = iter([
            "Do you want a default template or a custom column layout?",
            decision,
        ])
        with (
            patch("medical_summary_builder.agents.intent_agent.Prompt.ask", return_value="yes default"),
            patch("medical_summary_builder.agents.intent_agent._call_llm", side_effect=lambda *a, **kw: next(llm_replies)) as mock_llm,
        ):
            result = IntentAgent().run(base_context)

        assert mock_llm.call_count == 2
        assert result.layout_instruction is None

    def test_history_grows_each_turn(self, base_context: PipelineContext, monkeypatch):
        """Each LLM call should receive an expanded message history."""
        monkeypatch.setenv("AI_BUILDER_TOKEN", "test-token")
        decision = json.dumps({"done": True, "use_template": True})
        captured_histories: list[list] = []

        def fake_llm(system, history, model):
            captured_histories.append(list(history))
            if len(history) >= 3:
                return decision
            return "Can you clarify?"

        with (
            patch("medical_summary_builder.agents.intent_agent.Prompt.ask", return_value="ok"),
            patch("medical_summary_builder.agents.intent_agent._call_llm", side_effect=fake_llm),
        ):
            IntentAgent().run(base_context)

        # Second call should have more messages than the first
        assert len(captured_histories[1]) > len(captured_histories[0])


# ---------------------------------------------------------------------------
# IntentAgent — fallback when max turns reached
# ---------------------------------------------------------------------------

class TestIntentAgentFallback:
    def test_falls_back_to_template_after_max_turns(self, base_context: PipelineContext, monkeypatch):
        """If the LLM never returns a done signal, default to template (layout_instruction=None)."""
        monkeypatch.setenv("AI_BUILDER_TOKEN", "test-token")
        with (
            patch("medical_summary_builder.agents.intent_agent.Prompt.ask", return_value="hmm"),
            patch(
                "medical_summary_builder.agents.intent_agent._call_llm",
                return_value="I'm not sure what you want.",
            ),
        ):
            result = IntentAgent().run(base_context)
        assert result.layout_instruction is None
