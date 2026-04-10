"""IntentAgent — LLM-powered natural language conversation to determine output layout.

If --layout was already supplied on the CLI, this agent is a no-op.

Otherwise it starts a brief chat loop with the user:
  1. Show a welcome message and capture the user's free-text intent.
  2. Send the conversation history to the LLM.
  3. If the LLM needs clarification, it asks a follow-up question (plain text).
  4. When the LLM has enough information, it returns a JSON decision object.
  5. Write the resolved layout_instruction (or None for default template) to context.

LLM decision schema:
  {"done": true, "use_template": true}
  {"done": true, "use_template": false, "columns": "Date, Facility, Physician, Summary, Ref"}
"""

from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from .base import BaseAgent
from ..pipeline import PipelineContext

AI_BUILDERS_BASE_URL = "https://space.ai-builders.com/backend/v1"

console = Console()

SYSTEM_PROMPT = """You are a helpful assistant configuring the output format for a medical case summary report.

The report supports two modes:
1. DEFAULT template — fills a pre-built Word template with fixed columns: Date, Provider, Reason, Ref.
2. CUSTOM layout — user specifies their own column names for the timeline table.

Your job: hold a short conversation to understand what the user wants, then signal completion.

When you have enough information, respond with ONLY a JSON object on a single line — no explanation, no markdown:
  {"done": true, "use_template": true}
  OR
  {"done": true, "use_template": false, "columns": "Date, Facility, Physician, Summary, Ref"}

If you need more information first, reply with a friendly follow-up question as plain text (no JSON).

Rules:
- "default", "template", "standard", "keep it as is" → use_template: true
- Any mention of specific columns or custom format → use_template: false, list the column names
- If user says they are unsure or asks what options exist, briefly explain the two modes then ask
- Keep the conversation to 1–2 exchanges maximum; do not over-ask
- Respond in the same language the user writes in
""".strip()


class IntentAgent(BaseAgent):
    name = "Intent Agent"

    def _run(self, context: PipelineContext) -> PipelineContext:
        if context.layout_instruction is not None:
            console.print(
                f"[green]Layout already set:[/green] {context.layout_instruction}"
            )
            return context

        console.print(Panel(
            "I'll help you choose the output format for this medical summary.\n"
            "[dim]Type your preference in any language — "
            "e.g. 'use default', '我要自訂欄位', or 'custom: Date, Facility, Summary'[/dim]",
            title="[bold cyan]Output Format Setup[/bold cyan]",
            border_style="cyan",
        ))

        history: list[dict[str, str]] = []
        decision = self._converse(history, context.model)

        if decision.get("use_template"):
            context.layout_instruction = None
            console.print("[green]✓ Using default Word template layout.[/green]")
        else:
            columns = decision.get("columns", "Date, Provider, Reason, Ref")
            context.layout_instruction = columns
            console.print(f"[green]✓ Custom layout set:[/green] {columns}")

        return context

    # ------------------------------------------------------------------
    # Conversation loop
    # ------------------------------------------------------------------

    def _converse(
        self,
        history: list[dict[str, str]],
        model: str,
        max_turns: int = 4,
    ) -> dict[str, Any]:
        """Run the chat loop until the LLM signals done or max_turns is reached."""
        for _ in range(max_turns):
            user_input = Prompt.ask("[bold]You[/bold]").strip()
            if not user_input:
                continue

            history.append({"role": "user", "content": user_input})
            reply = _call_llm(SYSTEM_PROMPT, history, model)
            history.append({"role": "assistant", "content": reply})

            decision = _try_parse_decision(reply)
            if decision is not None:
                return decision

            console.print(f"[cyan]Assistant:[/cyan] {reply}\n")

        # Fallback: use template if we ran out of turns
        console.print("[yellow]No clear preference detected — using default template.[/yellow]")
        return {"done": True, "use_template": True}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _call_llm(system: str, history: list[dict[str, str]], model: str) -> str:
    """Call the AI Builders API with the conversation history."""
    token = os.environ.get("AI_BUILDER_TOKEN")
    if not token:
        raise EnvironmentError(
            "AI_BUILDER_TOKEN is not set. Copy .env.example to .env and add your token."
        )
    client = OpenAI(base_url=AI_BUILDERS_BASE_URL, api_key=token)
    messages = [{"role": "system", "content": system}, *history]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def _try_parse_decision(text: str) -> dict[str, Any] | None:
    """Return the parsed decision dict if *text* is a valid done-signal JSON, else None."""
    text = text.strip()
    # Only attempt parsing if text looks like a JSON object
    if not (text.startswith("{") and text.endswith("}")):
        return None
    try:
        data = json.loads(text)
        if data.get("done") is True:
            return data
    except (json.JSONDecodeError, AttributeError):
        pass
    return None
