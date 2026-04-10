"""BaseAgent abstract class for all pipeline agents."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod

from rich.console import Console

from ..pipeline import PipelineContext

console = Console()


class BaseAgent(ABC):
    """Abstract base class that every pipeline agent must implement."""

    name: str = "Agent"

    def run(self, context: PipelineContext) -> PipelineContext:
        """Execute the agent, wrapping work with timing and console output."""
        logger = logging.getLogger(self.__class__.__module__)
        console.rule(f"[bold cyan]{self.name}")
        t0 = time.perf_counter()
        logger.debug("%s started", self.name)

        try:
            context = self._run(context)
        except Exception:
            elapsed = time.perf_counter() - t0
            logger.exception("%s raised an exception after %.2fs", self.name, elapsed)
            raise

        elapsed = time.perf_counter() - t0
        logger.info("%s completed in %.2fs", self.name, elapsed)
        console.print(f"[dim]{self.name} completed in {elapsed:.2f}s[/dim]\n")
        return context

    @abstractmethod
    def _run(self, context: PipelineContext) -> PipelineContext:
        """Agent-specific logic. Subclasses implement this method."""
        ...
