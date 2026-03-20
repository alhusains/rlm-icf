"""
Debug logger for ICF pipeline runs.

Writes a compact JSONL trace of every RLM iteration so you can replay
exactly what code the model wrote, what it found, and how it proceeded —
without storing the full protocol text that would make the file huge.

What IS saved per iteration
----------------------------
  section_id / heading / sub_section  – which variable is being extracted
  iteration                           – iteration number within this section
  timestamp / iteration_time_s        – timing
  response                            – LLM's full text (reasoning + code it wrote)
  code_blocks[].code                  – exact Python executed in the REPL
  code_blocks[].stdout / .stderr      – REPL output (truncated to MAX_OUTPUT_CHARS)
  final_answer                        – the FINAL_VAR value if this was the last step

What is NOT saved
-----------------
  prompt   – the accumulated message history; grows with every iteration and
             ends up containing printed protocol chunks from earlier steps
  locals   – the REPL variable namespace; includes context_0 (full protocol)

Output
------
  <log_dir>/rlm_trace_<timestamp>_<run_id>.jsonl
  One JSON object per line, easy to grep / filter by section_id.
"""

import json
import os
import uuid
from datetime import datetime

from rlm.core.types import RLMIteration, RLMMetadata

# Truncate REPL stdout/stderr beyond this many characters per iteration.
# 8 000 chars ≈ a generous page of output; more than enough to see search
# results while preventing the file from blowing up when the LLM prints
# large slices of the protocol.
_MAX_OUTPUT_CHARS = 8_000


class ICFDebugLogger:
    """Logs per-iteration RLM trace to a JSONL file.

    Strips protocol-sized blobs (message history, REPL locals) and
    truncates large stdout/stderr to keep files manageable.

    Usage::

        logger = ICFDebugLogger(log_dir="output/debug_logs")
        logger.set_section("3", "Why Is This Study Being Done", "Overview")
        # pass logger to ExtractionEngine, which passes it to RLM(logger=...)
    """

    def __init__(self, log_dir: str, run_id: str | None = None):
        os.makedirs(log_dir, exist_ok=True)
        if run_id is None:
            run_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_file_path = os.path.join(log_dir, f"rlm_trace_{timestamp}_{run_id}.jsonl")

        self._iteration_count = 0  # global across all sections (used by RLM internals)

        # Per-section tracking (set by ExtractionEngine before each extraction)
        self._section_id: str = ""
        self._section_heading: str = ""
        self._section_sub: str = ""
        self._section_iter: int = 0

    # ------------------------------------------------------------------
    # Section context — call before starting each new variable
    # ------------------------------------------------------------------

    def set_section(self, section_id: str, heading: str, sub_section: str = "") -> None:
        """Register which ICF variable is about to be extracted."""
        self._section_id = section_id
        self._section_heading = heading
        self._section_sub = sub_section
        self._section_iter = 0

    # ------------------------------------------------------------------
    # RLMLogger-compatible interface (duck-typed by RLM)
    # ------------------------------------------------------------------

    def log_metadata(self, metadata: RLMMetadata) -> None:
        """Log RLM configuration metadata (once per section, tagged with section_id)."""
        entry = {
            "type": "metadata",
            "section_id": self._section_id,
            "heading": self._section_heading,
            "timestamp": datetime.now().isoformat(),
            **metadata.to_dict(),
        }
        self._write(entry)

    def log(self, iteration: RLMIteration) -> None:
        """Log a single RLM iteration, stripping large fields."""
        self._iteration_count += 1
        self._section_iter += 1

        code_blocks = []
        for cb in iteration.code_blocks:
            stdout = cb.result.stdout or ""
            stderr = cb.result.stderr or ""
            if len(stdout) > _MAX_OUTPUT_CHARS:
                truncated = len(stdout) - _MAX_OUTPUT_CHARS
                stdout = stdout[:_MAX_OUTPUT_CHARS] + f"\n...[{truncated} chars truncated]"
            if len(stderr) > _MAX_OUTPUT_CHARS:
                truncated = len(stderr) - _MAX_OUTPUT_CHARS
                stderr = stderr[:_MAX_OUTPUT_CHARS] + f"\n...[{truncated} chars truncated]"
            code_blocks.append(
                {
                    "code": cb.code,
                    "stdout": stdout,
                    "stderr": stderr,
                    "execution_time_s": cb.result.execution_time,
                }
            )

        entry = {
            "type": "iteration",
            "section_id": self._section_id,
            "heading": self._section_heading,
            "sub_section": self._section_sub,
            "section_iteration": self._section_iter,
            "global_iteration": self._iteration_count,
            "timestamp": datetime.now().isoformat(),
            "iteration_time_s": iteration.iteration_time,
            # response = LLM reasoning text + the ```repl blocks it wrote
            "response": iteration.response,
            "code_blocks": code_blocks,
            "final_answer": iteration.final_answer,
        }
        self._write(entry)

    @property
    def iteration_count(self) -> int:
        return self._iteration_count

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _write(self, entry: dict) -> None:
        with open(self.log_file_path, "a", encoding="utf-8") as f:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")
