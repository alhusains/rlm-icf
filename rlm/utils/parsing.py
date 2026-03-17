"""
Parsing utilities for RLM trjaectories.
"""

import re
from typing import TYPE_CHECKING

from rlm.core.types import REPLResult, RLMIteration

if TYPE_CHECKING:
    from rlm.environments.base_env import BaseEnv


def find_code_blocks(text: str) -> list[str]:
    """
    Find REPL code blocks in text wrapped in triple backticks and return List of content(s).
    Returns None if no code blocks are found.

    Uses a negative lookbehind ``(?<!`)`` so that ````repl`` (4-backtick outer
    documentation fences) are not mistakenly matched and their inner content
    (which starts with the inner ``\\```repl`` header line) executed as code.
    Similarly, a negative lookahead on the closing fence prevents the inner
    ``\\```\\n`` from being matched when it is followed by more backticks.
    """
    # (?<!`) — the opening ``` must NOT be preceded by another backtick
    # (?!`)  — the closing ``` must NOT be followed by another backtick
    pattern = r"(?<!`)```repl[ \t]*\n(.*?)\n```(?!`)"
    results = []

    for match in re.finditer(pattern, text, re.DOTALL):
        code_content = match.group(1).strip()
        results.append(code_content)

    return results


def _extract_balanced_parens(text: str, start: int) -> str | None:
    """Return the content between balanced parentheses.

    *start* must point to the character immediately after the opening ``(``.
    Returns the inner content (without the outer parens) if balanced parens
    are found, or ``None`` if the string ends before they close.
    """
    depth = 1
    pos = start
    while pos < len(text) and depth > 0:
        ch = text[pos]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        pos += 1
    if depth == 0:
        return text[start : pos - 1]
    return None


def find_final_answer(text: str, environment: "BaseEnv | None" = None) -> str | None:
    """
    Find FINAL(...) or FINAL_VAR(...) statement in response and return the final answer string.

    If FINAL_VAR is found and an environment is provided, executes code to retrieve the variable value.
    Returns None if neither pattern is found.

    Uses balanced-parenthesis matching so that content containing ``(`` or ``)``
    characters (e.g. JSON with parenthesised phrases) is not truncated.

    Args:
        text: The response text to parse
        environment: Optional environment to execute code for FINAL_VAR retrieval

    Returns:
        The final answer string, or None if no final answer pattern is found
    """
    # Check for FINAL_VAR pattern first - must be at start of line.
    # Variable names never contain parens, so balanced matching is just extra safety.
    for m in re.finditer(r"^\s*FINAL_VAR\(", text, re.MULTILINE):
        inner = _extract_balanced_parens(text, m.end())
        if inner is None:
            continue
        variable_name = inner.strip().strip('"').strip("'")
        if environment is not None:
            result = environment.execute_code(f"print(FINAL_VAR({variable_name!r}))")
            final_answer = result.stdout.strip()
            if final_answer == "":
                final_answer = result.stderr.strip() or ""
            return final_answer
        return None

    # Check for FINAL pattern - must be at start of line.
    # The content may be raw JSON or prose containing parentheses, so we MUST
    # use balanced matching here — (.*?) would stop at the first ')' inside
    # the content (e.g. a city name like "Montreal)".
    for m in re.finditer(r"^\s*FINAL\(", text, re.MULTILINE):
        inner = _extract_balanced_parens(text, m.end())
        if inner is not None:
            return inner.strip()

    return None


def format_iteration(
    iteration: RLMIteration, max_character_length: int = 20000
) -> list[dict[str, str]]:
    """
    Format an RLM iteration (including all code blocks) to append to the message history for
    the prompt of the LM in the next iteration. We also truncate code execution results
    that exceed the max_character_length.

    Args:
        iteration: The iteration to format
        max_character_length: The maximum character length of the result

    Returns:
        A list of messages to add to the next prompt
    """
    messages = [{"role": "assistant", "content": iteration.response}]

    for code_block in iteration.code_blocks:
        code = code_block.code
        result = code_block.result
        result = format_execution_result(result)
        if len(result) > max_character_length:
            result = (
                result[:max_character_length]
                + f"... + [{len(result) - max_character_length} chars...]"
            )

        execution_message = {
            "role": "user",
            "content": f"Code executed:\n```python\n{code}\n```\n\nREPL output:\n{result}",
        }
        messages.append(execution_message)
    return messages


################
# TODO: Remove and refactor these soon
################


def format_execution_result(result: REPLResult) -> str:
    """
    Format the execution result as a string for display.

    Args:
        result: The REPLResult object to format.
    """
    result_parts = []

    if result.stdout:
        result_parts.append(f"\n{result.stdout}")

    if result.stderr:
        result_parts.append(f"\n{result.stderr}")

    # Show some key variables (excluding internal ones)
    important_vars = {}
    for key, value in result.locals.items():
        if not key.startswith("_") and key not in [
            "__builtins__",
            "__name__",
            "__doc__",
        ]:
            # Only show simple types or short representations
            if isinstance(value, (str, int, float, bool, list, dict, tuple)):
                important_vars[key] = ""

    if important_vars:
        result_parts.append(f"REPL variables: {list(important_vars.keys())}\n")

    return "\n\n".join(result_parts) if result_parts else "No output"


def check_for_final_answer(response: str, repl_env, logger) -> str | None:
    """Check if response contains a final answer."""
    # Use the new find_final_answer function which handles both FINAL and FINAL_VAR
    return find_final_answer(response, environment=repl_env)


def convert_context_for_repl(context):
    """
    Convert REPL context to either some
    """
    if isinstance(context, dict):
        context_data = context
        context_str = None
    elif isinstance(context, str):
        context_data = None
        context_str = context
    elif isinstance(context, list):
        if len(context) > 0 and isinstance(context[0], dict):
            if "content" in context[0]:
                context_data = [msg.get("content", "") for msg in context]
            else:
                context_data = context
            context_str = None
        else:
            context_data = context
            context_str = None
    else:
        context_data = context
        context_str = None

    return context_data, context_str
