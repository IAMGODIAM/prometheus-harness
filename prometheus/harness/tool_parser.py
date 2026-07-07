"""Tool Call Parser — Model-agnostic parsing for four tool-call formats.

Supports the four parser variants required for 2026 open-weight models:
1. OpenAI JSON: DeepSeek, MiniMax, Doubao, most providers
2. Hermes-XML: <tool_call>{json}</tool_call> (Qwen3 default, Qwen3-Coder XML variant)
3. GLM XML: <tool_call>{name}<arg_key/><arg_value/></tool_call>
4. Kimi special tokens: <|tool_calls_section_begin|>...<|tool_calls_section_end|>

Follows the Hermes Agent pattern of model-agnostic tool calling.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ToolCallFormat(str, Enum):
    """Supported tool-call output formats."""
    OPENAI_JSON = "openai_json"
    HERMES_XML = "hermes_xml"
    GLM_XML = "glm_xml"
    KIMI_SPECIAL = "kimi_special"
    AUTO = "auto"  # Auto-detect from response


@dataclass
class ToolCall:
    """Parsed tool call."""
    id: str = ""
    name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    raw: str = ""

    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI-compatible format."""
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments),
            },
        }


class ToolCallParser:
    """Model-agnostic tool-call parser.

    Detects and parses tool calls from LLM responses regardless of the
    output format used by the model. Supports auto-detection or explicit
    format specification.
    """

    def __init__(self, default_format: ToolCallFormat = ToolCallFormat.AUTO):
        self.default_format = default_format
        self._parsers = {
            ToolCallFormat.OPENAI_JSON: self._parse_openai_json,
            ToolCallFormat.HERMES_XML: self._parse_hermes_xml,
            ToolCallFormat.GLM_XML: self._parse_glm_xml,
            ToolCallFormat.KIMI_SPECIAL: self._parse_kimi_special,
        }

    def parse(
        self,
        response: dict[str, Any] | str,
        format: ToolCallFormat | None = None,
    ) -> list[ToolCall]:
        """Parse tool calls from an LLM response.

        Args:
            response: LLM response (dict with 'content' and/or 'tool_calls', or raw string).
            format: Explicit format to use. None = auto-detect.

        Returns:
            List of parsed ToolCall objects.
        """
        fmt = format or self.default_format

        # If response is already structured (OpenAI API format)
        if isinstance(response, dict):
            # Check for native tool_calls field (OpenAI/compatible APIs)
            if "tool_calls" in response and response["tool_calls"]:
                return self._parse_native_tool_calls(response["tool_calls"])

            # Check message.tool_calls
            if "message" in response:
                msg = response["message"]
                if "tool_calls" in msg and msg["tool_calls"]:
                    return self._parse_native_tool_calls(msg["tool_calls"])

            # Fall through to content parsing
            content = response.get("content", "") or response.get("message", {}).get("content", "")
        else:
            content = response

        if not content:
            return []

        # Auto-detect format
        if fmt == ToolCallFormat.AUTO:
            fmt = self._detect_format(content)

        if fmt in self._parsers:
            return self._parsers[fmt](content)

        return []

    def _detect_format(self, content: str) -> ToolCallFormat:
        """Auto-detect the tool-call format from response content."""
        if "<|tool_calls_section_begin|>" in content:
            return ToolCallFormat.KIMI_SPECIAL
        if "<tool_call>" in content:
            # Distinguish Hermes from GLM
            # GLM uses <arg_key/> style, Hermes uses JSON
            if re.search(r'<[a-z_]+/>', content):
                return ToolCallFormat.GLM_XML
            return ToolCallFormat.HERMES_XML
        # Default to OpenAI JSON (look for function call patterns)
        return ToolCallFormat.OPENAI_JSON

    def _parse_native_tool_calls(self, tool_calls: list[dict[str, Any]]) -> list[ToolCall]:
        """Parse native OpenAI-format tool_calls array."""
        results = []
        for tc in tool_calls:
            func = tc.get("function", {})
            args_str = func.get("arguments", "{}")
            try:
                args = json.loads(args_str) if isinstance(args_str, str) else args_str
            except json.JSONDecodeError:
                args = {"raw": args_str}

            results.append(ToolCall(
                id=tc.get("id", f"call_{len(results)}"),
                name=func.get("name", ""),
                arguments=args,
                raw=json.dumps(tc),
            ))
        return results

    def _parse_openai_json(self, content: str) -> list[ToolCall]:
        """Parse OpenAI JSON format tool calls from content.

        Looks for JSON objects with 'name' and 'arguments' or 'parameters' fields.
        """
        results = []

        # Try to find JSON blocks in the content
        json_pattern = r'\{[^{}]*"(?:name|function)"[^{}]*\}'
        matches = re.finditer(json_pattern, content, re.DOTALL)

        for match in matches:
            try:
                obj = json.loads(match.group())
                name = obj.get("name") or obj.get("function", {}).get("name", "")
                args = (
                    obj.get("arguments")
                    or obj.get("parameters")
                    or obj.get("function", {}).get("arguments", {})
                )
                if isinstance(args, str):
                    args = json.loads(args)

                if name:
                    results.append(ToolCall(
                        id=obj.get("id", f"call_{len(results)}"),
                        name=name,
                        arguments=args or {},
                        raw=match.group(),
                    ))
            except (json.JSONDecodeError, TypeError):
                continue

        # Also try parsing the entire content as a tool call
        if not results:
            try:
                obj = json.loads(content)
                if isinstance(obj, dict) and "name" in obj:
                    args = obj.get("arguments", obj.get("parameters", {}))
                    if isinstance(args, str):
                        args = json.loads(args)
                    results.append(ToolCall(
                        id=obj.get("id", "call_0"),
                        name=obj["name"],
                        arguments=args or {},
                        raw=content,
                    ))
            except (json.JSONDecodeError, TypeError):
                pass

        return results

    def _parse_hermes_xml(self, content: str) -> list[ToolCall]:
        """Parse Hermes-XML format: <tool_call>{json}</tool_call>

        Used by Qwen3 default and Qwen3-Coder (with XML parameter variant).
        """
        results = []

        # Standard Hermes format: <tool_call>{"name": ..., "arguments": ...}</tool_call>
        pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
        matches = re.finditer(pattern, content, re.DOTALL)

        for match in matches:
            inner = match.group(1).strip()
            try:
                obj = json.loads(inner)
                name = obj.get("name", "")
                args = obj.get("arguments", obj.get("parameters", {}))
                if isinstance(args, str):
                    args = json.loads(args)

                results.append(ToolCall(
                    id=f"hermes_{len(results)}",
                    name=name,
                    arguments=args or {},
                    raw=match.group(),
                ))
            except json.JSONDecodeError:
                # Qwen3-Coder XML parameter variant
                # <tool_call>function_name\n<param1>value1</param1>\n<param2>value2</param2></tool_call>
                lines = inner.strip().split("\n")
                if lines:
                    name = lines[0].strip()
                    args = {}
                    for line in lines[1:]:
                        param_match = re.match(r'<(\w+)>(.*?)</\1>', line.strip())
                        if param_match:
                            key = param_match.group(1)
                            value = param_match.group(2)
                            # Try to parse value as JSON
                            try:
                                args[key] = json.loads(value)
                            except (json.JSONDecodeError, ValueError):
                                args[key] = value

                    if name:
                        results.append(ToolCall(
                            id=f"hermes_xml_{len(results)}",
                            name=name,
                            arguments=args,
                            raw=match.group(),
                        ))

        return results

    def _parse_glm_xml(self, content: str) -> list[ToolCall]:
        """Parse GLM XML format: <tool_call>{name}<arg_key/><arg_value/></tool_call>

        Used by GLM-4.6/4.7/5.x models.
        """
        results = []

        pattern = r'<tool_call>(.*?)</tool_call>'
        matches = re.finditer(pattern, content, re.DOTALL)

        for match in matches:
            inner = match.group(1).strip()

            # GLM format: function_name followed by self-closing arg tags
            # <tool_call>get_weather<city>London</city><units>celsius</units></tool_call>
            lines = inner.split("\n") if "\n" in inner else [inner]
            name = ""
            args = {}

            # First non-tag content is the function name
            first_line = lines[0] if lines else ""
            name_match = re.match(r'^([a-zA-Z_]\w*)', first_line)
            if name_match:
                name = name_match.group(1)

            # Parse argument tags
            arg_pattern = r'<(\w+)>(.*?)</\1>'
            for arg_match in re.finditer(arg_pattern, inner):
                key = arg_match.group(1)
                value = arg_match.group(2)
                try:
                    args[key] = json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    args[key] = value

            if name:
                results.append(ToolCall(
                    id=f"glm_{len(results)}",
                    name=name,
                    arguments=args,
                    raw=match.group(),
                ))

        return results

    def _parse_kimi_special(self, content: str) -> list[ToolCall]:
        """Parse Kimi special token format.

        Format: <|tool_calls_section_begin|>
                <|tool_call_begin|>functions.name:idx
                <|tool_sep|>{json_args}<|tool_call_end|>
                <|tool_calls_section_end|>

        Requires strict functions.<name>:<idx> ID canonicalization.
        """
        results = []

        # Extract tool calls section
        section_pattern = (
            r'<\|tool_calls_section_begin\|>(.*?)<\|tool_calls_section_end\|>'
        )
        section_match = re.search(section_pattern, content, re.DOTALL)
        if not section_match:
            return results

        section = section_match.group(1)

        # Parse individual tool calls
        call_pattern = (
            r'<\|tool_call_begin\|>\s*'
            r'functions\.(\w+):(\d+)\s*'
            r'<\|tool_sep\|>\s*(.*?)\s*'
            r'<\|tool_call_end\|>'
        )
        for call_match in re.finditer(call_pattern, section, re.DOTALL):
            name = call_match.group(1)
            idx = call_match.group(2)
            args_str = call_match.group(3).strip()

            try:
                args = json.loads(args_str)
            except json.JSONDecodeError:
                args = {"raw": args_str}

            results.append(ToolCall(
                id=f"functions.{name}:{idx}",
                name=name,
                arguments=args,
                raw=call_match.group(),
            ))

        return results

    def format_tools_for_model(
        self,
        tools: list[dict[str, Any]],
        format: ToolCallFormat = ToolCallFormat.OPENAI_JSON,
    ) -> str | list[dict[str, Any]]:
        """Format tool definitions for a specific model format.

        Args:
            tools: List of tool definitions in OpenAI format.
            format: Target format for the model.

        Returns:
            Formatted tools (string for XML formats, list for JSON formats).
        """
        if format == ToolCallFormat.OPENAI_JSON:
            return tools

        if format == ToolCallFormat.HERMES_XML:
            # Hermes system prompt format
            lines = ["You have access to the following tools:\n"]
            for tool in tools:
                func = tool.get("function", tool)
                lines.append(f"Tool: {func['name']}")
                lines.append(f"Description: {func.get('description', '')}")
                params = func.get("parameters", {})
                if params:
                    lines.append(f"Parameters: {json.dumps(params, indent=2)}")
                lines.append("")
            lines.append(
                "To call a tool, use: <tool_call>{\"name\": \"tool_name\", \"arguments\": {...}}</tool_call>"
            )
            return "\n".join(lines)

        if format == ToolCallFormat.GLM_XML:
            lines = ["Available tools:\n"]
            for tool in tools:
                func = tool.get("function", tool)
                lines.append(f"<tool>{func['name']}: {func.get('description', '')}</tool>")
            lines.append(
                "\nCall tools using: <tool_call>tool_name<param>value</param></tool_call>"
            )
            return "\n".join(lines)

        if format == ToolCallFormat.KIMI_SPECIAL:
            # Kimi uses standard OpenAI format for input, special tokens for output
            return tools

        return tools
