from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple, cast
from openai import APIConnectionError, APIStatusError, AsyncOpenAI, AsyncStream
from openai._types import FileTypes, Omit, omit
from openai.types.responses import (
    Response,
    ResponseInputParam,
    ResponseStreamEvent,
    ResponseTextConfigParam,
    ResponseUsage,
    ToolParam,
)

import asyncio
import base64
import io
import json

from src.common.logger import get_logger
from src.config.model_configs import APIProvider, ReasoningParseMode, ToolArgumentParseMode
from src.llm_models.exceptions import (
    EmptyResponseException,
    NetworkConnectionError,
    ReqAbortException,
    RespNotOkException,
    RespParseException,
)
from src.llm_models.openai_compat import (
    build_openai_compatible_client_config,
    split_openai_request_overrides,
)
from src.llm_models.payload_content.message import ImageMessagePart, Message, RoleType, TextMessagePart
from src.llm_models.payload_content.resp_format import RespFormat, RespFormatType
from src.llm_models.payload_content.tool_option import ToolCall, ToolOption

from .adapter_base import (
    AdapterClient,
    ProviderResponseParser,
    ProviderStreamResponseHandler,
    await_task_with_interrupt,
)
from .base_client import (
    APIResponse,
    AudioTranscriptionRequest,
    EmbeddingRequest,
    ResponseRequest,
    UsageTuple,
    client_registry,
)
from .openai_client import (
    SUPPORTED_OPENAI_IMAGE_FORMATS,
    _apply_xml_tool_call_fallback,
    _build_api_status_message,
    _build_fallback_tool_call_id,
    _coerce_openai_argument,
    _extract_reasoning_and_content,
    _extract_usage_record,
    _normalize_image_part_for_openai,
    _normalize_reasoning_parse_mode,
    _normalize_tool_argument_parse_mode,
    _parse_tool_arguments,
    _sanitize_messages_for_toolless_request,
    _snapshot_openai_argument,
)
from ..request_snapshot import (
    attach_request_snapshot,
    has_request_snapshot,
    save_failed_request_snapshot,
    serialize_audio_request_snapshot,
    serialize_embedding_request_snapshot,
    serialize_response_request_snapshot,
)

logger = get_logger("llm_models")

RESPONSES_RESERVED_EXTRA_BODY_KEYS = {
    "input",
    "max_output_tokens",
    "max_tokens",
    "model",
    "stream",
    "temperature",
    "text",
    "tools",
}
"""由 Responses 客户端显式承载、不应再落入 `extra_body` 的字段集合。"""


def _build_responses_text_part(text: str) -> Dict[str, str]:
    """构建 Responses API 文本输入片段。

    Args:
        text: 文本内容。

    Returns:
        Dict[str, str]: Responses API 所需的文本片段。
    """
    return {
        "type": "input_text",
        "text": text,
    }


def _build_responses_image_part(part: ImageMessagePart) -> Dict[str, str]:
    """构建 Responses API 图片输入片段。

    Args:
        part: 内部图片消息片段。

    Returns:
        Dict[str, str]: Responses API 所需的图片片段；图片不可用时返回文本占位片段。
    """
    normalized_image = _normalize_image_part_for_openai(part)
    if normalized_image is None:
        return _build_responses_text_part("[图片内容不可用]")

    image_format, image_base64 = normalized_image
    return {
        "type": "input_image",
        "image_url": f"data:image/{image_format};base64,{image_base64}",
        "detail": "auto",
    }


def _convert_text_parts_to_responses_content(message: Message) -> str | List[Dict[str, str]]:
    """将仅允许文本的消息转换为 Responses API 内容。

    Args:
        message: 内部统一消息对象。

    Returns:
        str | List[Dict[str, str]]: Responses API 支持的文本内容结构。

    Raises:
        ValueError: 当消息中包含非文本片段时抛出。
    """
    if not message.parts:
        return ""
    if len(message.parts) == 1 and isinstance(message.parts[0], TextMessagePart):
        return message.parts[0].text

    content: List[Dict[str, str]] = []
    for part in message.parts:
        if not isinstance(part, TextMessagePart):
            raise ValueError(f"{message.role.value} 消息仅支持文本片段")
        content.append(_build_responses_text_part(part.text))
    return content


def _convert_user_content_to_responses_content(message: Message) -> str | List[Dict[str, str]]:
    """将用户消息内容转换为 Responses API 支持的文本/图片片段。

    Args:
        message: 内部统一用户消息对象。

    Returns:
        str | List[Dict[str, str]]: Responses API 支持的用户消息内容结构。
    """
    if len(message.parts) == 1 and isinstance(message.parts[0], TextMessagePart):
        return message.parts[0].text

    content: List[Dict[str, str]] = []
    for part in message.parts:
        if isinstance(part, TextMessagePart):
            content.append(_build_responses_text_part(part.text))
            continue
        content.append(_build_responses_image_part(part))
    return content


def _convert_tool_call_to_responses_item(tool_call: ToolCall) -> Dict[str, Any]:
    """将内部工具调用转换为 Responses API 的 function_call item。

    Args:
        tool_call: 内部统一工具调用对象。

    Returns:
        Dict[str, Any]: Responses API input 中可复用的 function_call item。
    """
    item: Dict[str, Any] = {
        "type": "function_call",
        "call_id": tool_call.call_id,
        "name": tool_call.func_name,
        "arguments": json.dumps(tool_call.args or {}, ensure_ascii=False),
    }
    if tool_call.extra_content and isinstance(tool_call.extra_content.get("id"), str):
        item["id"] = tool_call.extra_content["id"]
    return item


def _convert_messages_to_responses_input(messages: List[Message]) -> List[Dict[str, Any]]:
    """将内部消息列表转换为 Responses API 的 input 列表。

    Args:
        messages: 内部统一消息列表。

    Returns:
        List[Dict[str, Any]]: OpenAI Responses SDK 所需的 input item 列表。

    Raises:
        ValueError: 当消息角色不受支持或 Tool 消息缺少 tool_call_id 时抛出。
    """
    converted_input: List[Dict[str, Any]] = []

    for message in messages:
        if message.role == RoleType.System:
            converted_input.append(
                {
                    "role": "system",
                    "content": _convert_text_parts_to_responses_content(message),
                }
            )
            continue

        if message.role == RoleType.User:
            converted_input.append(
                {
                    "role": "user",
                    "content": _convert_user_content_to_responses_content(message),
                }
            )
            continue

        if message.role == RoleType.Assistant:
            if message.parts:
                converted_input.append(
                    {
                        "role": "assistant",
                        "content": _convert_text_parts_to_responses_content(message),
                    }
                )
            if message.tool_calls:
                converted_input.extend(
                    _convert_tool_call_to_responses_item(tool_call) for tool_call in message.tool_calls
                )
            continue

        if message.role == RoleType.Tool:
            if message.tool_call_id is None:
                raise ValueError("Tool 消息缺少 tool_call_id")
            converted_input.append(
                {
                    "type": "function_call_output",
                    "call_id": message.tool_call_id,
                    "output": message.get_text_content(),
                }
            )
            continue

        raise ValueError(f"不支持的消息角色：{message.role}")

    return converted_input


def _convert_tool_options_to_responses_tools(tool_options: List[ToolOption]) -> List[Dict[str, Any]]:
    """将内部工具定义转换为 Responses API function tools。

    Args:
        tool_options: 内部统一工具定义列表。

    Returns:
        List[Dict[str, Any]]: Responses SDK 所需的扁平 function tool 定义列表。
    """
    converted_tools: List[Dict[str, Any]] = []
    for tool_option in tool_options:
        converted_tools.append(
            {
                "type": "function",
                "name": tool_option.name,
                "description": tool_option.description,
                "parameters": tool_option.parameters_schema or {"type": "object", "properties": {}},
                "strict": False,
            }
        )
    return converted_tools


def _convert_response_format_to_responses_text(response_format: RespFormat | None) -> ResponseTextConfigParam | Omit:
    """将内部响应格式转换为 Responses API 的 text 配置。

    Args:
        response_format: 内部统一响应格式定义。

    Returns:
        ResponseTextConfigParam | Omit: Responses SDK 的 text 参数；未指定时返回 `omit`。
    """
    if response_format is None or response_format.format_type == RespFormatType.TEXT:
        return omit
    if response_format.format_type == RespFormatType.JSON_OBJ:
        return cast(ResponseTextConfigParam, {"format": {"type": "json_object"}})
    if response_format.format_type == RespFormatType.JSON_SCHEMA and response_format.schema is not None:
        schema_wrapper = response_format.schema
        schema_object = schema_wrapper.get("schema")
        if not isinstance(schema_object, dict):
            raise ValueError("Responses API JSON Schema 配置缺少有效的 schema 字段。")
        schema_payload: Dict[str, Any] = {
            "type": "json_schema",
            "name": schema_wrapper["name"],
            "schema": schema_object,
        }
        description = schema_wrapper.get("description")
        if description:
            schema_payload["description"] = description
        strict = schema_wrapper.get("strict")
        if strict is not None:
            schema_payload["strict"] = strict
        return cast(ResponseTextConfigParam, {"format": schema_payload})
    return omit


def _coerce_responses_text_argument(value: Any) -> ResponseTextConfigParam | Omit:
    """将 Responses API 的 text 参数转换为 SDK 类型。"""
    if value is omit:
        return omit
    if value is None:
        return omit
    return cast(ResponseTextConfigParam, value)


def _extract_responses_usage_record(usage: ResponseUsage | None) -> UsageTuple | None:
    """从 Responses API usage 对象中提取统一使用量。

    Args:
        usage: OpenAI Responses SDK 返回的 usage 对象。

    Returns:
        UsageTuple | None: 统一使用量元组；缺少 usage 时返回 None。
    """
    if usage is None:
        return None

    prompt_tokens = usage.input_tokens
    completion_tokens = usage.output_tokens
    prompt_cache_hit_tokens = 0
    input_tokens_details = usage.input_tokens_details
    if input_tokens_details is not None:
        prompt_cache_hit_tokens = input_tokens_details.cached_tokens or 0
    prompt_cache_miss_tokens = max(prompt_tokens - prompt_cache_hit_tokens, 0)
    return (
        prompt_tokens,
        completion_tokens,
        usage.total_tokens,
        prompt_cache_hit_tokens,
        prompt_cache_miss_tokens,
    )


def _extract_text_from_output_message(output_item: Any) -> str | None:
    """从 Responses message output item 中提取文本。

    Args:
        output_item: Responses API 返回的 message output item。

    Returns:
        str | None: 聚合后的输出文本；缺少文本内容时返回 None。
    """
    text_parts: List[str] = []
    for content_part in output_item.content or []:
        content_type = content_part.type
        if content_type == "output_text" and isinstance(content_part.text, str):
            text_parts.append(content_part.text)
        if content_type == "refusal" and isinstance(content_part.refusal, str):
            text_parts.append(content_part.refusal)
    return "".join(text_parts) or None


def _extract_reasoning_text(output_item: Any) -> str | None:
    """从 Responses reasoning item 中提取可见推理摘要。

    Args:
        output_item: Responses API 返回的 reasoning output item。

    Returns:
        str | None: 聚合后的推理文本；缺少可见推理内容时返回 None。
    """
    text_parts: List[str] = []
    for content_part in output_item.content or []:
        text = content_part.text
        if isinstance(text, str) and text:
            text_parts.append(text)
    for summary_part in output_item.summary or []:
        text = summary_part.text
        if isinstance(text, str) and text:
            text_parts.append(text)
    return "\n".join(text_parts) or None


def _extract_incomplete_reason(response: Response) -> str | None:
    """提取 Responses API 不完整响应原因。

    Args:
        response: OpenAI Responses SDK 返回的响应对象。

    Returns:
        str | None: `incomplete_details.reason` 字段；缺少时返回 None。
    """
    incomplete_details = response.incomplete_details
    if incomplete_details is None:
        return None
    reason = incomplete_details.reason
    return reason if isinstance(reason, str) and reason else None


def _log_incomplete_response(response: Response) -> None:
    """记录 Responses API 不完整响应告警。

    Args:
        response: OpenAI Responses SDK 返回的响应对象。
    """
    if response.status != "incomplete":
        return

    model_name = response.model or ""
    reason = _extract_incomplete_reason(response)
    if reason == "max_output_tokens":
        logger.info(f"Responses API 模型{model_name}因为达到 max_output_tokens 限制，可能仅输出部分内容，可视情况调整")
        return
    if reason == "content_filter":
        logger.warning(f"Responses API 模型{model_name}因内容过滤返回不完整响应，可能仅包含部分内容")
        return
    logger.warning(f"Responses API 模型{model_name}返回 incomplete 状态，可能仅包含部分内容")


def _parse_responses_output(
    response: Response,
    *,
    reasoning_parse_mode: ReasoningParseMode,
    tool_argument_parse_mode: ToolArgumentParseMode,
) -> APIResponse:
    """解析 Responses API 非流式响应主体。

    Args:
        response: OpenAI Responses SDK 返回的响应对象。
        reasoning_parse_mode: 推理内容解析模式。
        tool_argument_parse_mode: 工具参数解析模式。

    Returns:
        APIResponse: 解析后的统一响应对象。

    Raises:
        RespParseException: 当响应状态异常或工具调用字段缺失时抛出。
        EmptyResponseException: 当响应既没有文本也没有工具调用时抛出。
    """
    response_error = response.error
    if response_error is not None:
        raise RespParseException(response, f"Responses API 返回错误：{response_error.message}")

    if response.status in {"failed", "cancelled"}:
        raise RespParseException(response, f"Responses API 返回异常状态：{response.status}")
    _log_incomplete_response(response)

    api_response = APIResponse(raw_data=response)
    content = response.output_text
    has_output_text = isinstance(content, str) and bool(content)
    message_text_parts: List[str] = []
    reasoning_parts: List[str] = []
    tool_calls: List[ToolCall] = []

    for output_item in response.output or []:
        if output_item.type == "message":
            if not has_output_text:
                message_text = _extract_text_from_output_message(output_item)
                if message_text:
                    message_text_parts.append(message_text)
            continue

        if output_item.type == "function_call":
            raw_arguments = output_item.arguments or ""
            arguments = _parse_tool_arguments(raw_arguments, tool_argument_parse_mode, response)
            function_name = output_item.name
            if not function_name:
                raise RespParseException(response, "Responses API 工具调用缺少函数名。")
            call_id = output_item.call_id or output_item.id
            if not isinstance(call_id, str) or not call_id:
                call_id = _build_fallback_tool_call_id("response_tool_call")
            tool_calls.append(
                ToolCall(
                    call_id=call_id,
                    func_name=function_name,
                    args=arguments,
                    extra_content={"id": output_item.id},
                )
            )
            continue

        if output_item.type == "reasoning" and reasoning_parse_mode != ReasoningParseMode.NONE:
            reasoning_text = _extract_reasoning_text(output_item)
            if reasoning_text:
                reasoning_parts.append(reasoning_text)

    if not has_output_text:
        content = "".join(message_text_parts) or None

    if isinstance(content, str) and content:
        reasoning_content, final_content = _extract_reasoning_and_content(
            content=content,
            parse_mode=reasoning_parse_mode,
        )
        if reasoning_content:
            reasoning_parts.append(reasoning_content)
        api_response.content = final_content

    if reasoning_parts:
        api_response.reasoning_content = "\n".join(reasoning_parts)
    if tool_calls:
        api_response.tool_calls = tool_calls

    _apply_xml_tool_call_fallback(api_response, tool_argument_parse_mode, response)

    if not api_response.content and not api_response.tool_calls:
        raise EmptyResponseException(response)

    return api_response


def _default_responses_response_parser(
    response: Response,
    *,
    reasoning_parse_mode: ReasoningParseMode,
    tool_argument_parse_mode: ToolArgumentParseMode,
) -> Tuple[APIResponse, UsageTuple | None]:
    """解析 Responses API 非流式响应。"""
    api_response = _parse_responses_output(
        response,
        reasoning_parse_mode=reasoning_parse_mode,
        tool_argument_parse_mode=tool_argument_parse_mode,
    )
    usage_record = _extract_responses_usage_record(response.usage)
    return api_response, usage_record


@dataclass(slots=True)
class _ResponsesStreamedToolCallState:
    """Responses API 流式工具调用累积状态。"""

    output_index: int
    call_id: str = ""
    function_name: str = ""
    arguments_buffer: io.StringIO = field(default_factory=io.StringIO)

    def append_arguments(self, arguments_chunk: str) -> None:
        self.arguments_buffer.write(arguments_chunk)

    def replace_arguments(self, arguments: str) -> None:
        if not self.arguments_buffer.closed:
            self.arguments_buffer.close()
        self.arguments_buffer = io.StringIO(arguments)

    def close(self) -> None:
        if not self.arguments_buffer.closed:
            self.arguments_buffer.close()


class _ResponsesStreamAccumulator:
    """Responses API 流式响应累积器。"""

    def __init__(
        self,
        *,
        reasoning_parse_mode: ReasoningParseMode,
        tool_argument_parse_mode: ToolArgumentParseMode,
    ) -> None:
        self.reasoning_parse_mode = reasoning_parse_mode
        self.tool_argument_parse_mode = tool_argument_parse_mode
        self.content_buffer = io.StringIO()
        self.reasoning_buffer = io.StringIO()
        self.tool_call_states: Dict[int, _ResponsesStreamedToolCallState] = {}
        self.completed_response: Response | None = None
        self.incomplete_response: Response | None = None
        self.stream_error: RespParseException | None = None
        self.usage_record: UsageTuple | None = None
        self.model_name: str | None = None
        self.content_written_keys: Set[Tuple[int, int, str]] = set()
        self.reasoning_written_keys: Set[Tuple[int, int, str]] = set()

    def process_event(self, event: ResponseStreamEvent) -> None:
        event_type = event.type
        event_data = cast(Any, event)

        if event_type == "error":
            self.stream_error = RespParseException(event, event_data.message)
            return

        if event_type in {"response.created", "response.in_progress"}:
            event_response = event_data.response
            if event_response.model:
                self.model_name = event_response.model
            return

        if event_type == "response.output_text.delta":
            if event_data.delta:
                self.content_written_keys.add(
                    self._event_part_key(event_data.output_index, event_data.content_index, "output_text")
                )
                self.content_buffer.write(event_data.delta)
            return

        if event_type == "response.refusal.delta":
            if event_data.delta:
                self.content_written_keys.add(
                    self._event_part_key(event_data.output_index, event_data.content_index, "refusal")
                )
                self.content_buffer.write(event_data.delta)
            return

        if event_type == "response.output_text.done":
            self._process_content_text_done(event_data, event_data.text, part_kind="output_text")
            return

        if event_type == "response.refusal.done":
            self._process_content_text_done(event_data, event_data.refusal, part_kind="refusal")
            return

        if event_type == "response.reasoning_summary_text.delta":
            if event_data.delta and self.reasoning_parse_mode != ReasoningParseMode.NONE:
                self.reasoning_written_keys.add(
                    self._event_part_key(event_data.output_index, event_data.summary_index, "reasoning_summary")
                )
                self.reasoning_buffer.write(event_data.delta)
            return

        if event_type == "response.reasoning_text.delta":
            if event_data.delta and self.reasoning_parse_mode != ReasoningParseMode.NONE:
                self.reasoning_written_keys.add(
                    self._event_part_key(event_data.output_index, event_data.content_index, "reasoning_text")
                )
                self.reasoning_buffer.write(event_data.delta)
            return

        if event_type == "response.output_item.added":
            self._process_output_item_added(event_data)
            return

        if event_type == "response.function_call_arguments.delta":
            self._process_function_arguments_delta(event_data)
            return

        if event_type == "response.function_call_arguments.done":
            self._process_function_arguments_done(event_data)
            return

        if event_type == "response.output_item.done":
            self._process_output_item_done(event_data)
            return

        if event_type in {"response.reasoning_summary_text.done", "response.reasoning_text.done"}:
            self._process_reasoning_text_done(event_data)
            return

        if event_type == "response.failed":
            completed_response = cast(Response, event_data.response)
            self.completed_response = completed_response
            self.usage_record = _extract_responses_usage_record(completed_response.usage)
            return

        if event_type == "response.incomplete":
            incomplete_response = cast(Response, event_data.response)
            self.incomplete_response = incomplete_response
            if incomplete_response.model:
                self.model_name = incomplete_response.model
            self.usage_record = _extract_responses_usage_record(incomplete_response.usage)
            return

        if event_type == "response.completed":
            self.incomplete_response = None
            completed_response = cast(Response, event_data.response)
            self.completed_response = completed_response
            self.usage_record = _extract_responses_usage_record(completed_response.usage)

    @staticmethod
    def _event_part_key(
        output_index: int,
        part_index: int,
        part_kind: str,
    ) -> Tuple[int, int, str]:
        """构建流式内容分片键，用于避免 done 事件重复写入。"""
        return int(output_index), int(part_index), part_kind

    def _process_output_item_added(self, event: Any) -> None:
        output_item = event.item
        if output_item.type != "function_call":
            return
        output_index = event.output_index
        state = self.tool_call_states.setdefault(
            output_index,
            _ResponsesStreamedToolCallState(output_index=output_index),
        )
        state.call_id = output_item.call_id or state.call_id
        state.function_name = output_item.name or state.function_name
        if output_item.arguments:
            state.append_arguments(output_item.arguments)

    def _process_function_arguments_delta(self, event: Any) -> None:
        output_index = event.output_index
        state = self.tool_call_states.setdefault(
            output_index,
            _ResponsesStreamedToolCallState(output_index=output_index),
        )
        state.append_arguments(event.delta)

    def _process_function_arguments_done(self, event: Any) -> None:
        output_index = event.output_index
        state = self.tool_call_states.setdefault(
            output_index,
            _ResponsesStreamedToolCallState(output_index=output_index),
        )
        state.function_name = event.name or state.function_name
        state.replace_arguments(event.arguments)

    def _process_output_item_done(self, event: Any) -> None:
        output_item = event.item
        output_type = output_item.type
        if output_type == "function_call":
            output_index = event.output_index
            state = self.tool_call_states.setdefault(
                output_index,
                _ResponsesStreamedToolCallState(output_index=output_index),
            )
            state.call_id = output_item.call_id or state.call_id
            state.function_name = output_item.name or state.function_name
            state.replace_arguments(output_item.arguments)
            return

        if output_type == "message":
            self._process_message_output_item_done(event, output_item)
            return

        if output_type == "reasoning" and self.reasoning_parse_mode != ReasoningParseMode.NONE:
            self._process_reasoning_output_item_done(event, output_item)

    def _process_message_output_item_done(self, event: Any, output_item: Any) -> None:
        output_index = event.output_index
        for content_index, content_part in enumerate(output_item.content or []):
            content_type = content_part.type
            text = None
            if content_type == "output_text" and isinstance(content_part.text, str):
                text = content_part.text
            if content_type == "refusal" and isinstance(content_part.refusal, str):
                text = content_part.refusal
            key = (int(output_index), int(content_index), str(content_type))
            if text and key not in self.content_written_keys:
                self.content_written_keys.add(key)
                self.content_buffer.write(text)

    def _process_reasoning_output_item_done(self, event: Any, output_item: Any) -> None:
        output_index = event.output_index
        for content_index, content_part in enumerate(output_item.content or []):
            text = content_part.text
            key = (int(output_index), int(content_index), "reasoning_text")
            if isinstance(text, str) and text and key not in self.reasoning_written_keys:
                self.reasoning_written_keys.add(key)
                self.reasoning_buffer.write(text)
        for summary_index, summary_part in enumerate(output_item.summary or []):
            text = summary_part.text
            key = (int(output_index), int(summary_index), "reasoning_summary")
            if isinstance(text, str) and text and key not in self.reasoning_written_keys:
                self.reasoning_written_keys.add(key)
                self.reasoning_buffer.write(text)

    def _process_content_text_done(
        self,
        event: Any,
        text: str,
        *,
        part_kind: str,
    ) -> None:
        key = self._event_part_key(event.output_index, event.content_index, part_kind)
        if isinstance(text, str) and text and key not in self.content_written_keys:
            self.content_written_keys.add(key)
            self.content_buffer.write(text)

    def _process_reasoning_text_done(self, event: Any) -> None:
        if self.reasoning_parse_mode == ReasoningParseMode.NONE:
            return
        event_type = event.type
        if event_type == "response.reasoning_summary_text.done":
            key = self._event_part_key(event.output_index, event.summary_index, "reasoning_summary")
        else:
            key = self._event_part_key(event.output_index, event.content_index, "reasoning_text")
        text = event.text
        if text and key not in self.reasoning_written_keys:
            self.reasoning_written_keys.add(key)
            self.reasoning_buffer.write(text)

    def build_response(self) -> Tuple[APIResponse, UsageTuple | None]:
        if self.stream_error is not None:
            raise self.stream_error

        if self.completed_response is not None:
            return _default_responses_response_parser(
                self.completed_response,
                reasoning_parse_mode=self.reasoning_parse_mode,
                tool_argument_parse_mode=self.tool_argument_parse_mode,
            )

        if self.incomplete_response is not None:
            if not self._has_buffered_response_parts():
                return _default_responses_response_parser(
                    self.incomplete_response,
                    reasoning_parse_mode=self.reasoning_parse_mode,
                    tool_argument_parse_mode=self.tool_argument_parse_mode,
                )
            _log_incomplete_response(self.incomplete_response)

        raw_data = self.incomplete_response or ({"model": self.model_name} if self.model_name else None)
        response = self._build_buffered_response(raw_data)
        return response, self.usage_record

    def _has_buffered_response_parts(self) -> bool:
        """判断是否已经从流式事件累积到可解析内容。"""
        return bool(
            self.content_buffer.getvalue()
            or self.reasoning_buffer.getvalue()
            or self.tool_call_states
        )

    def _build_buffered_response(self, raw_data: Any) -> APIResponse:
        """根据流式增量缓冲区构建响应对象。

        Args:
            raw_data: 需要挂载到响应上的原始数据。

        Returns:
            APIResponse: 基于已累积增量构建的统一响应对象。
        """
        response = APIResponse(raw_data=raw_data)
        content = self.content_buffer.getvalue().strip()
        reasoning_content = self.reasoning_buffer.getvalue().strip()
        if content:
            parsed_reasoning, final_content = _extract_reasoning_and_content(
                content,
                self.reasoning_parse_mode,
            )
            if parsed_reasoning:
                reasoning_content = "\n".join(part for part in [reasoning_content, parsed_reasoning] if part)
            response.content = final_content
        if reasoning_content:
            response.reasoning_content = reasoning_content

        if self.tool_call_states:
            response.tool_calls = []
            for output_index in sorted(self.tool_call_states):
                state = self.tool_call_states[output_index]
                if not state.function_name:
                    raise RespParseException(None, f"响应解析失败，工具调用 {output_index} 缺少函数名。")
                raw_arguments = state.arguments_buffer.getvalue().strip()
                arguments = (
                    _parse_tool_arguments(raw_arguments, self.tool_argument_parse_mode, None)
                    if raw_arguments
                    else {}
                )
                call_id = state.call_id or _build_fallback_tool_call_id(f"response_tool_call_{output_index}")
                response.tool_calls.append(ToolCall(call_id=call_id, func_name=state.function_name, args=arguments))

        _apply_xml_tool_call_fallback(response, self.tool_argument_parse_mode, response.raw_data)
        if not response.content and not response.tool_calls:
            raise EmptyResponseException(response.raw_data)
        return response

    def close(self) -> None:
        if not self.content_buffer.closed:
            self.content_buffer.close()
        if not self.reasoning_buffer.closed:
            self.reasoning_buffer.close()
        for state in self.tool_call_states.values():
            state.close()


async def _default_responses_stream_handler(
    resp_stream: AsyncStream[ResponseStreamEvent],
    interrupt_flag: asyncio.Event | None,
    *,
    reasoning_parse_mode: ReasoningParseMode,
    tool_argument_parse_mode: ToolArgumentParseMode,
) -> Tuple[APIResponse, UsageTuple | None]:
    """处理 Responses API 流式响应。"""
    accumulator = _ResponsesStreamAccumulator(
        reasoning_parse_mode=reasoning_parse_mode,
        tool_argument_parse_mode=tool_argument_parse_mode,
    )
    try:
        async for event in resp_stream:
            if interrupt_flag and interrupt_flag.is_set():
                raise ReqAbortException("请求被外部信号中断")
            accumulator.process_event(event)
        return accumulator.build_response()
    finally:
        accumulator.close()


@client_registry.register_client_class("openai_responses")
class OpenAIResponsesClient(AdapterClient[AsyncStream[ResponseStreamEvent], Response]):
    """OpenAI Responses API 客户端。"""

    client: AsyncOpenAI
    reasoning_parse_mode: ReasoningParseMode
    tool_argument_parse_mode: ToolArgumentParseMode

    def __init__(self, api_provider: APIProvider) -> None:
        """初始化 OpenAI Responses API 客户端。"""
        super().__init__(api_provider)
        client_config = build_openai_compatible_client_config(api_provider)
        self.reasoning_parse_mode = _normalize_reasoning_parse_mode(api_provider.reasoning_parse_mode)
        self.tool_argument_parse_mode = _normalize_tool_argument_parse_mode(api_provider.tool_argument_parse_mode)
        self.client = AsyncOpenAI(
            api_key=client_config.api_key,
            organization=api_provider.organization,
            project=api_provider.project,
            base_url=client_config.base_url,
            timeout=api_provider.timeout,
            max_retries=api_provider.max_retry,
            default_headers=client_config.default_headers or None,
            default_query=client_config.default_query or None,
        )

    def _build_default_stream_response_handler(
        self,
        request: ResponseRequest,
    ) -> ProviderStreamResponseHandler[AsyncStream[ResponseStreamEvent]]:
        del request

        async def default_stream_handler(
            resp_stream: AsyncStream[ResponseStreamEvent],
            flag: asyncio.Event | None,
        ) -> Tuple[APIResponse, UsageTuple | None]:
            return await _default_responses_stream_handler(
                resp_stream,
                flag,
                reasoning_parse_mode=self.reasoning_parse_mode,
                tool_argument_parse_mode=self.tool_argument_parse_mode,
            )

        return default_stream_handler

    def _build_default_response_parser(
        self,
        request: ResponseRequest,
    ) -> ProviderResponseParser[Response]:
        del request

        def default_response_parser(response: Response) -> Tuple[APIResponse, UsageTuple | None]:
            return _default_responses_response_parser(
                response,
                reasoning_parse_mode=self.reasoning_parse_mode,
                tool_argument_parse_mode=self.tool_argument_parse_mode,
            )

        return default_response_parser

    async def _execute_response_request(
        self,
        request: ResponseRequest,
        stream_response_handler: ProviderStreamResponseHandler[AsyncStream[ResponseStreamEvent]],
        response_parser: ProviderResponseParser[Response],
    ) -> Tuple[APIResponse, UsageTuple | None]:
        """执行 OpenAI Responses API 文本/多模态响应请求。"""
        snapshot_provider_request = {
            "base_url": self.api_provider.base_url,
            "endpoint": "/responses",
            "method": "POST",
            "operation": "responses.create",
            "organization": self.api_provider.organization,
            "project": self.api_provider.project,
            "request_kwargs": {},
        }
        model_info = request.model_info

        try:
            request_messages = (
                list(request.message_list)
                if request.tool_options
                else _sanitize_messages_for_toolless_request(request.message_list)
            )
            input_payload = _convert_messages_to_responses_input(request_messages)
            input_argument = cast(ResponseInputParam, input_payload)
            tools_payload = (
                _convert_tool_options_to_responses_tools(request.tool_options) if request.tool_options else None
            )
            tools_argument = cast(List[ToolParam], tools_payload) if tools_payload else omit
            text_payload = _convert_response_format_to_responses_text(request.response_format)
            raw_extra_params = dict(request.extra_params or {})
            request_overrides = split_openai_request_overrides(
                request.extra_params,
                reserved_body_keys=RESPONSES_RESERVED_EXTRA_BODY_KEYS,
            )

            temperature_value = raw_extra_params.get("temperature", request.temperature)
            max_output_tokens_value = raw_extra_params.get(
                "max_output_tokens",
                raw_extra_params.get("max_tokens", request.max_tokens),
            )
            if "text" in raw_extra_params:
                text_payload = _coerce_responses_text_argument(raw_extra_params["text"])

            temperature_argument = _coerce_openai_argument(temperature_value)
            max_output_tokens_argument = _coerce_openai_argument(max_output_tokens_value)
            if "temperature" in request_overrides.extra_body:
                temperature_argument = omit
            if "max_output_tokens" in request_overrides.extra_body:
                max_output_tokens_argument = omit
            if "text" in request_overrides.extra_body:
                text_payload = omit

            snapshot_provider_request["request_kwargs"] = {
                "extra_body": request_overrides.extra_body or None,
                "extra_headers": request_overrides.extra_headers or None,
                "extra_query": request_overrides.extra_query or None,
                "input": input_payload,
                "max_output_tokens": _snapshot_openai_argument(max_output_tokens_argument),
                "model": model_info.model_identifier,
                "stream": bool(model_info.force_stream_mode),
                "temperature": _snapshot_openai_argument(temperature_argument),
                "text": _snapshot_openai_argument(text_payload),
                "tools": tools_payload,
            }

            if model_info.force_stream_mode:
                stream_task: asyncio.Task[AsyncStream[ResponseStreamEvent]] = asyncio.create_task(
                    self.client.responses.create(
                        model=model_info.model_identifier,
                        input=input_argument,
                        tools=tools_argument,
                        temperature=temperature_argument,
                        max_output_tokens=max_output_tokens_argument,
                        stream=True,
                        text=text_payload,
                        extra_headers=request_overrides.extra_headers or None,
                        extra_query=request_overrides.extra_query or None,
                        extra_body=request_overrides.extra_body or None,
                    )
                )
                raw_response = cast(
                    AsyncStream[ResponseStreamEvent],
                    await await_task_with_interrupt(stream_task, request.interrupt_flag),
                )
                return await stream_response_handler(raw_response, request.interrupt_flag)

            response_task: asyncio.Task[Response] = asyncio.create_task(
                self.client.responses.create(
                    model=model_info.model_identifier,
                    input=input_argument,
                    tools=tools_argument,
                    temperature=temperature_argument,
                    max_output_tokens=max_output_tokens_argument,
                    stream=False,
                    text=text_payload,
                    extra_headers=request_overrides.extra_headers or None,
                    extra_query=request_overrides.extra_query or None,
                    extra_body=request_overrides.extra_body or None,
                )
            )
            raw_response = cast(Response, await await_task_with_interrupt(response_task, request.interrupt_flag))
            return response_parser(raw_response)
        except (EmptyResponseException, RespParseException) as exc:
            snapshot_path = save_failed_request_snapshot(
                api_provider=self.api_provider,
                client_type="openai_responses",
                error=exc,
                internal_request=serialize_response_request_snapshot(request),
                model_info=model_info,
                operation="responses.create",
                provider_request=snapshot_provider_request,
            )
            attach_request_snapshot(exc, snapshot_path)
            raise
        except APIConnectionError as exc:
            snapshot_path = save_failed_request_snapshot(
                api_provider=self.api_provider,
                client_type="openai_responses",
                error=exc,
                internal_request=serialize_response_request_snapshot(request),
                model_info=model_info,
                operation="responses.create",
                provider_request=snapshot_provider_request,
            )
            wrapped_error = NetworkConnectionError(str(exc))
            attach_request_snapshot(wrapped_error, snapshot_path)
            raise wrapped_error from exc
        except APIStatusError as exc:
            snapshot_path = save_failed_request_snapshot(
                api_provider=self.api_provider,
                client_type="openai_responses",
                error=exc,
                internal_request=serialize_response_request_snapshot(request),
                model_info=model_info,
                operation="responses.create",
                provider_request=snapshot_provider_request,
            )
            wrapped_error = RespNotOkException(exc.status_code, _build_api_status_message(exc))
            attach_request_snapshot(wrapped_error, snapshot_path)
            raise wrapped_error from exc
        except ReqAbortException:
            raise
        except Exception as exc:
            if has_request_snapshot(exc):
                raise
            snapshot_path = save_failed_request_snapshot(
                api_provider=self.api_provider,
                client_type="openai_responses",
                error=exc,
                internal_request=serialize_response_request_snapshot(request),
                model_info=model_info,
                operation="responses.create",
                provider_request=snapshot_provider_request,
            )
            attach_request_snapshot(exc, snapshot_path)
            raise

    async def _execute_embedding_request(
        self,
        request: EmbeddingRequest,
    ) -> Tuple[APIResponse, UsageTuple | None]:
        """执行 OpenAI 兼容的文本嵌入请求。"""
        model_info = request.model_info
        embedding_input = request.embedding_input
        snapshot_provider_request = {
            "base_url": self.api_provider.base_url,
            "endpoint": "/embeddings",
            "method": "POST",
            "operation": "embeddings.create",
            "organization": self.api_provider.organization,
            "project": self.api_provider.project,
            "request_kwargs": {},
        }

        try:
            request_overrides = split_openai_request_overrides(request.extra_params)
            snapshot_provider_request["request_kwargs"] = {
                "extra_body": request_overrides.extra_body or None,
                "extra_headers": request_overrides.extra_headers or None,
                "extra_query": request_overrides.extra_query or None,
                "input": embedding_input,
                "model": model_info.model_identifier,
            }
            raw_response = await self.client.embeddings.create(
                model=model_info.model_identifier,
                input=embedding_input,
                extra_headers=request_overrides.extra_headers or None,
                extra_query=request_overrides.extra_query or None,
                extra_body=request_overrides.extra_body or None,
            )
        except APIConnectionError as exc:
            snapshot_path = save_failed_request_snapshot(
                api_provider=self.api_provider,
                client_type="openai_responses",
                error=exc,
                internal_request=serialize_embedding_request_snapshot(request),
                model_info=model_info,
                operation="embeddings.create",
                provider_request=snapshot_provider_request,
            )
            wrapped_error = NetworkConnectionError(str(exc))
            attach_request_snapshot(wrapped_error, snapshot_path)
            raise wrapped_error from exc
        except APIStatusError as exc:
            snapshot_path = save_failed_request_snapshot(
                api_provider=self.api_provider,
                client_type="openai_responses",
                error=exc,
                internal_request=serialize_embedding_request_snapshot(request),
                model_info=model_info,
                operation="embeddings.create",
                provider_request=snapshot_provider_request,
            )
            wrapped_error = RespNotOkException(exc.status_code, _build_api_status_message(exc))
            attach_request_snapshot(wrapped_error, snapshot_path)
            raise wrapped_error from exc
        except Exception as exc:
            if has_request_snapshot(exc):
                raise
            snapshot_path = save_failed_request_snapshot(
                api_provider=self.api_provider,
                client_type="openai_responses",
                error=exc,
                internal_request=serialize_embedding_request_snapshot(request),
                model_info=model_info,
                operation="embeddings.create",
                provider_request=snapshot_provider_request,
            )
            attach_request_snapshot(exc, snapshot_path)
            raise

        if not raw_response.data:
            exc = RespParseException(raw_response, "嵌入响应解析失败，缺少 embeddings 数据。")
            snapshot_path = save_failed_request_snapshot(
                api_provider=self.api_provider,
                client_type="openai_responses",
                error=exc,
                internal_request=serialize_embedding_request_snapshot(request),
                model_info=model_info,
                operation="embeddings.create",
                provider_request=snapshot_provider_request,
            )
            attach_request_snapshot(exc, snapshot_path)
            raise exc

        response = APIResponse(embedding=raw_response.data[0].embedding)
        usage_record = _extract_usage_record(raw_response.usage)
        return response, usage_record

    async def _execute_audio_transcription_request(
        self,
        request: AudioTranscriptionRequest,
    ) -> Tuple[APIResponse, UsageTuple | None]:
        """执行 OpenAI 兼容的音频转录请求。"""
        model_info = request.model_info
        snapshot_provider_request = {
            "base_url": self.api_provider.base_url,
            "endpoint": "/audio/transcriptions",
            "method": "POST",
            "operation": "audio.transcriptions.create",
            "organization": self.api_provider.organization,
            "project": self.api_provider.project,
            "request_kwargs": {},
        }

        try:
            request_overrides = split_openai_request_overrides(request.extra_params)
            audio_file: FileTypes = ("audio.wav", io.BytesIO(base64.b64decode(request.audio_base64)))
            snapshot_provider_request["request_kwargs"] = {
                "audio_base64": request.audio_base64,
                "extra_body": request_overrides.extra_body or None,
                "extra_headers": request_overrides.extra_headers or None,
                "extra_query": request_overrides.extra_query or None,
                "file_name": "audio.wav",
                "model": model_info.model_identifier,
            }
            raw_response = await self.client.audio.transcriptions.create(
                model=model_info.model_identifier,
                file=audio_file,
                extra_headers=request_overrides.extra_headers or None,
                extra_query=request_overrides.extra_query or None,
                extra_body=request_overrides.extra_body or None,
            )
        except APIConnectionError as exc:
            snapshot_path = save_failed_request_snapshot(
                api_provider=self.api_provider,
                client_type="openai_responses",
                error=exc,
                internal_request=serialize_audio_request_snapshot(request),
                model_info=model_info,
                operation="audio.transcriptions.create",
                provider_request=snapshot_provider_request,
            )
            wrapped_error = NetworkConnectionError(str(exc))
            attach_request_snapshot(wrapped_error, snapshot_path)
            raise wrapped_error from exc
        except APIStatusError as exc:
            snapshot_path = save_failed_request_snapshot(
                api_provider=self.api_provider,
                client_type="openai_responses",
                error=exc,
                internal_request=serialize_audio_request_snapshot(request),
                model_info=model_info,
                operation="audio.transcriptions.create",
                provider_request=snapshot_provider_request,
            )
            wrapped_error = RespNotOkException(exc.status_code, _build_api_status_message(exc))
            attach_request_snapshot(wrapped_error, snapshot_path)
            raise wrapped_error from exc
        except Exception as exc:
            if has_request_snapshot(exc):
                raise
            snapshot_path = save_failed_request_snapshot(
                api_provider=self.api_provider,
                client_type="openai_responses",
                error=exc,
                internal_request=serialize_audio_request_snapshot(request),
                model_info=model_info,
                operation="audio.transcriptions.create",
                provider_request=snapshot_provider_request,
            )
            attach_request_snapshot(exc, snapshot_path)
            raise

        transcription_text = raw_response if isinstance(raw_response, str) else raw_response.text
        if not isinstance(transcription_text, str):
            exc = RespParseException(raw_response, "音频转写响应解析失败，缺少文本内容。")
            snapshot_path = save_failed_request_snapshot(
                api_provider=self.api_provider,
                client_type="openai_responses",
                error=exc,
                internal_request=serialize_audio_request_snapshot(request),
                model_info=model_info,
                operation="audio.transcriptions.create",
                provider_request=snapshot_provider_request,
            )
            attach_request_snapshot(exc, snapshot_path)
            raise exc
        return APIResponse(content=transcription_text), None

    def get_support_image_formats(self) -> List[str]:
        """获取支持的图片格式列表。"""
        return sorted(SUPPORTED_OPENAI_IMAGE_FORMATS | {"jpg", "gif"})
