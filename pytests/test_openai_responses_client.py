from types import SimpleNamespace

import pytest

from src.config.model_configs import ReasoningParseMode, ToolArgumentParseMode
from src.llm_models.exceptions import RespParseException
from src.llm_models.model_client import ensure_client_type_loaded
from src.llm_models.model_client.base_client import client_registry
from src.llm_models.model_client.openai_responses_client import (
    _ResponsesStreamAccumulator,
    _convert_messages_to_responses_input,
    _convert_response_format_to_responses_text,
    _convert_tool_options_to_responses_tools,
    _default_responses_response_parser,
    _extract_responses_usage_record,
)
from src.llm_models.payload_content.message import Message, RoleType, TextMessagePart
from src.llm_models.payload_content.resp_format import RespFormat, RespFormatType
from src.llm_models.payload_content.tool_option import ToolCall, ToolOption


def test_openai_responses_client_type_can_be_loaded() -> None:
    ensure_client_type_loaded("openai_responses")

    assert "openai_responses" in client_registry.client_registry


def test_convert_messages_to_responses_input_preserves_function_call_chain() -> None:
    messages = [
        Message(role=RoleType.System, parts=[TextMessagePart(text="你是助手")]),
        Message(role=RoleType.User, parts=[TextMessagePart(text="查天气")]),
        Message(
            role=RoleType.Assistant,
            tool_calls=[ToolCall(call_id="call_weather", func_name="get_weather", args={"city": "杭州"})],
        ),
        Message(
            role=RoleType.Tool,
            parts=[TextMessagePart(text='{"temperature": 22}')],
            tool_call_id="call_weather",
        ),
    ]

    converted_input = _convert_messages_to_responses_input(messages)

    assert converted_input[0] == {"role": "system", "content": "你是助手"}
    assert converted_input[1] == {"role": "user", "content": "查天气"}
    assert converted_input[2] == {
        "type": "function_call",
        "call_id": "call_weather",
        "name": "get_weather",
        "arguments": '{"city": "杭州"}',
    }
    assert converted_input[3] == {
        "type": "function_call_output",
        "call_id": "call_weather",
        "output": '{"temperature": 22}',
    }


def test_convert_tool_options_to_responses_tools_uses_flat_function_schema() -> None:
    tool = ToolOption(
        name="finish",
        description="结束对话",
        parameters_schema_override={"type": "object", "properties": {}},
    )

    converted_tools = _convert_tool_options_to_responses_tools([tool])

    assert converted_tools == [
        {
            "type": "function",
            "name": "finish",
            "description": "结束对话",
            "parameters": {"type": "object", "properties": {}},
            "strict": False,
        }
    ]


def test_convert_response_format_to_responses_text_json_schema() -> None:
    response_format = RespFormat(
        RespFormatType.JSON_SCHEMA,
        {
            "name": "Reply",
            "description": "回复结构",
            "schema": {"type": "object", "properties": {"text": {"type": "string"}}},
            "strict": True,
        },
    )

    text_payload = _convert_response_format_to_responses_text(response_format)

    assert text_payload == {
        "format": {
            "type": "json_schema",
            "name": "Reply",
            "description": "回复结构",
            "schema": {"type": "object", "properties": {"text": {"type": "string"}}},
            "strict": True,
        }
    }


def test_default_responses_response_parser_extracts_text_reasoning_tool_calls_and_usage() -> None:
    response = SimpleNamespace(
        status="completed",
        error=None,
        output_text="正式回复",
        output=[
            SimpleNamespace(type="reasoning", summary=[SimpleNamespace(text="推理摘要")], content=[]),
            SimpleNamespace(
                type="function_call",
                id="fc_1",
                call_id="call_1",
                name="reply",
                arguments='{"msg": "hello"}',
            ),
        ],
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            input_tokens_details=SimpleNamespace(cached_tokens=3),
        ),
    )

    api_response, usage_record = _default_responses_response_parser(
        response,
        reasoning_parse_mode=ReasoningParseMode.AUTO,
        tool_argument_parse_mode=ToolArgumentParseMode.AUTO,
    )

    assert api_response.content == "正式回复"
    assert api_response.reasoning_content == "推理摘要"
    assert api_response.tool_calls == [
        ToolCall(call_id="call_1", func_name="reply", args={"msg": "hello"}, extra_content={"id": "fc_1"})
    ]
    assert usage_record == (10, 5, 15, 3, 7)


def test_extract_responses_usage_record_counts_uncached_prompt_tokens() -> None:
    usage = SimpleNamespace(
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
        input_tokens_details=SimpleNamespace(cached_tokens=0),
    )

    assert _extract_responses_usage_record(usage) == (10, 5, 15, 0, 10)


def test_default_responses_response_parser_aggregates_message_items_when_output_text_empty() -> None:
    response = SimpleNamespace(
        status="completed",
        error=None,
        output_text="",
        output=[
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text="第一段")],
            ),
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="refusal", refusal="第二段")],
            ),
        ],
        usage=None,
    )

    api_response, usage_record = _default_responses_response_parser(
        response,
        reasoning_parse_mode=ReasoningParseMode.AUTO,
        tool_argument_parse_mode=ToolArgumentParseMode.AUTO,
    )

    assert api_response.content == "第一段第二段"
    assert usage_record is None


def test_default_responses_response_parser_reports_failed_status() -> None:
    response = SimpleNamespace(
        status="failed",
        error=SimpleNamespace(message="bad request"),
        output=[],
        output_text="",
        usage=None,
    )

    with pytest.raises(RespParseException, match="bad request"):
        _default_responses_response_parser(
            response,
            reasoning_parse_mode=ReasoningParseMode.AUTO,
            tool_argument_parse_mode=ToolArgumentParseMode.AUTO,
        )


def test_responses_stream_accumulator_builds_text_and_function_call_without_completed_event() -> None:
    accumulator = _ResponsesStreamAccumulator(
        reasoning_parse_mode=ReasoningParseMode.AUTO,
        tool_argument_parse_mode=ToolArgumentParseMode.AUTO,
    )

    try:
        accumulator.process_event(
            SimpleNamespace(type="response.output_text.delta", output_index=0, content_index=0, delta="你好")
        )
        accumulator.process_event(
            SimpleNamespace(
                type="response.output_item.added",
                output_index=1,
                item=SimpleNamespace(type="function_call", call_id="call_1", name="reply", arguments=""),
            )
        )
        accumulator.process_event(
            SimpleNamespace(
                type="response.function_call_arguments.done",
                output_index=1,
                name="reply",
                arguments='{"text": "你好"}',
            )
        )

        api_response, usage_record = accumulator.build_response()
    finally:
        accumulator.close()

    assert api_response.content == "你好"
    assert api_response.tool_calls == [ToolCall(call_id="call_1", func_name="reply", args={"text": "你好"})]
    assert usage_record is None


def test_responses_stream_accumulator_reports_error_event() -> None:
    accumulator = _ResponsesStreamAccumulator(
        reasoning_parse_mode=ReasoningParseMode.AUTO,
        tool_argument_parse_mode=ToolArgumentParseMode.AUTO,
    )

    try:
        accumulator.process_event(SimpleNamespace(type="error", message="stream failed"))

        with pytest.raises(RespParseException, match="stream failed"):
            accumulator.build_response()
    finally:
        accumulator.close()


def test_responses_stream_accumulator_appends_message_done_without_text_deltas() -> None:
    accumulator = _ResponsesStreamAccumulator(
        reasoning_parse_mode=ReasoningParseMode.AUTO,
        tool_argument_parse_mode=ToolArgumentParseMode.AUTO,
    )

    try:
        accumulator.process_event(
            SimpleNamespace(
                type="response.output_item.done",
                output_index=0,
                item=SimpleNamespace(
                    type="message",
                    content=[SimpleNamespace(type="output_text", text="第一段")],
                ),
            )
        )
        accumulator.process_event(
            SimpleNamespace(
                type="response.output_item.done",
                output_index=1,
                item=SimpleNamespace(
                    type="message",
                    content=[SimpleNamespace(type="output_text", text="第二段")],
                ),
            )
        )

        api_response, usage_record = accumulator.build_response()
    finally:
        accumulator.close()

    assert api_response.content == "第一段第二段"
    assert usage_record is None


def test_responses_stream_accumulator_deduplicates_text_done_by_content_index() -> None:
    accumulator = _ResponsesStreamAccumulator(
        reasoning_parse_mode=ReasoningParseMode.AUTO,
        tool_argument_parse_mode=ToolArgumentParseMode.AUTO,
    )

    try:
        accumulator.process_event(
            SimpleNamespace(
                type="response.output_text.delta",
                output_index=0,
                content_index=0,
                delta="第一段",
            )
        )
        accumulator.process_event(
            SimpleNamespace(
                type="response.output_item.done",
                output_index=0,
                item=SimpleNamespace(
                    type="message",
                    content=[
                        SimpleNamespace(type="output_text", text="第一段"),
                        SimpleNamespace(type="refusal", refusal="第二段"),
                    ],
                ),
            )
        )

        api_response, usage_record = accumulator.build_response()
    finally:
        accumulator.close()

    assert api_response.content == "第一段第二段"
    assert usage_record is None


def test_responses_stream_accumulator_deduplicates_refusal_done() -> None:
    accumulator = _ResponsesStreamAccumulator(
        reasoning_parse_mode=ReasoningParseMode.AUTO,
        tool_argument_parse_mode=ToolArgumentParseMode.AUTO,
    )

    try:
        accumulator.process_event(
            SimpleNamespace(
                type="response.refusal.delta",
                output_index=0,
                content_index=0,
                delta="拒绝内容",
            )
        )
        accumulator.process_event(
            SimpleNamespace(
                type="response.refusal.done",
                output_index=0,
                content_index=0,
                refusal="拒绝内容",
            )
        )

        api_response, usage_record = accumulator.build_response()
    finally:
        accumulator.close()

    assert api_response.content == "拒绝内容"
    assert usage_record is None


def test_responses_stream_accumulator_appends_output_text_done_without_delta() -> None:
    accumulator = _ResponsesStreamAccumulator(
        reasoning_parse_mode=ReasoningParseMode.AUTO,
        tool_argument_parse_mode=ToolArgumentParseMode.AUTO,
    )

    try:
        accumulator.process_event(
            SimpleNamespace(
                type="response.output_text.done",
                output_index=0,
                content_index=0,
                text="完整回复",
            )
        )
        accumulator.process_event(
            SimpleNamespace(
                type="response.output_item.done",
                output_index=0,
                item=SimpleNamespace(
                    type="message",
                    content=[SimpleNamespace(type="output_text", text="完整回复")],
                ),
            )
        )

        api_response, usage_record = accumulator.build_response()
    finally:
        accumulator.close()

    assert api_response.content == "完整回复"
    assert usage_record is None


def test_responses_stream_accumulator_parses_incomplete_event_response() -> None:
    response = SimpleNamespace(
        status="incomplete",
        error=None,
        model="demo-model",
        incomplete_details=SimpleNamespace(reason="max_output_tokens"),
        output_text="部分回复",
        output=[],
        usage=SimpleNamespace(
            input_tokens=8,
            output_tokens=4,
            total_tokens=12,
            input_tokens_details=SimpleNamespace(cached_tokens=0),
        ),
    )
    accumulator = _ResponsesStreamAccumulator(
        reasoning_parse_mode=ReasoningParseMode.AUTO,
        tool_argument_parse_mode=ToolArgumentParseMode.AUTO,
    )

    try:
        accumulator.process_event(SimpleNamespace(type="response.incomplete", response=response))

        api_response, usage_record = accumulator.build_response()
    finally:
        accumulator.close()

    assert api_response.content == "部分回复"
    assert usage_record == (8, 4, 12, 0, 8)


def test_responses_stream_accumulator_keeps_delta_text_for_incomplete_event() -> None:
    response = SimpleNamespace(
        status="incomplete",
        error=None,
        model="demo-model",
        incomplete_details=SimpleNamespace(reason="max_output_tokens"),
        output_text="",
        output=[],
        usage=SimpleNamespace(
            input_tokens=8,
            output_tokens=4,
            total_tokens=12,
            input_tokens_details=SimpleNamespace(cached_tokens=0),
        ),
    )
    accumulator = _ResponsesStreamAccumulator(
        reasoning_parse_mode=ReasoningParseMode.AUTO,
        tool_argument_parse_mode=ToolArgumentParseMode.AUTO,
    )

    try:
        accumulator.process_event(
            SimpleNamespace(type="response.output_text.delta", output_index=0, content_index=0, delta="部分回复")
        )
        accumulator.process_event(SimpleNamespace(type="response.incomplete", response=response))

        api_response, usage_record = accumulator.build_response()
    finally:
        accumulator.close()

    assert api_response.content == "部分回复"
    assert api_response.raw_data is response
    assert usage_record == (8, 4, 12, 0, 8)


def test_responses_stream_accumulator_collects_reasoning_text_delta() -> None:
    accumulator = _ResponsesStreamAccumulator(
        reasoning_parse_mode=ReasoningParseMode.AUTO,
        tool_argument_parse_mode=ToolArgumentParseMode.AUTO,
    )

    try:
        accumulator.process_event(
            SimpleNamespace(type="response.reasoning_text.delta", output_index=0, content_index=0, delta="推理增量")
        )
        accumulator.process_event(
            SimpleNamespace(type="response.output_text.delta", output_index=1, content_index=0, delta="正式回复")
        )

        api_response, usage_record = accumulator.build_response()
    finally:
        accumulator.close()

    assert api_response.reasoning_content == "推理增量"
    assert api_response.content == "正式回复"
    assert usage_record is None


def test_responses_stream_accumulator_deduplicates_reasoning_done() -> None:
    accumulator = _ResponsesStreamAccumulator(
        reasoning_parse_mode=ReasoningParseMode.AUTO,
        tool_argument_parse_mode=ToolArgumentParseMode.AUTO,
    )

    try:
        accumulator.process_event(
            SimpleNamespace(
                type="response.reasoning_text.done",
                output_index=0,
                content_index=0,
                text="推理内容",
            )
        )
        accumulator.process_event(
            SimpleNamespace(
                type="response.output_item.done",
                output_index=0,
                item=SimpleNamespace(
                    type="reasoning",
                    content=[SimpleNamespace(text="推理内容")],
                    summary=[],
                ),
            )
        )
        accumulator.process_event(
            SimpleNamespace(type="response.output_text.delta", output_index=1, content_index=0, delta="正式回复")
        )

        api_response, usage_record = accumulator.build_response()
    finally:
        accumulator.close()

    assert api_response.reasoning_content == "推理内容"
    assert api_response.content == "正式回复"
    assert usage_record is None
