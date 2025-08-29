import re
import asyncio
import time

from enum import Enum
from rich.traceback import install
from typing import Tuple, List, Dict, Optional, Callable, Any

from src.common.logger import get_logger
from src.config.config import model_config
from src.config.api_ada_configs import APIProvider, ModelInfo, TaskConfig
from .payload_content.message import MessageBuilder, Message
from .payload_content.resp_format import RespFormat
from .payload_content.tool_option import ToolOption, ToolCall, ToolOptionBuilder, ToolParamType
from .model_client.base_client import BaseClient, APIResponse, client_registry
from .utils import compress_messages, llm_usage_recorder
from .exceptions import (
    NetworkConnectionError,
    RespNotOkException,
    EmptyResponseException,
    ModelAttemptFailed,
)

install(extra_lines=3)

logger = get_logger("model_utils")

# 常见Error Code Mapping
error_code_mapping = {
    400: "参数不正确",
    401: "API key 错误，认证失败，请检查 config/model_config.toml 中的配置是否正确",
    402: "账号余额不足",
    403: "需要实名,或余额不足",
    404: "Not Found",
    429: "请求过于频繁，请稍后再试",
    500: "服务器内部故障",
    503: "服务器负载过高",
}


class RequestType(Enum):
    """请求类型枚举"""

    RESPONSE = "response"
    EMBEDDING = "embedding"
    AUDIO = "audio"


class LLMRequest:
    """LLM请求类"""

    def __init__(self, model_set: TaskConfig, request_type: str = "") -> None:
        self.task_name = request_type
        self.model_for_task = model_set
        self.request_type = request_type
        self.model_usage: Dict[str, Tuple[int, int, int]] = {
            model: (0, 0, 0) for model in self.model_for_task.model_list
        }
        """模型使用量记录，用于进行负载均衡，对应为(total_tokens, penalty, usage_penalty)，惩罚值是为了能在某个模型请求不给力或正在被使用的时候进行调整"""

    async def generate_response_for_image(
        self,
        prompt: str,
        image_base64: str,
        image_format: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Tuple[str, Tuple[str, str, Optional[List[ToolCall]]]]:
        """
        为图像生成响应
        Args:
            prompt (str): 提示词
            image_base64 (str): 图像的Base64编码字符串
            image_format (str): 图像格式（如 'png', 'jpeg' 等）
        Returns:
            (Tuple[str, str, str, Optional[List[ToolCall]]]): 响应内容、推理内容、模型名称、工具调用列表
        """
        start_time = time.time()

        def message_factory(client: BaseClient) -> List[Message]:
            message_builder = MessageBuilder()
            message_builder.add_text_content(prompt)
            message_builder.add_image_content(
                image_base64=image_base64,
                image_format=image_format,
                support_formats=client.get_support_image_formats()
            )
            return [message_builder.build()]

        response, model_info = await self._execute_request(
            request_type=RequestType.RESPONSE,
            message_factory=message_factory,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.content or ""
        reasoning_content = response.reasoning_content or ""
        tool_calls = response.tool_calls
        if not reasoning_content and content:
            content, extracted_reasoning = self._extract_reasoning(content)
            reasoning_content = extracted_reasoning
        if usage := response.usage:
            llm_usage_recorder.record_usage_to_database(
                model_info=model_info,
                model_usage=usage,
                user_id="system",
                request_type=self.request_type,
                endpoint="/chat/completions",
                time_cost=time.time() - start_time,
            )
        return content, (reasoning_content, model_info.name, tool_calls)

    async def generate_response_for_voice(self, voice_base64: str) -> Optional[str]:
        """
        为语音生成响应
        Args:
            voice_base64 (str): 语音的Base64编码字符串
        Returns:
            (Optional[str]): 生成的文本描述或None
        """
        response, _ = await self._execute_request(
            request_type=RequestType.AUDIO,
            audio_base64=voice_base64,
        )
        return response.content or None

    async def generate_response_async(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        raise_when_empty: bool = True,
    ) -> Tuple[str, Tuple[str, str, Optional[List[ToolCall]]]]:
        """
        异步生成响应
        Args:
            prompt (str): 提示词
            temperature (float, optional): 温度参数
            max_tokens (int, optional): 最大token数
            tools (Optional[List[Dict[str, Any]]]): 工具列表
            raise_when_empty (bool): 当响应为空时是否抛出异常
        Returns:
            (Tuple[str, str, str, Optional[List[ToolCall]]]): 响应内容、推理内容、模型名称、工具调用列表
        """
        start_time = time.time()
        
        def message_factory(client: BaseClient) -> List[Message]:
            message_builder = MessageBuilder()
            message_builder.add_text_content(prompt)
            return [message_builder.build()]

        tool_built = self._build_tool_options(tools)

        response, model_info = await self._execute_request(
            request_type=RequestType.RESPONSE,
            message_factory=message_factory,
            temperature=temperature,
            max_tokens=max_tokens,
            tool_options=tool_built,
        )

        logger.debug(f"LLM请求总耗时: {time.time() - start_time}")
        content = response.content
        reasoning_content = response.reasoning_content or ""
        tool_calls = response.tool_calls
        if not reasoning_content and content:
            content, extracted_reasoning = self._extract_reasoning(content)
            reasoning_content = extracted_reasoning
        if usage := response.usage:
            llm_usage_recorder.record_usage_to_database(
                model_info=model_info,
                model_usage=usage,
                user_id="system",
                request_type=self.request_type,
                endpoint="/chat/completions",
                time_cost=time.time() - start_time,
            )
        return content, (reasoning_content, model_info.name, tool_calls)

    async def get_embedding(self, embedding_input: str) -> Tuple[List[float], str]:
        """
        获取嵌入向量
        Args:
            embedding_input (str): 获取嵌入的目标
        Returns:
            (Tuple[List[float], str]): (嵌入向量，使用的模型名称)
        """
        start_time = time.time()
        response, model_info = await self._execute_request(
            request_type=RequestType.EMBEDDING,
            embedding_input=embedding_input,
        )
        embedding = response.embedding
        if usage := response.usage:
            llm_usage_recorder.record_usage_to_database(
                model_info=model_info,
                model_usage=usage,
                user_id="system",
                request_type=self.request_type,
                endpoint="/embeddings",
                time_cost=time.time() - start_time,
            )
        if not embedding:
            raise RuntimeError("获取embedding失败")
        return embedding, model_info.name

    def _select_model(self, exclude_models: set = None) -> Tuple[ModelInfo, APIProvider, BaseClient]:
        """
        根据总tokens和惩罚值选择的模型
        """
        available_models = {
            model: scores
            for model, scores in self.model_usage.items()
            if not exclude_models or model not in exclude_models
        }
        if not available_models:
            raise RuntimeError("没有可用的模型可供选择。所有模型均已尝试失败。")

        least_used_model_name = min(
            available_models,
            key=lambda k: available_models[k][0] + available_models[k][1] * 300 + available_models[k][2] * 1000,
        )
        model_info = model_config.get_model_info(least_used_model_name)
        api_provider = model_config.get_provider(model_info.api_provider)
        force_new_client = (self.request_type == "embedding")
        client = client_registry.get_client_class_instance(api_provider, force_new=force_new_client)
        logger.debug(f"选择请求模型: {model_info.name}")
        total_tokens, penalty, usage_penalty = self.model_usage[model_info.name]
        self.model_usage[model_info.name] = (total_tokens, penalty, usage_penalty + 1)
        return model_info, api_provider, client

    async def _attempt_request_on_model(
        self,
        model_info: ModelInfo,
        api_provider: APIProvider,
        client: BaseClient,
        request_type: RequestType,
        message_list: List[Message],
        tool_options: list[ToolOption] | None,
        response_format: RespFormat | None,
        stream_response_handler: Optional[Callable],
        async_response_parser: Optional[Callable],
        temperature: Optional[float],
        max_tokens: Optional[int],
        embedding_input: str | None,
        audio_base64: str | None,
        compressed_messages: Optional[List[Message]] = None,
    ) -> APIResponse:
        """
        在单个模型上执行请求，包含针对临时错误的重试逻辑。
        如果成功，返回APIResponse。如果失败（重试耗尽或硬错误），则抛出ModelAttemptFailed异常。
        """
        retry_remain = api_provider.max_retry

        while retry_remain > 0:
            try:
                if request_type == RequestType.RESPONSE:
                    return await client.get_response(
                        model_info=model_info,
                        message_list=(compressed_messages or message_list),
                        tool_options=tool_options,
                        max_tokens=self.model_for_task.max_tokens if max_tokens is None else max_tokens,
                        temperature=self.model_for_task.temperature if temperature is None else temperature,
                        response_format=response_format,
                        stream_response_handler=stream_response_handler,
                        async_response_parser=async_response_parser,
                        extra_params=model_info.extra_params,
                    )
                elif request_type == RequestType.EMBEDDING:
                    assert embedding_input is not None
                    return await client.get_embedding(
                        model_info=model_info,
                        embedding_input=embedding_input,
                        extra_params=model_info.extra_params,
                    )
                elif request_type == RequestType.AUDIO:
                    assert audio_base64 is not None
                    return await client.get_audio_transcriptions(
                        model_info=model_info,
                        audio_base64=audio_base64,
                        extra_params=model_info.extra_params,
                    )
            except (EmptyResponseException, NetworkConnectionError) as e:
                retry_remain -= 1
                if retry_remain <= 0:
                    logger.error(f"模型 '{model_info.name}' 在用尽对临时错误的重试次数后仍然失败。")
                    raise ModelAttemptFailed(f"模型 '{model_info.name}' 重试耗尽", original_exception=e) from e

                logger.warning(f"模型 '{model_info.name}' 遇到可重试错误: {str(e)}。剩余重试次数: {retry_remain}")
                await asyncio.sleep(api_provider.retry_interval)

            except RespNotOkException as e:
                # 可重试的HTTP错误
                if e.status_code == 429 or e.status_code >= 500:
                    retry_remain -= 1
                    if retry_remain <= 0:
                        logger.error(f"模型 '{model_info.name}' 在遇到 {e.status_code} 错误并用尽重试次数后仍然失败。")
                        raise ModelAttemptFailed(f"模型 '{model_info.name}' 重试耗尽", original_exception=e) from e

                    logger.warning(f"模型 '{model_info.name}' 遇到可重试的HTTP错误: {str(e)}。剩余重试次数: {retry_remain}")
                    await asyncio.sleep(api_provider.retry_interval)
                    continue

                # 特殊处理413，尝试压缩
                if e.status_code == 413 and message_list and not compressed_messages:
                    logger.warning(f"模型 '{model_info.name}' 返回413请求体过大，尝试压缩后重试...")
                    # 压缩消息本身不消耗重试次数
                    compressed_messages = compress_messages(message_list)
                    continue

                # 不可重试的HTTP错误
                logger.warning(f"模型 '{model_info.name}' 遇到不可重试的HTTP错误: {str(e)}")
                raise ModelAttemptFailed(f"模型 '{model_info.name}' 遇到硬错误", original_exception=e) from e

            except Exception as e:
                logger.warning(f"模型 '{model_info.name}' 遇到未知的不可重试错误: {str(e)}")
                raise ModelAttemptFailed(f"模型 '{model_info.name}' 遇到硬错误", original_exception=e) from e

        raise ModelAttemptFailed(f"模型 '{model_info.name}' 未被尝试，因为重试次数已配置为0或更少。")

    async def _execute_request(
        self,
        request_type: RequestType,
        message_factory: Optional[Callable[[BaseClient], List[Message]]] = None,
        tool_options: list[ToolOption] | None = None,
        response_format: RespFormat | None = None,
        stream_response_handler: Optional[Callable] = None,
        async_response_parser: Optional[Callable] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        embedding_input: str | None = None,
        audio_base64: str | None = None,
    ) -> Tuple[APIResponse, ModelInfo]:
        """
        调度器函数，负责模型选择、故障切换。
        """
        failed_models_this_request = set()
        max_attempts = len(self.model_for_task.model_list)
        last_exception: Optional[Exception] = None
        compressed_messages: Optional[List[Message]] = None

        for _attempt in range(max_attempts):
            model_info, api_provider, client = self._select_model(exclude_models=failed_models_this_request)

            message_list = []
            if message_factory:
                message_list = message_factory(client)

            try:
                response = await self._attempt_request_on_model(
                    model_info, api_provider, client, request_type,
                    message_list=message_list,
                    tool_options=tool_options,
                    response_format=response_format,
                    stream_response_handler=stream_response_handler,
                    async_response_parser=async_response_parser,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    embedding_input=embedding_input,
                    audio_base64=audio_base64,
                    compressed_messages=compressed_messages,
                )
                return response, model_info

            except ModelAttemptFailed as e:
                last_exception = e.original_exception or e
                logger.warning(f"模型 '{model_info.name}' 尝试失败，切换到下一个模型。原因: {e}")
                total_tokens, penalty, usage_penalty = self.model_usage[model_info.name]
                self.model_usage[model_info.name] = (total_tokens, penalty + 1, usage_penalty)
                failed_models_this_request.add(model_info.name)

                if isinstance(last_exception, RespNotOkException) and last_exception.status_code == 400:
                    logger.error("收到不可恢复的客户端错误 (400)，中止所有尝试。")
                    raise last_exception from e

            finally:
                total_tokens, penalty, usage_penalty = self.model_usage[model_info.name]
                if usage_penalty > 0:
                    self.model_usage[model_info.name] = (total_tokens, penalty, usage_penalty - 1)

        logger.error(f"所有 {max_attempts} 个模型均尝试失败。")
        if last_exception:
            raise last_exception
        raise RuntimeError("请求失败，所有可用模型均已尝试失败。")

    def _build_tool_options(self, tools: Optional[List[Dict[str, Any]]]) -> Optional[List[ToolOption]]:
        # sourcery skip: extract-method
        """构建工具选项列表"""
        if not tools:
            return None
        tool_options: List[ToolOption] = []
        for tool in tools:
            tool_legal = True
            tool_options_builder = ToolOptionBuilder()
            tool_options_builder.set_name(tool.get("name", ""))
            tool_options_builder.set_description(tool.get("description", ""))
            parameters: List[Tuple[str, str, str, bool, List[str] | None]] = tool.get("parameters", [])
            for param in parameters:
                try:
                    assert isinstance(param, tuple) and len(param) == 5, "参数必须是包含5个元素的元组"
                    assert isinstance(param[0], str), "参数名称必须是字符串"
                    assert isinstance(param[1], ToolParamType), "参数类型必须是ToolParamType枚举"
                    assert isinstance(param[2], str), "参数描述必须是字符串"
                    assert isinstance(param[3], bool), "参数是否必填必须是布尔值"
                    assert isinstance(param[4], list) or param[4] is None, "参数枚举值必须是列表或None"
                    tool_options_builder.add_param(
                        name=param[0],
                        param_type=param[1],
                        description=param[2],
                        required=param[3],
                        enum_values=param[4],
                    )
                except AssertionError as ae:
                    tool_legal = False
                    logger.error(f"{param[0]} 参数定义错误: {str(ae)}")
                except Exception as e:
                    tool_legal = False
                    logger.error(f"构建工具参数失败: {str(e)}")
            if tool_legal:
                tool_options.append(tool_options_builder.build())
        return tool_options or None

    @staticmethod
    def _extract_reasoning(content: str) -> Tuple[str, str]:
        """CoT思维链提取，向后兼容"""
        match = re.search(r"(?:<think>)?(.*?)</think>", content, re.DOTALL)
        content = re.sub(r"(?:<think>)?.*?</think>", "", content, flags=re.DOTALL, count=1).strip()
        reasoning = match[1].strip() if match else ""
        return content, reasoning