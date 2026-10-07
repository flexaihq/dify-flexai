from collections.abc import Generator
from typing import Optional, Union

from dify_plugin import OAICompatLargeLanguageModel
from dify_plugin.entities.model import AIModelEntity, ModelFeature
from dify_plugin.entities.model.llm import LLMResult
from dify_plugin.entities.model.message import PromptMessage, PromptMessageTool

from models._common import apply_endpoint


class FlexAILargeLanguageModel(OAICompatLargeLanguageModel):
    """FlexAI chat models over the OpenAI-compatible chat completions API."""

    def _prepare(self, model: str, credentials: dict) -> None:
        apply_endpoint(credentials)
        credentials["mode"] = "chat"
        # The OpenAI-compatible base only sends `tools` when function_calling_type
        # is "tool_call". Predefined models declare tool support in their YAML
        # features; customizable models set it in the model credential form.
        if "function_calling_type" not in credentials:
            schema = self.get_model_schema(model, credentials)
            features = (schema.features or []) if schema else []
            if ModelFeature.TOOL_CALL in features or ModelFeature.MULTI_TOOL_CALL in features:
                credentials["function_calling_type"] = "tool_call"
                if ModelFeature.STREAM_TOOL_CALL in features:
                    credentials["stream_function_calling"] = "supported"

    def _invoke(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        model_parameters: dict,
        tools: Optional[list[PromptMessageTool]] = None,
        stop: Optional[list[str]] = None,
        stream: bool = True,
        user: Optional[str] = None,
    ) -> Union[LLMResult, Generator]:
        self._prepare(model, credentials)
        return super()._invoke(model, credentials, prompt_messages, model_parameters, tools, stop, stream, user)

    def validate_credentials(self, model: str, credentials: dict) -> None:
        apply_endpoint(credentials)
        credentials["mode"] = "chat"
        super().validate_credentials(model, credentials)

    def get_customizable_model_schema(self, model: str, credentials: dict) -> AIModelEntity:
        apply_endpoint(credentials)
        credentials["mode"] = "chat"
        return super().get_customizable_model_schema(model, credentials)

    def get_num_tokens(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        tools: Optional[list[PromptMessageTool]] = None,
    ) -> int:
        apply_endpoint(credentials)
        credentials["mode"] = "chat"
        return super().get_num_tokens(model, credentials, prompt_messages, tools)
