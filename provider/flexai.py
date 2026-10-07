import logging

from dify_plugin import ModelProvider
from dify_plugin.entities.model import ModelType
from dify_plugin.errors.model import CredentialsValidateFailedError

logger = logging.getLogger(__name__)

# A cheap, non-reasoning-by-default model that FlexAI serves on its global endpoint.
VALIDATION_MODEL = "DeepSeek-V4-Flash-0731"


class FlexAIProvider(ModelProvider):
    def validate_provider_credentials(self, credentials: dict) -> None:
        try:
            model_instance = self.get_model_instance(ModelType.LLM)
            model_instance.validate_credentials(model=VALIDATION_MODEL, credentials=credentials)
        except CredentialsValidateFailedError:
            raise
        except Exception:
            logger.exception("FlexAI credentials validation failed")
            raise
