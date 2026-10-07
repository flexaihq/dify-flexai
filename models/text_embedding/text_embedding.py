from typing import Optional

from dify_plugin import OAICompatEmbeddingModel
from dify_plugin.entities.model import EmbeddingInputType
from dify_plugin.entities.model.text_embedding import TextEmbeddingResult

from models._common import apply_endpoint


class FlexAITextEmbeddingModel(OAICompatEmbeddingModel):
    """FlexAI embedding models over the OpenAI-compatible embeddings API."""

    def validate_credentials(self, model: str, credentials: dict) -> None:
        apply_endpoint(credentials)
        super().validate_credentials(model, credentials)

    def _invoke(
        self,
        model: str,
        credentials: dict,
        texts: list[str],
        user: Optional[str] = None,
        input_type: EmbeddingInputType = EmbeddingInputType.DOCUMENT,
    ) -> TextEmbeddingResult:
        apply_endpoint(credentials)
        return super()._invoke(model, credentials, texts, user, input_type)

    def get_num_tokens(self, model: str, credentials: dict, texts: list[str]) -> list[int]:
        apply_endpoint(credentials)
        return super().get_num_tokens(model, credentials, texts)
