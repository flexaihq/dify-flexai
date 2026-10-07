"""Live checks of the plugin's model classes against https://api.flex.ai/v1.

These drive the same classes Dify loads (FlexAILargeLanguageModel,
FlexAITextEmbeddingModel) with the model schemas built from this plugin's own
YAML files, so a passing run means the shipped configuration works, not a copy
of it. They need a FlexAI key and are skipped without one:

    FLEXAI_API_KEY=sk-... uv run --with "dify_plugin>=0.10.2,<0.11" --with pytest --with pyyaml pytest -q tests
"""

import base64
import os
import struct
import sys
import zlib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dify_plugin.entities.model import AIModelEntity  # noqa: E402
from dify_plugin.entities.model.message import (  # noqa: E402
    AssistantPromptMessage,
    ImagePromptMessageContent,
    PromptMessageTool,
    TextPromptMessageContent,
    ToolPromptMessage,
    UserPromptMessage,
)
from dify_plugin.errors.model import CredentialsValidateFailedError  # noqa: E402

from models.llm.llm import FlexAILargeLanguageModel  # noqa: E402
from models.text_embedding.text_embedding import FlexAITextEmbeddingModel  # noqa: E402
from provider.flexai import VALIDATION_MODEL  # noqa: E402

KEY = os.environ.get("FLEXAI_API_KEY")
pytestmark = pytest.mark.skipif(not KEY, reason="FLEXAI_API_KEY not set")

WEATHER = PromptMessageTool(
    name="get_weather",
    description="Get the current weather for a city.",
    parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
)


def schemas(kind: str) -> list[AIModelEntity]:
    out = []
    for f in sorted((ROOT / "models" / kind).glob("*.yaml")):
        if f.name.startswith("_"):
            continue
        data = yaml.safe_load(f.read_text())
        data["fetch_from"] = "predefined-model"
        out.append(AIModelEntity(**data))
    return out


def creds() -> dict:
    return {"api_key": KEY}


def llm() -> FlexAILargeLanguageModel:
    return FlexAILargeLanguageModel(model_schemas=schemas("llm"))


def features_of(model: str) -> list[str]:
    return [f.value for f in llm().get_model_schema(model, creds()).features]


def first_model_with(feature: str) -> str:
    order = [line[2:].strip() for line in (ROOT / "models/llm/_position.yaml").read_text().splitlines()]
    return next(m for m in order if feature in features_of(m))


@pytest.mark.parametrize("model", [m.model for m in schemas("llm") if "vision" in [f.value for f in m.features]])
def test_every_vision_model_reads_an_image(model):
    img = base64.b64encode(png(0, 0, 255)).decode()
    msg = UserPromptMessage(content=[
        TextPromptMessageContent(data="What single color fills this image? Answer with one word only."),
        ImagePromptMessageContent(format="png", base64_data=img, mime_type="image/png"),
    ])
    text, _ = collect(llm().invoke(model, creds(), [msg], {"max_tokens": 4096}, stream=True))
    assert "blue" in text.lower()[-200:]


def collect(gen):
    text, calls = "", []
    for chunk in gen:
        msg = chunk.delta.message
        if isinstance(msg.content, str):
            text += msg.content
        calls += msg.tool_calls or []
    return text, calls


# Some vision encoders cannot read tiny images: Muse Glimmer answered
# "unknown" for a 16x16 fill and read a 512x512 one correctly.
SIZE = 512


def png(r, g, b):
    raw = b"".join(b"\x00" + bytes([r, g, b]) * SIZE for _ in range(SIZE))

    def ch(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

    return b"\x89PNG\r\n\x1a\n" + ch(b"IHDR", struct.pack(">IIBBBBB", SIZE, SIZE, 8, 2, 0, 0, 0)) + ch(b"IDAT", zlib.compress(raw)) + ch(b"IEND", b"")


def test_every_yaml_loads_as_a_model_schema():
    assert len(schemas("llm")) > 0 and len(schemas("text_embedding")) > 0


def test_credentials_validate_and_a_bad_key_is_refused():
    llm().validate_credentials(VALIDATION_MODEL, creds())
    with pytest.raises(CredentialsValidateFailedError):
        llm().validate_credentials(VALIDATION_MODEL, {"api_key": "sk-not-a-real-key"})


def test_blocking_chat_returns_text():
    # The SDK hands a non-streamed result back as a single-chunk generator.
    text, _ = collect(llm().invoke(VALIDATION_MODEL, creds(), [UserPromptMessage(content="Reply with the word OK.")],
                                   {"max_tokens": 256}, stream=False))
    assert "ok" in text.lower()


POSITION = [line[2:].strip() for line in (ROOT / "models/llm/_position.yaml").read_text().splitlines()]


@pytest.mark.parametrize("model", POSITION)
def test_streamed_tool_call_then_tool_result_round_trip(model):
    """Every listed chat model claims tool calling, so every one is exercised."""
    m = llm()
    msgs = [UserPromptMessage(content="What's the weather in Paris? Use the tool.")]
    _, calls = collect(m.invoke(model, creds(), msgs, {"max_tokens": 4096}, tools=[WEATHER], stream=True))
    assert calls and calls[0].function.name == "get_weather"
    msgs += [
        AssistantPromptMessage(content="", tool_calls=calls),
        ToolPromptMessage(content='{"city": "Paris", "temperature_c": 21, "conditions": "clear"}',
                          tool_call_id=calls[0].id, name="get_weather"),
    ]
    text, _ = collect(m.invoke(model, creds(), msgs, {"max_tokens": 4096}, tools=[WEATHER], stream=True))
    assert "21" in text


def test_reasoning_is_returned_inside_think_tags():
    text, _ = collect(llm().invoke("Qwen3-30B-A3B-Thinking-2507-FP8", creds(),
                                   [UserPromptMessage(content="What is 17*23? Answer with the number.")],
                                   {"max_tokens": 4096}, stream=True))
    assert "<think>" in text and "391" in text.split("</think>")[-1]


def test_customizable_model_with_tool_calling():
    c = {"api_key": KEY, "context_size": "131072", "max_tokens_to_sample": "8192",
         "function_calling_type": "tool_call", "vision_support": "not_support"}
    m = FlexAILargeLanguageModel(model_schemas=[])
    schema = m.get_customizable_model_schema("gpt-oss-20b", dict(c))
    assert "multi-tool-call" in [f.value for f in schema.features]
    _, calls = collect(m.invoke("gpt-oss-20b", dict(c), [UserPromptMessage(content="Weather in Tokyo? Use the tool.")],
                                {"max_tokens": 4096}, tools=[WEATHER], stream=True))
    assert calls and calls[0].function.name == "get_weather"


def test_embeddings():
    emb = FlexAITextEmbeddingModel(model_schemas=schemas("text_embedding"))
    r = emb.invoke("bge-m3", creds(), ["FlexAI serves open-weight models.", "Dify builds AI apps."])
    assert len(r.embeddings) == 2 and len(r.embeddings[0]) > 100
