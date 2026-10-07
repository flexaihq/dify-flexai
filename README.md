# FlexAI

Use [FlexAI](https://flex.ai)'s open-weight models in Dify: chat models with tool calling for agents and workflows, vision models, reasoning models, and the BGE-M3 embedding model for knowledge bases.

FlexAI serves an OpenAI-compatible API at `https://api.flex.ai/v1`. This plugin points Dify at it, so you don't need to configure a generic OpenAI-compatible provider by hand.

## Setup

1. Create a FlexAI account at [platform.flex.ai](https://platform.flex.ai/?utm_source=dify&utm_medium=marketplace&utm_campaign=dify_plugin), create a workspace and generate an API key (`sk-...`). Requests are served once the workspace has a funded balance.
2. In Dify, install **FlexAI** from the Marketplace.
3. Go to **Settings → Model Provider → FlexAI**, paste the API key and save. Dify validates the key with one short request.

## Usage

The models below appear in every model picker: LLM nodes, chatflows, agents and knowledge-base embedding settings.

- **Agents and tools:** every listed chat model supports tool calling, including streamed tool calls. The *Parallel tool calls* column shows which models emitted two tool calls in one turn when asked for two lookups.
- **Vision:** models marked *Vision* read images attached in chat or passed from a workflow.
- **Reasoning:** reasoning models return their thinking separately, and Dify shows it as a collapsible thought.
- **Embeddings:** choose `bge-m3` as the embedding model for a knowledge base.
- **Other models:** FlexAI's catalog changes. To use a model that isn't listed, choose **Add Model**, enter its ID exactly as `GET https://api.flex.ai/v1/models` returns it, and set its context size, tool calling and vision options.

| Model | Context | Vision | Parallel tool calls |
|---|---|---|---|
| `DeepSeek-V4-Flash-0731` | 1M | — | yes |
| `DeepSeek-V4.1-Flash` | 1M | yes | yes |
| `GLM-4.5-Air-FP8` | 128K | — | yes |
| `GLM-5.2` | 128K | — | yes |
| `GLM-5.3-Flash` | 1M | yes | yes |
| `Llama-3.3-70B-Instruct-FP8` | 128K | — | yes |
| `Meta-Llama-3.1-8B-Instruct-FP8` | 128K | — | yes |
| `MiniMax-M2.7` | 200K | — | yes |
| `Mistral-Nemo-Instruct-2407-FP8` | 128K | — | yes |
| `Muse-Glimmer-30B` | 128K | yes | — |
| `NVIDIA-Nemotron-3.5-Lightning-30B-A3B` | 1M | — | — |
| `Qwen3-30B-A3B-Thinking-2507-FP8` | 256K | — | yes |
| `Qwen3-8B-FP8` | 40K | — | yes |
| `Qwen3-Coder-30B-A3B-Instruct-FP8` | 256K | — | yes |
| `Qwen3.5-9B` | 256K | yes | yes |
| `Qwen3.6-27B-FP8` | 256K | yes | yes |
| `Qwen3.6-35B-A3B-FP8` | 256K | yes | yes |
| `Qwen3.8-27B` | 256K | yes | yes |
| `Qwen3.8-Flash-Next` | 256K | yes | yes |
| `Step-3.7-Flash` | 256K | yes | yes |
| `gemma-4-26B-A4B-it` | 256K | yes | yes |
| `gemma-4-31b-it` | 256K | yes | yes |
| `gpt-oss-120b` | 128K | — | — |
| `gpt-oss-20b` | 128K | — | — |

Embedding: `bge-m3` (8K context).

Every capability in this table was observed against the live API before release: a tool call, two tool calls in one turn, a streamed tool call, and a 512×512 image read correctly in two different colours. `tools/generate_models.py` builds the model files from those results, and `tools/probe.py` re-runs the checks. Prices are on [flex.ai/pricing](https://flex.ai/pricing).

## Credentials

One FlexAI API key. Dify stores it encrypted and sends it as a Bearer token on each request.

## Connection requirements

Your Dify instance needs outbound HTTPS access to `api.flex.ai` (port 443). No other endpoint is called.

## Source and support

- Source repository: https://github.com/flexaihq/dify-flexai
- Issues: https://github.com/flexaihq/dify-flexai/issues
- Contact: support@flex.ai
- Privacy: [PRIVACY.md](PRIVACY.md)
