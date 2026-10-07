"""Generate the model YAML files from FlexAI's live catalog and a capability probe.

Nothing under models/llm/*.yaml or models/text_embedding/*.yaml is hand-written.
Two inputs, both committed beside this script so a regeneration is reviewable:

  tools/v1-models.snapshot.json  GET https://api.flex.ai/v1/models (ids, names,
                                 context windows, prices, input modalities)
  tools/probe.snapshot.json      tools/probe.py run against the same endpoint
                                 (tool calls, parallel tool calls, streamed tool
                                 calls, and whether an image is actually read)

A chat model is published only if the probe saw it complete a tool call; a
feature is claimed only if the probe observed it. Refresh both snapshots, then:

    python3 tools/generate_models.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LLM_DIR = ROOT / "models" / "llm"
EMB_DIR = ROOT / "models" / "text_embedding"
FIRST = ["DeepSeek-V4-Flash-0731"]  # shown first in Dify's model picker
MAX_TOKENS_CAP = 65536


def price(v):
    return f"{float(v):g}" if v is not None else None


def llm_yaml(m, p):
    ctx = int(m["context_length"])
    feats = ["agent-thought", "tool-call"]
    if isinstance(p.get("tool_calls"), int) and p["tool_calls"] >= 2:
        feats.append("multi-tool-call")
    if p.get("stream_tool") is True:
        feats.append("stream-tool-call")
    if p.get("vision") is True:
        feats.append("vision")
    lines = [
        f"model: {m['id']}",
        "label:",
        f"  en_US: {m.get('name') or m['id']}",
        "model_type: llm",
        "features:",
        *[f"  - {f}" for f in feats],
        "model_properties:",
        "  mode: chat",
        f"  context_size: {ctx}",
        "parameter_rules:",
        "  - name: temperature",
        "    use_template: temperature",
        "  - name: top_p",
        "    use_template: top_p",
        "  - name: max_tokens",
        "    use_template: max_tokens",
        "    default: 4096",
        "    min: 1",
        f"    max: {min(ctx, MAX_TOKENS_CAP)}",
    ]
    if "response_format" in (m.get("supported_parameters") or []):
        lines += [
            "  - name: response_format",
            "    use_template: response_format",
            "    options:",
            "      - text",
            "      - json_object",
        ]
    pr = m.get("pricing") or {}
    if pr.get("input_per_mtok") is not None and pr.get("output_per_mtok") is not None:
        lines += [
            "pricing:",
            f"  input: \"{price(pr['input_per_mtok'])}\"",
            f"  output: \"{price(pr['output_per_mtok'])}\"",
            "  unit: \"0.000001\"",
            "  currency: USD",
        ]
    return "\n".join(lines) + "\n"


def emb_yaml(m):
    pr = m.get("pricing") or {}
    lines = [
        f"model: {m['id']}",
        "label:",
        f"  en_US: {m.get('name') or m['id']}",
        "model_type: text-embedding",
        "model_properties:",
        f"  context_size: {int(m['context_length'])}",
        "  max_chunks: 32",
    ]
    if pr.get("input_per_mtok") is not None:
        lines += [
            "pricing:",
            f"  input: \"{price(pr['input_per_mtok'])}\"",
            "  unit: \"0.000001\"",
            "  currency: USD",
        ]
    return "\n".join(lines) + "\n"


def main():
    catalog = json.loads((ROOT / "tools" / "v1-models.snapshot.json").read_text())["data"]
    probe = {r["id"]: r for r in json.loads((ROOT / "tools" / "probe.snapshot.json").read_text())}
    for d in (LLM_DIR, EMB_DIR):
        for f in d.glob("*.yaml"):
            f.unlink()
    order = []
    for m in catalog:
        if m.get("category") == "embedding":
            (EMB_DIR / f"{m['id']}.yaml").write_text(emb_yaml(m))
            continue
        p = probe.get(m["id"])
        if p and isinstance(p.get("vision"), str):
            raise SystemExit(f"{m['id']}: the vision probe errored ({p['vision']}); re-run tools/probe.py")
        if not p:
            continue
        tc = p.get("tool_calls")
        if isinstance(tc, str) and "Error code: 400" not in tc:
            # Only a 400 is the server rejecting the capability; a 429, 5xx or
            # timeout says nothing about the model, so never publish from it.
            raise SystemExit(f"{m['id']}: the tool probe errored ({tc}); re-run tools/probe.py")
        if not isinstance(tc, int) or tc < 1:
            continue
        (LLM_DIR / f"{m['id']}.yaml").write_text(llm_yaml(m, p))
        order.append(m["id"])
    order = [i for i in FIRST if i in order] + sorted(i for i in order if i not in FIRST)
    (LLM_DIR / "_position.yaml").write_text("".join(f"- {i}\n" for i in order))
    print(f"{len(order)} chat models, {len(list(EMB_DIR.glob('*.yaml')))} embedding models")


if __name__ == "__main__":
    main()
