# /// script
# requires-python = ">=3.12"
# dependencies = ["openai>=1.40"]
# ///
"""Probe each served chat model for what the plugin claims: a tool call, parallel
tool calls, a streamed tool call, and whether an image is actually read.

    curl -s https://api.flex.ai/v1/models -H "Authorization: Bearer $FLEXAI_API_KEY" > tools/v1-models.snapshot.json
    FLEXAI_API_KEY=sk-... uv run tools/probe.py
"""
import json, os, base64, concurrent.futures as cf, struct, zlib
from openai import OpenAI
c = OpenAI(base_url="https://api.flex.ai/v1", api_key=os.environ["FLEXAI_API_KEY"], timeout=90, max_retries=1)
d = json.load(open("tools/v1-models.snapshot.json"))["data"]
chat = [m for m in d if (m.get("output_modalities") or []) == ["text"] and "text" in (m.get("input_modalities") or [])]
tools = [{"type":"function","function":{"name":"get_weather","description":"Get weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]
def png(r,g,b):
    raw=b"".join(b"\x00"+bytes([r,g,b])*16 for _ in range(16))
    ch=lambda t,dat: struct.pack(">I",len(dat))+t+dat+struct.pack(">I",zlib.crc32(t+dat)&0xffffffff)
    return b"\x89PNG\r\n\x1a\n"+ch(b"IHDR",struct.pack(">IIBBBBB",16,16,8,2,0,0,0))+ch(b"IDAT",zlib.compress(raw))+ch(b"IEND",b"")
def probe(m):
    mid=m["id"]; r={"id":mid}
    try:
        x=c.chat.completions.create(model=mid,messages=[{"role":"user","content":"Say OK."}],max_tokens=2048)
        r["chat"]=bool(x.choices[0].message.content or getattr(x.choices[0].message,"reasoning_content",None))
    except Exception as e: r["chat"]="ERR "+str(e)[:80]
    try:
        x=c.chat.completions.create(model=mid,messages=[{"role":"user","content":"What's the weather in Paris and in Tokyo? Call the tool for each city."}],tools=tools,max_tokens=4096)
        tc=x.choices[0].message.tool_calls or []; r["tool_calls"]=len(tc)
    except Exception as e: r["tool_calls"]="ERR "+str(e)[:80]
    try:
        n=0
        for ch in c.chat.completions.create(model=mid,messages=[{"role":"user","content":"What's the weather in Paris? Use the tool."}],tools=tools,max_tokens=4096,stream=True):
            if ch.choices and ch.choices[0].delta.tool_calls: n+=1
        r["stream_tool"]=n>0
    except Exception as e: r["stream_tool"]="ERR "+str(e)[:80]
    if "image" in (m.get("input_modalities") or []):
        seen = []
        for colour, rgb in (("red", (255, 0, 0)), ("blue", (0, 0, 255))):
            url = "data:image/png;base64," + base64.b64encode(png(*rgb)).decode()
            try:
                x = c.chat.completions.create(model=mid, messages=[{"role": "user", "content": [
                    {"type": "text", "text": "What single color fills this image? Answer with one word only."},
                    {"type": "image_url", "image_url": {"url": url}}]}], max_tokens=4096)
                # Reasoning models may think aloud first; judge the final words.
                seen.append(colour in (x.choices[0].message.content or "").lower()[-40:])
            except Exception:
                seen.append(False)
        r["vision"] = all(seen)
    return r
with cf.ThreadPoolExecutor(8) as ex:
    res=list(ex.map(probe,chat))
json.dump(res,open("tools/probe.snapshot.json","w"),indent=1)
for r in res: print(r)
