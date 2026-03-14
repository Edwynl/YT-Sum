import httpx
import json
import re

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_TRANSLATE_MODEL = "qwen3.5:9b"

TRANSLATE_PROMPT = """将以下英文字幕翻译为中文。要求：
- 口语化、自然流畅
- 技术术语保留英文原词（如 API、Docker、LLM、token 等）
- 只输出 JSON 数组，元素顺序与输入完全一致
- 不要任何解释、markdown 或代码块标记

输入 JSON 数组：
{items}

输出（纯 JSON 数组）："""


def _clean_json_response(raw: str) -> str:
    """清理模型输出，提取纯 JSON 数组"""
    raw = raw.strip()
    # 去掉 markdown 代码块
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    # 找到第一个 [ 到最后一个 ]
    start = raw.find("[")
    end = raw.rfind("]")
    if start != -1 and end != -1 and end > start:
        return raw[start:end + 1]
    return raw


async def batch_translate_stream(
    transcript: list[dict],
    model: str = DEFAULT_TRANSLATE_MODEL,
    batch_size: int = 15,
):
    """
    分批翻译字幕并流式 yield 结果。
    Yield 格式: (index_start, translated_batch_list)
    """
    texts = [t["text"] for t in transcript]
    
    async with httpx.AsyncClient(timeout=120) as client:
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            prompt = TRANSLATE_PROMPT.format(
                items=json.dumps(batch, ensure_ascii=False)
            )

            try:
                resp = await client.post(
                    OLLAMA_URL,
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.1,
                            "num_ctx": 4096,
                        },
                    },
                )
                resp.raise_for_status()
                raw = resp.json()["response"]
                cleaned = _clean_json_response(raw)
                translated = json.loads(cleaned)

                if isinstance(translated, list) and len(translated) == len(batch):
                    yield i, translated
                else:
                    yield i, batch  # 数量不对，返回原文
                
            except Exception as e:
                print(f"Translation batch {i} failed: {e}")
                yield i, batch


async def batch_translate(
    texts: list[str],
    model: str = DEFAULT_TRANSLATE_MODEL,
    batch_size: int = 15,
) -> list[str]:
    """旧接口适配，一次性返回全部结果"""
    results = [None] * len(texts)
    # 构造 dummy transcript
    dummy = [{"text": t} for t in texts]
    async for i, translated in batch_translate_stream(dummy, model, batch_size):
        for idx, val in enumerate(translated):
            if i + idx < len(results):
                results[i + idx] = val
    
    # 填充未成功的部分
    return [r if r is not None else texts[idx] for idx, r in enumerate(results)]
