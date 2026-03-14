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


def batch_translate(
    texts: list[str],
    model: str = DEFAULT_TRANSLATE_MODEL,
    batch_size: int = 15,
) -> list[str]:
    """
    批量翻译字幕文本，每批 batch_size 条。
    如果某批翻译失败，返回原文（英文），不中断整体流程。
    """
    results = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        prompt = TRANSLATE_PROMPT.format(
            items=json.dumps(batch, ensure_ascii=False)
        )

        try:
            resp = httpx.post(
                OLLAMA_URL,
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,   # 翻译要稳定
                        "num_ctx": 4096,
                    },
                },
                timeout=120,
            )
            resp.raise_for_status()
            raw = resp.json()["response"]
            cleaned = _clean_json_response(raw)
            translated = json.loads(cleaned)

            # 确保数量一致
            if isinstance(translated, list) and len(translated) == len(batch):
                results.extend(translated)
            else:
                results.extend(batch)  # 数量不对，保留原文

        except (json.JSONDecodeError, KeyError):
            results.extend(batch)  # 解析失败，保留原文
        except Exception:
            results.extend(batch)

    return results
