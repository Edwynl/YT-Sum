import httpx
import json
import re

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_TRANSLATE_MODEL = "qwen3.5:9b"

# 极简协议 Prompt：直接逐行输出翻译，不带任何废话
TRANSLATE_PROMPT = """You are a professional translator. Translate the following subtitles into Chinese.
Rules:
1. Translate line by line.
2. Output ONLY the translated Chinese text. 
3. One line of input = One line of output. 
4. DO NOT add any explanations, notes, or bullet points.
5. Keep technical terms like API, LLM, Docker in English.

Input Text (to be translated):
{items}

Output (Chinese only):"""


def _clean_text_response(raw: str, expected_count: int) -> list[str]:
    """
    清理纯文本响应，确保行数对齐。
    """
    # 移除 Markdown 代码块
    raw = re.sub(r"```[a-z]*", "", raw).strip()
    lines = [line.strip() for line in raw.split('\n') if line.strip()]
    
    # 如果行数正好对齐，直接返回
    if len(lines) == expected_count:
        return lines
    
    # 如果行数不对，尝试根据某种特征进行二次对齐或直接降级
    if len(lines) > expected_count:
        return lines[:expected_count]
    else:
        # 行数不足，用原文填充缺失部分
        return lines + [""] * (expected_count - len(lines))


async def batch_translate_stream(
    transcript: list[dict],
    model: str = DEFAULT_TRANSLATE_MODEL,
    batch_size: int = 20,
):
    """
    极速流式翻译：使用纯文本协议
    """
    texts = [t["text"] for t in transcript]
    
    async with httpx.AsyncClient(timeout=30) as client: # 缩短超时，追求效率
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            prompt = TRANSLATE_PROMPT.format(
                items="\n".join(batch)
            )

            try:
                resp = await client.post(
                    OLLAMA_URL,
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.0, # 翻译最稳定
                            "num_ctx": 4096,
                            "stop": ["Input Text:", "Output:"] # 强力止损
                        },
                    },
                )
                resp.raise_for_status()
                raw = resp.json().get("response", "")
                
                translated_lines = _clean_text_response(raw, len(batch))
                
                # 最后的兜底对齐：如果某行翻译完全为空，用原文
                final_batch = []
                for idx, t_line in enumerate(translated_lines):
                    final_batch.append(t_line if t_line else batch[idx])

                yield i, final_batch
                
            except Exception as e:
                print(f"Translation boost failed at batch {i}: {e}")
                yield i, batch


async def batch_translate(
    texts: list[str],
    model: str = DEFAULT_TRANSLATE_MODEL,
    batch_size: int = 20,
) -> list[str]:
    """适配旧接口"""
    results = [None] * len(texts)
    dummy = [{"text": t} for t in texts]
    async for i, translated in batch_translate_stream(dummy, model, batch_size):
        for idx, val in enumerate(translated):
            if i + idx < len(results):
                results[i + idx] = val
    
    return [r if r is not None else texts[idx] for idx, r in enumerate(results)]
