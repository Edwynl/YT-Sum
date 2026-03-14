import httpx
import json
import re
import asyncio

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_TRANSLATE_MODEL = "qwen3.5:9b"

# 极简协议 Prompt：增加强制中文输出指令
TRANSLATE_PROMPT = """You are a professional translator. 
Task: Translate the following English subtitles into Chinese.

Rules:
1. Translate line by line.
2. Output ONLY the translated Chinese text. 
3. One line of input = One line of output. 
4. DO NOT repeat the original English. 
5. DO NOT provide any English explanations.
6. Keep technical terms like API, LLM, Docker in English.

Input Text:
{items}

Output (Chinese ONLY):"""


def _is_mostly_chinese(text: str) -> bool:
    """
    检查文本是否主要是中文。如果不包含中文字符且长度较长，则认为翻译失败。
    """
    if not text: return False
    # 统计中文字符
    zh_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    return zh_chars > 0


def _clean_text_response(raw: str, expected_count: int) -> list[str]:
    """
    清理纯文本响应，确保行数对齐。
    """
    # 移除 Markdown 代码块
    raw = re.sub(r"```[a-z]*", "", raw).strip()
    raw = raw.replace('\r', '')
    lines = [line.strip() for line in raw.split('\n') if line.strip()]
    
    # 过滤掉可能的回复性短语 (如 "Here is the translation:" 等)
    filtered = []
    for l in lines:
        if "translation" in l.lower() and ":" in l and len(l) < 30: continue
        filtered.append(l)
    
    lines = filtered

    if len(lines) == expected_count:
        return lines
    
    if len(lines) > expected_count:
        return lines[:expected_count]
    else:
        return lines + [""] * (expected_count - len(lines))


async def _translate_batch_task(client: httpx.AsyncClient, batch: list[str], start_idx: int, model: str, retry_count: int = 2):
    """
    内部原子任务：翻译一个批次并进行校验重试
    """
    prompt = TRANSLATE_PROMPT.format(items="\n".join(batch))
    
    for attempt in range(retry_count + 1):
        try:
            resp = await client.post(
                OLLAMA_URL,
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.0,
                        "num_ctx": 4096,
                        "stop": ["Input Text:", "Output:"]
                    },
                },
                timeout=30 # 稍微延长超时以应对 9b 模型
            )
            resp.raise_for_status()
            raw = resp.json().get("response", "")
            
            translated_lines = _clean_text_response(raw, len(batch))
            
            sample_text = "".join(translated_lines)
            if _is_mostly_chinese(sample_text):
                return start_idx, translated_lines
            
            print(f"Quality check failed for batch {start_idx}, retry {attempt+1}...")
        except Exception as e:
            print(f"Task attempt {attempt} for batch {start_idx} failed: {e}")
            
    # 最终失败返回原文
    return start_idx, batch


async def batch_translate_stream(
    transcript: list[dict],
    model: str = DEFAULT_TRANSLATE_MODEL,
    batch_size: int = 25, # 增加批次大小，减少请求次数
    concurrency: int = 4, # 增加并发度
):
    """
    极致并行流式翻译：使用 as_completed 只要有一个翻译完就立刻推送
    """
    texts = [t["text"] for t in transcript]
    total = len(texts)
    
    async with httpx.AsyncClient() as client:
        # 分组任务
        for i in range(0, total, batch_size * concurrency):
            tasks = []
            for j in range(concurrency):
                start_idx = i + (j * batch_size)
                if start_idx >= total: break
                
                batch = texts[start_idx : start_idx + batch_size]
                tasks.append(_translate_batch_task(client, batch, start_idx, model))
            
            if not tasks: break
            
            # 使用 as_completed 特性：哪个先好发哪个
            for coro in asyncio.as_completed(tasks):
                start_pos, translated_batch = await coro
                
                # 对齐处理
                expected_len = min(batch_size, total - start_pos)
                final_batch = []
                for k in range(expected_len):
                    t_line = translated_batch[k] if k < len(translated_batch) else texts[start_pos + k]
                    final_batch.append(t_line if t_line else texts[start_pos + k])
                
                yield start_pos, final_batch


async def batch_translate(
    texts: list[str],
    model: str = DEFAULT_TRANSLATE_MODEL,
    batch_size: int = 25,
) -> list[str]:
    """适配旧接口"""
    results = [None] * len(texts)
    dummy = [{"text": t} for t in texts]
    async for i, translated in batch_translate_stream(dummy, model, batch_size):
        for idx, val in enumerate(translated):
            if i + idx < len(results):
                results[i + idx] = val
    
    return [r if r is not None else texts[idx] for idx, r in enumerate(results)]
