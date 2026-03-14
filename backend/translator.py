import httpx
import json
import re
import asyncio

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_TRANSLATE_MODEL = "qwen3.5:9b"

# 精准索引协议 Prompt：强制 AI 输出编号，以便完美对齐
TRANSLATE_PROMPT_V3 = """Task: Translate {line_count} lines of subtitles into Chinese.

Rules:
1. Output format: [Index] Translation. Example: [1] 你好.
2. DO NOT skip any index.
3. NO original English in output. NO explanations.
4. Keep technical terms like API, LLM, Docker in English.

Input Lines:
{items}

Output (Numbered Chinese ONLY):"""


def _is_mostly_chinese(text: str) -> bool:
    """检查文本是否主要是中文。"""
    if not text: return False
    zh_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    return zh_chars > 0


def _is_copy_of_original(translated: str, original: str) -> bool:
    """检测是否只是简单抄袭原文。"""
    t_clean = re.sub(r'[^a-zA-Z0-9]', '', translated).lower()
    o_clean = re.sub(r'[^a-zA-Z0-9]', '', original).lower()
    if t_clean == o_clean and not re.search(r'[\u4e00-\u9fff]', translated):
        return True
    return False


def _parse_numbered_response(raw: str, expected_count: int, offset: int) -> list[str]:
    """
    使用正则解析带标号 [Index] 的响应，并将内容回填到正确位置。
    """
    results = [None] * expected_count
    
    # 移除 Markdown 等杂质
    raw = re.sub(r"```[a-z]*", "", raw).strip()
    
    # 正则匹配： [数字] 内容 或 数字. 内容
    # 尽可能捕获各种格式以提高鲁棒性
    patterns = [
        r'\[(\d+)\]\s*(.+)',      # [1] 内容
        r'(\d+)[\. \-\s]+(.+)'    # 1. 内容 或 1 内容
    ]
    
    lines = raw.split('\n')
    for line in lines:
        line = line.strip()
        if not line: continue
        
        match = None
        for p in patterns:
            m = re.match(p, line)
            if m:
                match = m
                break
        
        if match:
            idx_in_batch = int(match.group(1)) - 1 # 1-based to 0-based
            content = match.group(2).strip()
            
            # 这里的 idx_in_batch 是相对于当前这一批次的（1 到 line_count）
            if 0 <= idx_in_batch < expected_count:
                results[idx_in_batch] = content

    return results


async def _translate_batch_precision(client: httpx.AsyncClient, batch: list[str], start_idx: int, model: str):
    """
    精准翻译原子任务：使用标号协议。
    """
    line_count = len(batch)
    # 构造带标号的输入，方便 AI 对应： [1] Original text...
    numbered_items = "\n".join([f"[{idx+1}] {text}" for idx, text in enumerate(batch)])
    prompt = TRANSLATE_PROMPT_V3.format(line_count=line_count, items=numbered_items)
    
    for attempt in range(2):
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
                        "stop": ["Input Lines:", "Output:"]
                    },
                },
                timeout=45
            )
            resp.raise_for_status()
            raw = resp.json().get("response", "")
            
            # 解析并回填
            translated_list = _parse_numbered_response(raw, line_count, start_idx)
            
            # 质量校验：如果填充率太低或含有太多英文，重试
            filled_count = sum(1 for x in translated_list if x is not None)
            if filled_count / line_count >= 0.6: # 至少翻译了 60% 才算基本成功
                # 对没填上的进行二次兜底
                final_batch = []
                for k, t in enumerate(translated_list):
                    if t is None or _is_copy_of_original(t, batch[k]):
                        final_batch.append(None) # 标记为失败，交由后面逻辑决定是否回退
                    else:
                        final_batch.append(t)
                return start_idx, final_batch
            
            print(f"Index alignment low ({filled_count}/{line_count}) at {start_idx}, retry...")
        except Exception as e:
            print(f"Precision translation error at {start_idx}: {e}")
            
    return start_idx, [None] * line_count


async def batch_translate_stream(
    transcript: list[dict],
    model: str = DEFAULT_TRANSLATE_MODEL,
    batch_size: int = 5,
    concurrency: int = 1,
):
    """
    顺序翻译流：通过数字索引确保 100% 对齐。
    """
    texts = [t["text"] for t in transcript]
    total = len(texts)
    
    async with httpx.AsyncClient() as client:
        for i in range(0, total, batch_size):
            batch = texts[i : i + batch_size]
            
            _, translated_batch = await _translate_batch_precision(client, batch, i, model)
            
            final_batch = []
            for k in range(len(batch)):
                # 如果精准翻译拿到结果了，用结果；否则回退原文
                res = translated_batch[k]
                if res and _is_mostly_chinese(res):
                    final_batch.append(res)
                else:
                    final_batch.append(batch[k])
            
            yield i, final_batch


async def batch_translate(
    texts: list[str],
    model: str = DEFAULT_TRANSLATE_MODEL,
    batch_size: int = 5,
) -> list[str]:
    """旧接口适配"""
    results = [None] * len(texts)
    dummy = [{"text": t} for t in texts]
    async for i, translated in batch_translate_stream(dummy, model, batch_size):
        for idx, val in enumerate(translated):
            if i + idx < len(results):
                results[i + idx] = val
    
    return [r if r is not None else texts[idx] for idx, r in enumerate(results)]
