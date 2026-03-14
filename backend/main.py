import json
import os
import hashlib
import asyncio
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import httpx
from transcript import get_transcript, extract_video_id
from summarizer import summarize, summarize_stream
from translator import batch_translate, batch_translate_stream

# ── 初始化 ────────────────────────────────────────────────
app = FastAPI(title="YouTube Summarizer", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CACHE_DIR = Path(__file__).parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

DEFAULT_MODEL = "qwen3.5:9b"


# ── 请求 / 响应结构 ────────────────────────────────────────
class SummarizeRequest(BaseModel):
    url: str
    model: str = DEFAULT_MODEL
    translate: bool = True          # 是否翻译字幕
    use_cache: bool = True          # 是否使用本地缓存
    stream: bool = False            # 是否流式输出总结
    proxy: str | None = None        # 可选网络代理


# ── 缓存工具 ───────────────────────────────────────────────
def get_cache_path(video_id: str) -> Path:
    return CACHE_DIR / f"{video_id}.json"


def load_cache(video_id: str) -> dict | None:
    path = get_cache_path(video_id)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_cache(video_id: str, data: dict):
    path = get_cache_path(video_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ── 接口 ──────────────────────────────────────────────────

@app.get("/health")
async def health():
    """检查服务是否正常"""
    return {"status": "ok", "model": DEFAULT_MODEL}


@app.post("/summarize")
async def api_summarize(req: SummarizeRequest):
    """一键总结接口（非流式）- 同步版本保留用于旧兼容或缓存"""
    try:
        video_id = extract_video_id(req.url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if req.use_cache:
        cached = load_cache(video_id)
        if cached:
            return {**cached, "from_cache": True}

    try:
        transcript = get_transcript(req.url, proxy=req.proxy)
        summary = summarize(transcript, model=req.model)
        
        texts = [t["text"] for t in transcript]
        translations = await batch_translate(texts, model=req.model)
        bilingual = [
            {
                "text": t["text"],
                "translation": tr,
                "start": t["start"],
                "duration": t["duration"],
            }
            for t, tr in zip(transcript, translations)
        ]

        result = {
            "video_id": video_id,
            "summary": summary,
            "transcript": bilingual,
            "from_cache": False,
        }
        save_cache(video_id, result)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/summarize/stream")
async def api_summarize_stream(req: SummarizeRequest):
    """
    流式总结接口（SSE）- 并行增强版。
    """
    try:
        video_id = extract_video_id(req.url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 如果有完全缓存，直接秒回
    if req.use_cache:
        cached = load_cache(video_id)
        if cached:
            async def cached_stream():
                # 分两波发，模仿流式感但极快
                yield f"data: {json.dumps({'type': 'transcript', 'data': cached['transcript'], 'cached': True})}\n\n"
                tokens = cached['summary'].split(' ')
                for t in tokens:
                    yield f"data: {json.dumps({'type': 'token', 'text': t + ' '})}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'video_id': video_id})}\n\n"
            return StreamingResponse(cached_stream(), media_type="text/event-stream")

    async def event_stream():
        # Step 1：立即获取字幕
        try:
            transcript = get_transcript(req.url, proxy=req.proxy)
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': f'获取字幕失败: {e}'})}\n\n"
            return

        # Step 2：立即推送原始字幕，消除白屏
        yield f"data: {json.dumps({'type': 'transcript', 'data': transcript})}\n\n"

        # Step 3：并发启动【总结流】和【翻译流】
        summary_queue = asyncio.Queue()
        translate_queue = asyncio.Queue()

        async def run_summary():
            try:
                async for token in summarize_stream(transcript, model=req.model):
                    await summary_queue.put(token)
            except Exception as e:
                await summary_queue.put(f"ERROR: {e}")
            await summary_queue.put(None) # Sentinel

        async def run_translation():
            try:
                async for i, trans_batch in batch_translate_stream(transcript, model=req.model):
                    await translate_queue.put((i, trans_batch))
            except Exception as e:
                print(f"Parallel translation error: {e}")
            await translate_queue.put(None) # Sentinel

        # 启动后台任务
        s_task = asyncio.create_task(run_summary())
        t_task = asyncio.create_task(run_translation())

        full_summary = ""
        bilingual_map = {idx: t["text"] for idx, t in enumerate(transcript)} # Placeholder
        translation_results = [None] * len(transcript)

        # 混合监听两个队列并行输出
        while not (s_task.done() and t_task.done() and summary_queue.empty() and translate_queue.empty()):
            # 优先处理总结，让用户看到文字在动
            try:
                while not summary_queue.empty():
                    token = await summary_queue.get()
                    if token is None: break
                    if token.startswith("ERROR: "):
                         yield f"data: {json.dumps({'type': 'error', 'message': token})}\n\n"
                    else:
                        full_summary += token
                        yield f"data: {json.dumps({'type': 'token', 'text': token})}\n\n"
            except Exception: pass

            # 其次处理翻译包
            try:
                while not translate_queue.empty():
                    item = await translate_queue.get()
                    if item is None: break
                    idx_start, trans_list = item
                    # 发送分段更新
                    yield f"data: {json.dumps({'type': 'transcript_chunk', 'start_idx': idx_start, 'data': trans_list})}\n\n"
                    # 更新缓存数据结构
                    for offset, val in enumerate(trans_list):
                        if idx_start + offset < len(translation_results):
                            translation_results[idx_start + offset] = val
            except Exception: pass
            
            await asyncio.sleep(0.05) # 稍微让出控制权

        # 完成后整理最终状态并持久化
        final_bilingual = []
        for i, t in enumerate(transcript):
            final_bilingual.append({
                "text": t["text"],
                "translation": translation_results[i] or t["text"],
                "start": t["start"],
                "duration": t["duration"]
            })

        result = {
            "video_id": video_id,
            "summary": full_summary,
            "transcript": final_bilingual,
            "from_cache": False,
        }
        save_cache(video_id, result)
        yield f"data: {json.dumps({'type': 'done', 'video_id': video_id})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.delete("/cache/{video_id}")
async def clear_cache(video_id: str):
    path = get_cache_path(video_id)
    if path.exists():
        path.unlink()
        return {"message": f"已清除缓存：{video_id}"}
    return {"message": "缓存不存在"}


@app.get("/cache")
async def list_cache():
    files = list(CACHE_DIR.glob("*.json"))
    result = []
    for f in files:
        try:
            with open(f) as fp:
                data = json.load(fp)
            result.append({
                "video_id": data.get("video_id", f.stem),
                "summary_preview": data.get("summary", "")[:100] + "...",
            })
        except Exception: pass
    return {"cached": result}


@app.get("/models")
async def list_models():
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                data = response.json()
                models = [m["name"] for m in data.get("models", [])]
                return {"models": models}
            return {"models": [DEFAULT_MODEL], "error": "Ollama API error"}
    except Exception as e:
        return {"models": [DEFAULT_MODEL], "error": str(e)}

frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
