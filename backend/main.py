import json
import os
import hashlib
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import httpx
from transcript import get_transcript, extract_video_id
from summarizer import summarize, summarize_stream
from translator import batch_translate

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
    """
    一键总结接口（非流式）。
    返回：{ video_id, summary, transcript: [{text, translation, start, duration}] }
    """
    try:
        video_id = extract_video_id(req.url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 命中缓存直接返回
    if req.use_cache:
        cached = load_cache(video_id)
        if cached:
            return {**cached, "from_cache": True}

    # 获取字幕
    try:
        transcript = get_transcript(req.url, proxy=req.proxy)
    except Exception as e:
        print(f"Error during summarization process: {e}")
        raise HTTPException(status_code=422, detail=f"处理视频失败: {str(e)}")

    # AI 总结
    try:
        summary = summarize(transcript, model=req.model)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # 翻译字幕（可选）
    bilingual = transcript
    if req.translate:
        texts = [t["text"] for t in transcript]
        translations = batch_translate(texts, model=req.model)
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


@app.post("/summarize/stream")
async def api_summarize_stream(req: SummarizeRequest):
    """
    流式总结接口（SSE）。
    先推送字幕，再流式推送 AI 总结内容。
    格式：data: {"type": "transcript"|"token"|"done"|"error", ...}
    """
    try:
        video_id = extract_video_id(req.url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    async def event_stream():
        # Step 1：获取字幕
        try:
            transcript = get_transcript(req.url, proxy=req.proxy)
        except Exception as e:
            print(f"SSE Error during transcript fetch: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            return

        # Step 2：翻译字幕（批量，先推送）
        texts = [t["text"] for t in transcript]
        translations = batch_translate(texts, model=req.model) if req.translate else texts
        bilingual = [
            {
                "text": t["text"],
                "translation": tr,
                "start": t["start"],
                "duration": t["duration"],
            }
            for t, tr in zip(transcript, translations)
        ]
        yield f"data: {json.dumps({'type': 'transcript', 'data': bilingual})}\n\n"

        # Step 3：流式生成总结
        full_summary = ""
        async for token in summarize_stream(transcript, model=req.model):
            full_summary += token
            yield f"data: {json.dumps({'type': 'token', 'text': token})}\n\n"

        # Step 4：完成，写缓存
        result = {
            "video_id": video_id,
            "summary": full_summary,
            "transcript": bilingual,
            "from_cache": False,
        }
        save_cache(video_id, result)
        yield f"data: {json.dumps({'type': 'done', 'video_id': video_id})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.delete("/cache/{video_id}")
async def clear_cache(video_id: str):
    """清除指定视频的缓存"""
    path = get_cache_path(video_id)
    if path.exists():
        path.unlink()
        return {"message": f"已清除缓存：{video_id}"}
    return {"message": "缓存不存在"}


@app.get("/cache")
async def list_cache():
    """列出所有已缓存的视频"""
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
        except Exception:
            pass
    return {"cached": result}


@app.get("/models")
async def list_models():
    """获取本地 Ollama 模型列表"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                data = response.json()
                models = [m["name"] for m in data.get("models", [])]
                return {"models": models}
            else:
                return {"models": [DEFAULT_MODEL], "error": "Ollama API error"}
    except Exception as e:
        return {"models": [DEFAULT_MODEL], "error": str(e)}


# ── 静态文件（前端）────────────────────────────────────────
frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
