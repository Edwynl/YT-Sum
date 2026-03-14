import httpx
import json
from transcript import transcript_to_text

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen3.5:9b"

TECH_SUMMARY_PROMPT = """你是一个专业的技术内容分析助手，擅长提炼工程师视角的学习要点。

以下是一段技术视频的英文字幕，请严格按照下方格式用中文输出，不要添加格式以外的任何内容。

---
## 【核心主题】
用 1-2 句话说明：这个视频在解决什么问题 / 教什么技术 / 演示什么工具。

## 【技术要点】
列出 3-6 个关键学习点，每条格式如下：
- **要点标题**：具体说明（用工程师能直接理解的语言，关注核心概念、实现思路、踩坑点、与其他方案的区别）

## 【部署 / 实施步骤】
如果视频涉及工具安装、项目搭建、系统配置，列出具体步骤（含命令或操作）：
1. 步骤一
2. 步骤二
...
如视频不涉及部署，此节写：无

## 【费用说明】
说明视频中提到的工具或服务：
- 是否免费 / 开源 / 收费（注明定价模型，如按量、订阅等）
- 是否有免费套餐或替代方案
如视频未提及，此节写：未提及

## 【实际收益 / 能做到什么】
完成视频内容后，可以实现哪些具体成果：
- 解决了什么实际问题
- 节省了什么时间或资源
- 比原来的方式好在哪里
---

字幕原文（英文）：
{transcript}
"""


def summarize(transcript: list[dict], model: str = DEFAULT_MODEL) -> str:
    """同步调用 Ollama 生成总结"""
    text = transcript_to_text(transcript)
    prompt = TECH_SUMMARY_PROMPT.format(transcript=text)

    try:
        response = httpx.post(
            OLLAMA_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_ctx": 8192,
                },
            },
            timeout=180,
        )
        response.raise_for_status()
        return response.json()["response"].strip()
    except httpx.ConnectError:
        raise RuntimeError("无法连接 Ollama，请确认 Ollama 已启动（ollama serve）")
    except Exception as e:
        raise RuntimeError(f"Ollama 推理失败：{e}")


async def summarize_stream(transcript: list[dict], model: str = DEFAULT_MODEL):
    """
    异步流式调用 Ollama，逐字返回生成内容。
    用于前端 SSE 实时显示。
    """
    text = transcript_to_text(transcript)
    prompt = TECH_SUMMARY_PROMPT.format(transcript=text)

    try:
        async with httpx.AsyncClient(timeout=300) as client:
            async with client.stream(
                "POST",
                OLLAMA_URL,
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": True,
                    "options": {
                        "temperature": 0.3,
                        "num_ctx": 8192,
                    },
                },
            ) as resp:
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    token = chunk.get("response", "")
                    if token:
                        yield token
                    if chunk.get("done"):
                        break
    except httpx.ConnectError:
        yield "\n\n❌ 无法连接 Ollama，请确认已运行 `ollama serve`"
    except Exception as e:
        yield f"\n\n❌ 推理出错：{e}"
