import os
import re
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)


def extract_video_id(url: str) -> str:
    """从各种格式的 YouTube URL 中提取 Video ID"""
    patterns = [
        r"(?:v=)([a-zA-Z0-9_-]{11})",
        r"(?:youtu\.be/)([a-zA-Z0-9_-]{11})",
        r"(?:shorts/)([a-zA-Z0-9_-]{11})",
        r"(?:embed/)([a-zA-Z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    # 如果整个字符串就是 11 位 ID
    if re.fullmatch(r"[a-zA-Z0-9_-]{11}", url.strip()):
        return url.strip()
    raise ValueError(f"无法从 URL 中解析出 Video ID：{url}")


def get_transcript(video_url: str, proxy: str = None) -> list[dict]:
    """
    获取视频字幕。
    优先级：手动英文 → 自动生成英文 → 任意可用语言
    """
    vid = extract_video_id(video_url)
    
    # 构建代理字典
    proxies = {}
    if proxy:
        # 如果用户传了具体代理字符串，优先使用
        proxies = {"http": proxy, "https": proxy}
    else:
        # 否则尝试从环境变量获取
        http_proxy = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
        https_proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
        if https_proxy: proxies["https"] = https_proxy
        if http_proxy: proxies["http"] = http_proxy
    
    # 强制让 httpx 不在本地回环使用代理
    os.environ["NO_PROXY"] = "localhost,127.0.0.1"

    if proxies:
        print(f"DEBUG: Using proxies for YouTube: {proxies}")

    try:
        # 优先拿英文，增加代理支持
        transcript = YouTubeTranscriptApi.get_transcript(
            vid, languages=["en", "en-US", "en-GB", "a.en"],
            proxies=proxies if proxies else None
        )
        return transcript

    except NoTranscriptFound:
        pass

    try:
        # Fallback，增加代理支持
        transcript_list = YouTubeTranscriptApi.list_transcripts(vid, proxies=proxies if proxies else None)
        first = next(iter(transcript_list))
        return first.fetch()

    except TranscriptsDisabled:
        raise RuntimeError("该视频已禁用字幕，无法提取。")
    except VideoUnavailable:
        raise RuntimeError("视频不可用，请检查链接是否正确。")
    except Exception as e:
        error_msg = str(e)
        if "no element found" in error_msg.lower():
            raise RuntimeError("字幕 API 解析失败（可能是由于网络环境导致无法连接 YouTube API）")
        raise RuntimeError(f"字幕获取失败：{error_msg}")


def transcript_to_text(transcript: list[dict], max_chars: int = 10000) -> str:
    """
    将字幕列表拼接为纯文本，用于喂给 AI。
    超长视频截取前 max_chars 字符（约 40-50 分钟内容）。
    """
    full_text = " ".join(
        item["text"].replace("\n", " ").strip()
        for item in transcript
    )
    if len(full_text) > max_chars:
        # 截到最近的句号，避免句子截断
        truncated = full_text[:max_chars]
        last_period = max(truncated.rfind("."), truncated.rfind("?"), truncated.rfind("!"))
        if last_period > max_chars * 0.8:
            truncated = truncated[:last_period + 1]
        return truncated + "\n\n[注：字幕过长，以上为前段内容]"
    return full_text
