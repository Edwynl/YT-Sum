import os
import re
import json
import urllib.request
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


def _fetch_transcript_via_ytdlp(vid: str, lang: str = 'en') -> list[dict]:
    """使用 yt-dlp 获取字幕作为备用方案"""
    try:
        import yt_dlp
        import json as json_module
        
        ydl_opts = {
            'skip_download': True,
            'quiet': True,
            'no_warnings': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f'https://www.youtube.com/watch?v={vid}', download=False)
            
            # 先尝试手动字幕
            subtitles = info.get('subtitles', {})
            for l in [lang, 'en', 'en-US']:
                if l in subtitles and subtitles[l]:
                    url = subtitles[l][0].get('url')
                    if url:
                        return _download_and_parse_caption(url)
            
            # 再尝试自动字幕
            automatic_captions = info.get('automatic_captions', {})
            for l in [lang, 'en', 'en-US']:
                if l in automatic_captions and automatic_captions[l]:
                    url = automatic_captions[l][0].get('url')
                    if url:
                        return _download_and_parse_caption(url)
                            
    except Exception as e:
        print(f"yt-dlp fallback failed: {e}")
    
    return None


def _download_and_parse_caption(url: str) -> list[dict]:
    """下载并解析字幕 URL，支持 JSON 和 XML 格式"""
    import xml.etree.ElementTree as ET
    
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read().decode('utf-8')
        
        # 尝试 JSON 格式 (pb3)
        if data.strip().startswith('{'):
            return _parse_json_captions(data)
        
        # 尝试 XML 格式
        return _parse_xml_captions(data)
        
    except Exception as e:
        print(f"Caption download/parse failed: {e}")
        return None


def _parse_json_captions(json_data: str) -> list[dict]:
    """解析 YouTube JSON 格式字幕 (pb3)"""
    try:
        data = json.loads(json_data)
        transcript = []
        
        events = data.get('events', [])
        for event in events:
            segs = event.get('segs', [])
            for seg in segs:
                text = seg.get('utf8', '') or seg.get('text', '')
                # 跳过特殊控制字符
                if text and not text.startswith('\u200C'):
                    text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
                    text = text.replace('\n', ' ').replace('\xa0', ' ').strip()
                    # 清理特殊 Unicode 字符
                    text = ''.join(c for c in text if ord(c) >= 32 or c in '\n\t')
                    if text:
                        transcript.append({"text": text})
        
        return transcript if transcript else None
    except Exception as e:
        print(f"JSON parsing failed: {e}")
        return None


def _parse_xml_captions(xml_data: str) -> list[dict]:
    """解析 YouTube XML 格式字幕"""
    import xml.etree.ElementTree as ET
    try:
        root = ET.fromstring(xml_data)
        transcript = []
        for p in root.findall('.//p'):
            text = p.text or ""
            # 处理 XML 实体和换行符
            text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
            text = text.replace('\n', ' ').strip()
            if text:
                transcript.append({"text": text})
        return transcript if transcript else None
    except Exception as e:
        print(f"XML parsing failed: {e}")
        return None


def get_transcript(video_url: str, proxy: str = None) -> list[dict]:
    """
    获取视频字幕。
    优先级：手动英文 → 自动生成英文 → 任意可用语言 → yt-dlp 备用方案
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

    # 方法 1: 尝试 youtube-transcript-api
    try:
        # 优先拿英文，增加代理支持
        transcript = YouTubeTranscriptApi.get_transcript(
            vid, languages=["en", "en-US", "en-GB", "a.en"],
            proxies=proxies if proxies else None
        )
        return transcript

    except NoTranscriptFound:
        pass
    except Exception as e:
        error_msg = str(e)
        if "no element found" in error_msg.lower() or "network" in error_msg.lower():
            print(f"youtube-transcript-api failed, trying yt-dlp fallback: {e}")
            # 直接使用 yt-dlp 备用方案
            transcript = _fetch_transcript_via_ytdlp(vid)
            if transcript:
                return transcript

    # 方法 2: 尝试列出所有字幕
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(vid, proxies=proxies if proxies else None)
        first = next(iter(transcript_list))
        return first.fetch()

    except NoTranscriptFound:
        pass
    except TranscriptsDisabled:
        pass
    except Exception as e:
        print(f"list_transcripts failed: {e}")

    # 方法 3: 使用 yt-dlp 作为最终备用方案
    print("Trying yt-dlp as final fallback...")
    transcript = _fetch_transcript_via_ytdlp(vid)
    if transcript:
        return transcript

    raise RuntimeError("字幕获取失败：所有方法都未能获取到字幕（可能是由于网络环境限制或视频禁用字幕）")


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
