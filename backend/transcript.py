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
    """解析 YouTube JSON 格式字幕 (pb3)，合并细碎单词为长句"""
    try:
        data = json.loads(json_data)
        raw_chunks = []
        events = data.get('events', [])
        
        for event in events:
            start_ms = event.get('tStartMs', 0)
            segs = event.get('segs', [])
            chunk_text = ""
            for seg in segs:
                text = seg.get('utf8', '') or seg.get('text', '')
                if text and not text.startswith('\u200C'):
                    chunk_text += text
            
            # 清理文本
            chunk_text = chunk_text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
            chunk_text = chunk_text.replace('\n', ' ').replace('\xa0', ' ').strip()
            chunk_text = ''.join(c for c in chunk_text if ord(c) >= 32)
            
            if chunk_text:
                raw_chunks.append({
                    "text": chunk_text,
                    "start": start_ms / 1000.0,
                    "duration": event.get('dDurationMs', 0) / 1000.0
                })

        if not raw_chunks: return None

        # 合并策略：按时间间隔和标点符号合并
        merged = []
        if not raw_chunks: return None
        
        current = raw_chunks[0]
        for next_chunk in raw_chunks[1:]:
            # 如果两个片段间隔很短（小于1.5秒）或者当前片段不以标点结尾
            time_gap = next_chunk['start'] - (current['start'] + current['duration'])
            
            # 判断是否以结束标点结尾
            ends_with_punc = any(current['text'].endswith(p) for p in ('.', '?', '!', '。', '？', '！'))
            
            # 质量加固：如果合并后太长（超过 80 个单词或 500 字符），则停止合并，防止 AI 后面翻译不动
            is_too_long = len(current['text'].split()) > 80 or len(current['text']) > 500
            
            if ((time_gap < 1.5 and not ends_with_punc) or len(current['text'].split()) < 8) and not is_too_long:
                current['text'] += " " + next_chunk['text']
                current['duration'] = (next_chunk['start'] + next_chunk['duration']) - current['start']
            else:
                merged.append(current)
                current = next_chunk
        merged.append(current)
        
        return merged
    except Exception as e:
        print(f"JSON parsing failed: {e}")
        return None


def _parse_xml_captions(xml_data: str) -> list[dict]:
    """解析 YouTube XML 格式字幕，支持时间戳"""
    import xml.etree.ElementTree as ET
    try:
        root = ET.fromstring(xml_data)
        transcript = []
        for p in root.findall('.//p'):
            text = p.text or ""
            # 有些 ASR 数据在 <s> 标签里
            if not text:
                text = "".join(s.text for s in p.findall('s') if s.text)
            
            text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
            text = text.replace('\n', ' ').strip()
            
            if text:
                # 获取时间戳属性（毫秒或秒）
                t = float(p.get('t', 0))
                # 如果数值很大，通常是毫秒
                start = t / 1000.0 if t > 10000 else t 
                transcript.append({
                    "text": text,
                    "start": start,
                    "duration": float(p.get('d', 0)) / 1000.0
                })
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
