import json
import httpx
import os
from transcript import get_transcript
from translator import batch_translate
from summarizer import summarize

test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
model = "qwen3.5:9b"

print(f"--- Step 1: Extracting Transcript ---")
try:
    transcript = get_transcript(test_url)
    print(f"Success: Extracted {len(transcript)} lines.")
except Exception as e:
    print(f"Failed: {e}")
    exit(1)

print(f"\n--- Step 2: Testing Batch Translation ---")
try:
    test_texts = [t["text"] for t in transcript[:5]]
    translations = batch_translate(test_texts, model=model)
    print(f"Success: Translated {len(translations)} lines.")
    print(f"First translation: {translations[0]}")
except Exception as e:
    print(f"Failed: {e}")

print(f"\n--- Step 3: Testing Summary ---")
try:
    summary = summarize(transcript, model=model)
    print(f"Success: Summary length {len(summary)} chars.")
    print(f"Summary Start: {summary[:100]}...")
except Exception as e:
    print(f"Failed: {e}")
