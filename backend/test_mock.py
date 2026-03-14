import json
import httpx
from summarizer import summarize
from translator import batch_translate

# 模拟一段关于 Python 高级特性的技术视频字幕
mock_transcript = [
    {"text": "Welcome to our deep dive into Python's advanced concurrency decorators.", "start": 1.0, "duration": 4.5},
    {"text": "Today we will see how to leverage AsyncIO with custom wrappers to build high-performance APIs.", "start": 5.5, "duration": 5.0},
    {"text": "You can find the source code on our GitHub repo, it's completely open source.", "start": 10.5, "duration": 4.0},
    {"text": "First, we create a decorator that yields a future and handles exceptions gracefully.", "start": 15.0, "duration": 5.5}
]

model = "qwen3.5:9b"

print(f"--- Step 1: Mocking Transcript ---")
print(f"Using {len(mock_transcript)} lines of mock technical content.")

print(f"\n--- Step 2: Testing Batch Translation ---")
try:
    test_texts = [t["text"] for t in mock_transcript]
    translations = batch_translate(test_texts, model=model)
    print(f"Success: AI Translation completed.")
    for i, t in enumerate(translations):
        print(f"[{i}] {t}")
except Exception as e:
    print(f"Failed: {e}")

print(f"\n--- Step 3: Testing AI Summary ---")
try:
    summary = summarize(mock_transcript, model=model)
    print(f"\nSuccess! Final AI Summary Output:\n")
    print("-" * 50)
    print(summary)
    print("-" * 50)
except Exception as e:
    print(f"Failed: {e}")
