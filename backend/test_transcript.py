from transcript import get_transcript
import sys

test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
if len(sys.argv) > 1:
    test_url = sys.argv[1]

print(f"Testing transcript extraction for: {test_url}")
try:
    transcript = get_transcript(test_url)
    print(f"Success! Extracted {len(transcript)} lines.")
except Exception as e:
    print(f"Failed: {e}")
