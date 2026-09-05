"""Direct OpenRouter diagnostic script; only runs when explicitly invoked."""
import base64
import os

import cv2
import numpy as np
import requests

API_KEY = os.getenv("OPENROUTER_API_KEY")
URL = "https://openrouter.ai/api/v1/chat/completions"


def main() -> int:
    if not API_KEY:
        print("SKIPPED: OPENROUTER_API_KEY is not configured")
        return 0

    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    cv2.putText(frame, "TEST", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    _, buf = cv2.imencode(".png", frame)
    b64 = base64.b64encode(buf).decode("utf-8")

    for model in (
        "google/gemini-flash-1.5",
        "google/gemini-1.5-flash",
        "openai/gpt-4o-mini",
    ):
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "Say hello."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]}],
            "max_tokens": 50,
        }
        headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
        print(f"\nTrying model: {model}")
        try:
            response = requests.post(URL, headers=headers, json=payload, timeout=30)
            print(f"  Status: {response.status_code}")
            if response.status_code == 200:
                print(f"  Response: {response.json()['choices'][0]['message']['content']}")
                return 0
            print(f"  Error body: {response.text[:300]}")
        except requests.RequestException as error:
            print(f"  Exception: {error}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
