"""Quick LLM smoke test."""
import os, sys, time
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from backend.app.generation.llm import get_client
from backend.app.config import LLM_MODEL, LLM_BASE_URL

print(f"Model: {LLM_MODEL}")
print(f"Base URL: {LLM_BASE_URL}")

client = get_client()
print("Client created. Calling API...")

t0 = time.time()
try:
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": "Say hello in one word."}],
        max_tokens=10,
        temperature=0.1,
    )
    elapsed = (time.time() - t0) * 1000
    print(f"Response: {response.choices[0].message.content}")
    print(f"Latency: {elapsed:.0f}ms")
except Exception as e:
    elapsed = (time.time() - t0) * 1000
    print(f"ERROR ({elapsed:.0f}ms): {type(e).__name__}: {e}")
