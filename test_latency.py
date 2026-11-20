import json
import time
import urllib.request

url = "http://127.0.0.1:8000/ask"

questions = [
    "What is RAG?",
    "What are embeddings?",
    "What do vector databases do?"
]

times = []

for question in questions:
    payload = json.dumps({"question": question}).encode("utf-8")

    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    start = time.perf_counter()

    with urllib.request.urlopen(request, timeout=120) as response:
        result = json.loads(response.read().decode("utf-8"))

    elapsed = time.perf_counter() - start
    times.append(elapsed)

    print(f"\nQuestion: {question}")
    print(f"Response time: {elapsed:.2f} seconds")
    print(f"Answer: {result['answer']}")

print("\n-------------------------")
print(f"Average latency: {sum(times) / len(times):.2f} seconds")
print(f"Maximum latency: {max(times):.2f} seconds")
