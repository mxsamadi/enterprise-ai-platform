
import re
from pypdf import PdfReader

def smart_chunk_text(text, max_chars=300):
    # Split text into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_chars:
            current_chunk += " " + sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


reader = PdfReader("documents/sample.pdf")

text = "\n".join(
    page.extract_text() or ""
    for page in reader.pages
)

chunks = smart_chunk_text(text, max_chars=100)

print(f"Total chunks: {len(chunks)}")

for i, chunk in enumerate(chunks, start=1):
    print(f"\n--- Chunk {i} ---")
    print(chunk)

