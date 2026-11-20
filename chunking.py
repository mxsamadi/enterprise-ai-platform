from pypdf import PdfReader

reader = PdfReader("documents/sample.pdf")

full_text = ""

for page in reader.pages:
    full_text += (page.extract_text() or "") + "\n"

def chunk_text(text, chunk_size=100, overlap=20):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)

        if end >= len(text):
            break

        start += chunk_size - overlap

    return chunks

chunks = chunk_text(full_text)

print(f"Total chunks: {len(chunks)}")

for i, chunk in enumerate(chunks, start=1):
    print(f"\n--- Chunk {i} ---")
    print(chunk)
