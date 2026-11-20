from pathlib import Path
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

load_dotenv()
client = OpenAI()

# Step 1: Read PDF
reader = PdfReader("documents/sample.pdf")
text = "\n".join(page.extract_text() or "" for page in reader.pages)

# Step 2: Split into chunks
def chunk_text(text, chunk_size=100, overlap=20):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])

        if end >= len(text):
            break

        start += chunk_size - overlap

    return chunks

chunks = chunk_text(text)

# Step 3: Create embeddings
embedding_response = client.embeddings.create(
    model="text-embedding-3-small",
    input=chunks
)

vectors = np.array([
    item.embedding for item in embedding_response.data
])

# Step 4: Ask a question
question = input("Ask a question about your PDF: ")

query_response = client.embeddings.create(
    model="text-embedding-3-small",
    input=question
)

query_vector = np.array(query_response.data[0].embedding)

# Step 5: Calculate cosine similarity
scores = (
    vectors @ query_vector
    / (
        np.linalg.norm(vectors, axis=1)
        * np.linalg.norm(query_vector)
    )
)

# Retrieve the two most relevant chunks
top_indices = np.argsort(scores)[::-1][:2]

context = "\n\n".join(
    f"[Chunk {i + 1}]\n{chunks[i]}"
    for i in top_indices
)

# Step 6: Generate an answer using retrieved context
response = client.responses.create(
    model="gpt-5.5",
    instructions=(
        "Answer the user's question using only the provided "
        "document context. If the answer is not in the context, "
        "say you cannot find it in the document. "
        "Cite the chunk numbers supporting your answer."
    ),
    input=f"Document context:\n{context}\n\nQuestion: {question}"
)

print("\nAI Answer:")
print(response.output_text)

print("\nRetrieved sources:")
for i in top_indices:
    print(f"Chunk {i + 1} | Similarity: {scores[i]:.4f}")
