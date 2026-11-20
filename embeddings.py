from openai import OpenAI
from dotenv import load_dotenv
from chunking import chunk_text
from pypdf import PdfReader
import numpy as np

load_dotenv()
client = OpenAI()

reader = PdfReader("documents/sample.pdf")

full_text = "\n".join(
    page.extract_text() or ""
    for page in reader.pages
)

chunks = chunk_text(full_text)

response = client.embeddings.create(
    model="text-embedding-3-small",
    input=chunks
)

vectors = np.array([
    item.embedding
    for item in response.data
])

print("Number of chunks:", len(chunks))
print("Embedding matrix shape:", vectors.shape)

question = "What is RAG?"

query_response = client.embeddings.create(
    model="text-embedding-3-small",
    input=question
)

query_vector = np.array(
    query_response.data[0].embedding
)

# Cosine similarity
scores = (
    vectors @ query_vector
    / (
        np.linalg.norm(vectors, axis=1)
        * np.linalg.norm(query_vector)
    )
)

best_index = int(np.argmax(scores))

print("\nQuestion:", question)
print("Best matching chunk:", best_index + 1)
print("Similarity:", round(float(scores[best_index]), 4))
print("\nRetrieved text:")
print(chunks[best_index])
