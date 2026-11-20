

import re
import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

load_dotenv()
openai_client = OpenAI()

# Create a persistent local database
db = chromadb.PersistentClient(path="./chroma_db")

collection = db.get_or_create_collection(
    name="enterprise_documents",
    metadata={"hnsw:space": "cosine"}
)

# Read PDF
reader = PdfReader("documents/sample.pdf")

text = "\n".join(
    page.extract_text() or ""
    for page in reader.pages
)

# Split into sentences
chunks = [
    sentence.strip()
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip())
    if sentence.strip()
]

# Generate embeddings
response = openai_client.embeddings.create(
    model="text-embedding-3-small",
    input=chunks
)

embeddings = [
    item.embedding
    for item in response.data
]

# Store embeddings and text
collection.upsert(
    ids=[f"sample_chunk_{i}" for i in range(len(chunks))],
    documents=chunks,
    embeddings=embeddings,
    metadatas=[
        {"source": "sample.pdf", "chunk": i + 1}
        for i in range(len(chunks))
    ]
)

print("Documents stored successfully!")
print("Total chunks:", collection.count())
