
import chromadb
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

db = chromadb.PersistentClient(path="./chroma_db")

collection = db.get_collection(
    name="enterprise_documents"
)

question = input("Ask a question: ")

# Embed only the new question
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=question
)

query_embedding = response.data[0].embedding

# Search stored document vectors
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=min(2, collection.count()),
    include=["documents", "metadatas", "distances"]
)

for document, metadata, distance in zip(
    results["documents"][0],
    results["metadatas"][0],
    results["distances"][0]
):
    print("\nSource:", metadata["source"])
    print("Chunk:", metadata["chunk"])
    print("Cosine similarity:", round(1 - distance, 4))
    print("Text:", document)
