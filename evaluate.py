import chromadb
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()
db = chromadb.PersistentClient(path="./chroma_db")
collection = db.get_collection(name="enterprise_documents")

# Questions with known relevant text
test_cases = [
    {
        "question": "What is RAG?",
        "expected_text": "RAG combines information retrieval with language models."
    },
    {
        "question": "What are embeddings?",
        "expected_text": "Embeddings represent text as numerical vectors."
    },
    {
        "question": "What do vector databases do?",
        "expected_text": "Vector databases help find semantically similar documents."
    }
]

correct = 0

for test in test_cases:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=test["question"]
    )

    query_vector = response.data[0].embedding

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=min(2, collection.count()),
        include=["documents"]
    )

    retrieved_text = " ".join(results["documents"][0])

    found = test["expected_text"].lower() in retrieved_text.lower()

    if found:
        correct += 1

    print(f"\nQuestion: {test['question']}")
    print(f"Relevant text retrieved: {found}")

recall_at_2 = correct / len(test_cases)

print("\n-------------------------")
print(f"Retrieval Recall@2: {recall_at_2:.2%}")
print(f"Tests passed: {correct}/{len(test_cases)}")
