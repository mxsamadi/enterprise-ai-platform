import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Enterprise AI Document Intelligence",
    page_icon="📄",
    layout="wide"
)

st.title("Enterprise AI Document Intelligence Platform")
st.write(
    "Upload PDF documents and ask questions using "
    "Retrieval-Augmented Generation (RAG)."
)

with st.sidebar:
    st.header("About")
    st.write("Built with Python, FastAPI, OpenAI, ChromaDB, and Streamlit.")
    st.write("Answers are generated using retrieved document context.")

st.header("1. Upload a PDF")

uploaded_file = st.file_uploader(
    "Choose a PDF document",
    type=["pdf"]
)

if uploaded_file is not None:
    if st.button("Upload and Index Document"):
        with st.spinner("Processing document..."):
            try:
                response = requests.post(
                    f"{API_URL}/upload",
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            "application/pdf"
                        )
                    },
                    timeout=120
                )
                response.raise_for_status()
                result = response.json()

                st.success(result["message"])
                st.write("Filename:", result["filename"])
                st.write("Chunks stored:", result["chunks_stored"])

            except requests.exceptions.RequestException as error:
                st.error(f"Upload failed: {error}")

st.divider()

st.header("2. Ask a Question")

question = st.text_input(
    "Enter your question",
    placeholder="What is Retrieval-Augmented Generation?"
)

if st.button("Ask AI"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching documents and generating an answer..."):
            try:
                response = requests.post(
                    f"{API_URL}/ask",
                    json={"question": question},
                    timeout=120
                )
                response.raise_for_status()
                result = response.json()

                st.subheader("AI Answer")
                st.write(result["answer"])

                st.subheader("Retrieved Sources")

                for source in result["sources"]:
                    st.write(
                        f"**{source['source']}** | "
                        f"Chunk {source['chunk']} | "
                        f"Similarity: {source['similarity']}"
                    )

            except requests.exceptions.RequestException as error:
                st.error(f"Question failed: {error}")
