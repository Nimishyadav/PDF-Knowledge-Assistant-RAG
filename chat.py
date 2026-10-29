"""
Command-line interface for interacting
with the indexed PDF using the RAG pipeline.
"""

import pickle

from src.embedder import Embedder
from src.vector_store import VectorStore
from src.retriever import retrieve
from src.llm import ask_llm


# Load saved FAISS index
index = VectorStore.load(
    "storage/faiss_index.bin"
)

# Load saved document chunks
with open(
    "storage/chunks.pkl",
    "rb"
) as f:

    chunks = pickle.load(f)

# Initialize embedding model
embedder = Embedder()

# Keep accepting questions until user exits
while True:

    question = input(
        "\nAsk Question: "
    )

    # Exit the application
    if question.lower() == "exit":
        break

    # Convert question into embedding
    query_embedding = (
        embedder.encode(
            [question]
        )[0]
    )

    # Retrieve relevant chunks
    retrieved_chunks = retrieve(
        query_embedding,
        index,
        chunks
    )

    # Combine retrieved chunks into context
    context = "\n".join(
        retrieved_chunks
    )

    # Generate answer using LLM
    answer = ask_llm(
        question,
        context
    )

    # Display answer
    print("\nAnswer:\n")
    print(answer)