"""
Builds the vector index from the PDF.

Flow:
PDF -> Text -> Chunks -> Embeddings
-> FAISS Index -> Save to Disk
"""

from src.pdf_reader import read_pdf
from src.chunker import chunk_text
from src.embedder import Embedder
from src.vector_store import VectorStore

import pickle


# PDF to be indexed
PDF_PATH = "data/sample.pdf"

# Extract text from PDF
text = read_pdf(PDF_PATH)

# Split document into chunks
chunks = chunk_text(text)

# Load embedding model
embedder = Embedder()

# Generate embeddings for all chunks
embeddings = embedder.encode(chunks)

# Get embedding dimension
dimension = embeddings.shape[1]

# Create FAISS index
vector_store = VectorStore(dimension)

# Add embeddings to index
vector_store.add(embeddings)

# Save FAISS index
vector_store.save(
    "storage/faiss_index.bin"
)

# Save chunks separately
with open(
    "storage/chunks.pkl",
    "wb"
) as f:

    pickle.dump(
        chunks,
        f
    )

print("Indexing Complete")