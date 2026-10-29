import numpy as np


def retrieve(
    query_embedding,
    index,
    chunks,
    top_k=3
):
    """
    Retrieve the most relevant chunks
    for a given query.
    """

    if top_k < 1:
        raise ValueError("top_k must be positive")
    if not chunks or index.ntotal == 0:
        return []
    # Search similar vectors in FAISS
    distances, indices = index.search(
        np.array(
            [query_embedding],
            dtype=np.float32
        ),
        min(top_k, index.ntotal, len(chunks))
    )

    # Store retrieved text chunks
    retrieved = []

    # Convert indices into actual text
    for idx in indices[0]:
        if 0 <= idx < len(chunks):
            retrieved.append(chunks[idx])

    return retrieved
