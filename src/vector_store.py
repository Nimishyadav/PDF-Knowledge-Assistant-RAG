import faiss
import numpy as np


class VectorStore:
    """
    Handles storing and searching
    embeddings using FAISS.
    """

    def __init__(self, dimension):
        # Create FAISS index
        self.index = faiss.IndexFlatL2(
            dimension
        )

    def add(self, embeddings):
        # Add embeddings to the index
        self.index.add(
            np.array(
                embeddings,
                dtype=np.float32
            )
        )

    def search(
        self,
        query_embedding,
        top_k=3
    ):
        # Find nearest vectors
        distances, indices = self.index.search(
            np.array(
                [query_embedding],
                dtype=np.float32
            ),
            top_k
        )

        return distances, indices

    def get_total_vectors(self):
        # Return total vectors stored
        return self.index.ntotal

    def save(self, path):
        # Save index to disk
        faiss.write_index(
            self.index,
            path
        )

    @staticmethod
    def load(path):
        # Load index from disk
        return faiss.read_index(path)