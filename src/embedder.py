from sentence_transformers import SentenceTransformer


class Embedder:
    """
    Loads the embedding model and converts
    text into vector embeddings.
    """

    def __init__(self):
        # Load MiniLM embedding model
        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

    def encode(self, texts):
        """
        Generate embeddings for input text.
        Returns NumPy vectors that can be
        stored in FAISS.
        """

        return self.model.encode(
            texts,
            convert_to_numpy=True
        )