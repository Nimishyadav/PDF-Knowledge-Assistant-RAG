import unittest
import numpy as np
from src.chunker import chunk_text
from src.retriever import retrieve
from src.vector_store import VectorStore


class PipelineTests(unittest.TestCase):
    def test_invalid_overlap_does_not_loop(self):
        for size, overlap in [(0, 0), (10, 10), (10, -1)]:
            with self.assertRaises(ValueError):
                chunk_text('document', size, overlap)

    def test_chunks_preserve_overlap(self):
        self.assertEqual(chunk_text('abcdefghij', 6, 2), ['abcdef', 'efghij', 'ij'])

    def test_top_k_larger_than_corpus_has_no_duplicate_last_chunk(self):
        store = VectorStore(2)
        store.add(np.array([[0, 0], [10, 10]], dtype=np.float32))
        self.assertEqual(retrieve([0, 0], store.index, ['near', 'far'], 10), ['near', 'far'])

    def test_empty_index(self):
        self.assertEqual(retrieve([0, 0], VectorStore(2).index, [], 3), [])


if __name__ == '__main__':
    unittest.main()
