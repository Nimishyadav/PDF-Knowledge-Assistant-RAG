import unittest
from unittest.mock import patch
import ollama
from src.llm import ask_llm


class LLMTests(unittest.TestCase):
    @patch('src.llm.ollama.chat')
    def test_cpu_is_default(self, chat):
        chat.return_value = {'message': {'content': 'answer'}}
        self.assertEqual(ask_llm('question', 'context'), 'answer')
        self.assertEqual(chat.call_args.kwargs['options']['num_gpu'], 0)

    @patch('src.llm.ollama.chat')
    def test_cuda_error_retries_on_cpu(self, chat):
        chat.side_effect = [ollama.ResponseError('CUDA error: shared object initialization failed', 500), {'message': {'content': 'answer'}}]
        self.assertEqual(ask_llm('question', 'context', cpu_only=False), 'answer')
        self.assertEqual(chat.call_count, 2)
        self.assertEqual(chat.call_args.kwargs['options']['num_gpu'], 0)

    @patch('src.llm.ollama.chat')
    def test_missing_model_is_not_retried(self, chat):
        chat.side_effect = ollama.ResponseError('model not found', 404)
        with self.assertRaises(ollama.ResponseError):
            ask_llm('question', 'context', cpu_only=False)
        self.assertEqual(chat.call_count, 1)
