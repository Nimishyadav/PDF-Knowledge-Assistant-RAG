"""UI smoke checks; no model downloads or Ollama calls required."""
import unittest
from pathlib import Path
import inspect
from streamlit.testing.v1 import AppTest


class AppTests(unittest.TestCase):
    def test_rerun_refreshes_stale_llm_function(self):
        from src import llm
        llm.ask_llm = lambda question, context: "old implementation"
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=30).run()
        self.assertEqual(len(app.exception), 0)
        self.assertIn('cpu_only', inspect.signature(llm.ask_llm).parameters)

    def test_initial_screen_and_clear(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=30).run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(app.chat_input[0].disabled)
        self.assertEqual(app.title[0].value, '📚 Local PDF Knowledge Assistant (RAG)')
        clear = next(button for button in app.button if button.label == 'Clear chat')
        clear.click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.session_state['messages'], [])


if __name__ == '__main__':
    unittest.main()



