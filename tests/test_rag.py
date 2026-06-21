import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

# Set dummy environment variable before imports to ensure it runs
import os
os.environ["GEMINI_API_KEY"] = "fake-key-for-testing"

from config import settings
from utils.helpers import clean_text, get_file_extension
from loaders.txt_loader import load_txt
from loaders.pdf_loader import load_pdf
from loaders.docx_loader import load_docx
from loaders.pptx_loader import load_pptx
from agents.context_agent import context_optimization_node

class TestRAGAssistant(unittest.TestCase):

    def setUp(self):
        # Create a temp text file for loading test
        self.test_dir = Path("data/test_temp")
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.txt_file = self.test_dir / "test_sample.txt"
        with open(self.txt_file, "w", encoding="utf-8") as f:
            f.write("This is a sample document for RAG testing. It contains basic text data.")

    def tearDown(self):
        # Clean up temp file
        if self.txt_file.exists():
            os.remove(self.txt_file)
        if self.test_dir.exists():
            try:
                self.test_dir.rmdir()
            except OSError:
                pass

    def test_config_paths(self):
        """Verifies settings paths are resolved properly."""
        self.assertIsNotNone(settings.BASE_DIR)
        self.assertIsNotNone(settings.UPLOAD_DIR)
        self.assertIsNotNone(settings.CHROMA_DIR)
        self.assertTrue(settings.UPLOAD_DIR.exists())

    def test_clean_text_helper(self):
        """Verifies text cleaning cleans tabs and extra spacing."""
        raw_text = "Hello    World!   \n\n  This is a   test.  "
        expected = "Hello World!\n\nThis is a test."
        self.assertEqual(clean_text(raw_text), expected)

    def test_get_file_extension(self):
        """Verifies file extensions helper behaves correctly."""
        self.assertEqual(get_file_extension("foo.pdf"), ".pdf")
        self.assertEqual(get_file_extension("bar/baz.DOCX"), ".docx")
        self.assertEqual(get_file_extension("test"), "")

    def test_txt_loader(self):
        """Verifies load_txt correctly loads TXT files."""
        docs = load_txt(str(self.txt_file))
        self.assertEqual(len(docs), 1)
        self.assertIn("RAG testing", docs[0].page_content)
        self.assertEqual(docs[0].metadata["file_name"], "test_sample.txt")
        self.assertEqual(docs[0].metadata["file_type"], "TXT")

    @patch("pathlib.Path.exists", return_value=True)
    @patch("loaders.pdf_loader.PyPDFLoader")
    def test_pdf_loader(self, mock_loader, mock_exists):
        """Verifies load_pdf parses documents and sets metadata."""
        mock_instance = MagicMock()
        mock_instance.load.return_value = [
            MagicMock(page_content="PDF Content Page 1", metadata={})
        ]
        mock_loader.return_value = mock_instance
        
        docs = load_pdf("fake_file.pdf")
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].metadata["file_name"], "fake_file.pdf")
        self.assertEqual(docs[0].metadata["file_type"], "PDF")
        self.assertEqual(docs[0].metadata["page_number"], 1)

    @patch("pathlib.Path.exists", return_value=True)
    @patch("loaders.docx_loader.Docx2txtLoader")
    def test_docx_loader(self, mock_loader, mock_exists):
        """Verifies load_docx parses Word docs and sets metadata."""
        mock_instance = MagicMock()
        mock_instance.load.return_value = [
            MagicMock(page_content="DOCX Word Content", metadata={})
        ]
        mock_loader.return_value = mock_instance
        
        docs = load_docx("fake_file.docx")
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].metadata["file_name"], "fake_file.docx")
        self.assertEqual(docs[0].metadata["file_type"], "DOCX")

    @patch("pathlib.Path.exists", return_value=True)
    @patch("pptx.Presentation")
    def test_pptx_loader(self, mock_presentation, mock_exists):
        """Verifies load_pptx extracts PowerPoint content slide-by-slide."""
        mock_prs = MagicMock()
        mock_slide = MagicMock()
        mock_shape = MagicMock()
        mock_shape.text = "PPTX Slide Text Content"
        mock_slide.shapes = [mock_shape]
        mock_prs.slides = [mock_slide]
        mock_presentation.return_value = mock_prs
        
        docs = load_pptx("fake_file.pptx")
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].metadata["file_name"], "fake_file.pptx")
        self.assertEqual(docs[0].metadata["page_number"], 1)
        self.assertEqual(docs[0].metadata["file_type"], "PPTX")

    def test_context_optimization_node(self):
        """Verifies Context Agent de-duplicates and sorts correctly."""
        doc1 = MagicMock(page_content="Chunk Content A", metadata={"file_name": "file1.pdf", "page_number": 2})
        doc2 = MagicMock(page_content="Chunk Content A", metadata={"file_name": "file1.pdf", "page_number": 2}) # Duplicate
        doc3 = MagicMock(page_content="Chunk Content B", metadata={"file_name": "file1.pdf", "page_number": 1}) # Earlier page
        
        state = {"retrieved_documents": [doc1, doc2, doc3]}
        output = context_optimization_node(state)
        
        optimized = output["optimized_context"]
        # De-duplicated
        self.assertEqual(len(optimized), 2)
        # Sorted (page 1 first, then page 2)
        self.assertEqual(optimized[0].metadata["page_number"], 1)
        self.assertEqual(optimized[1].metadata["page_number"], 2)

if __name__ == "__main__":
    unittest.main()
