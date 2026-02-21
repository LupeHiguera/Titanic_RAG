import pytest
from pathlib import Path
import tempfile
import os
import sys

# Add the root directory to path
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

from Services.document_ingestion import DocumentIngestion, DocumentMetadata


class TestDocumentIngestion:

    @pytest.fixture
    def ingestion(self):
        return DocumentIngestion()

    @pytest.fixture
    def real_pdf_path(self):
        return root_dir / "Text" / "one page.pdf"

    @pytest.fixture
    def us_inquiry_pdf_path(self):
        return root_dir / "Text" / "USInq.pdf"

    def test_extract_text_from_real_pdf(self, ingestion, real_pdf_path):
        """Test with actual one page.pdf file"""
        if not real_pdf_path.exists():
            pytest.skip("PDF file not found")

        result = ingestion.extract_text_from_pdf(real_pdf_path)

        assert isinstance(result, dict)
        assert "text" in result
        assert "metadata" in result
        assert isinstance(result["text"], str)
        assert len(result["text"]) > 100  # Should have substantial text
        assert isinstance(result["metadata"], DocumentMetadata)

    def test_extract_text_preserves_page_count(self, ingestion, real_pdf_path):
        """Test that page count is correctly identified"""
        if not real_pdf_path.exists():
            pytest.skip("PDF file not found")

        result = ingestion.extract_text_from_pdf(real_pdf_path)

        assert result["metadata"].total_pages >= 1
        assert isinstance(result["metadata"].total_pages, int)

    def test_identifies_document_source_type(self, ingestion, real_pdf_path):
        """Test document type identification"""
        if not real_pdf_path.exists():
            pytest.skip("PDF file not found")

        result = ingestion.extract_text_from_pdf(real_pdf_path)

        source_type = result["metadata"].source_type
        assert source_type in ["us_inquiry", "british_inquiry", "other"]

    def test_extracts_document_name_from_content(self, ingestion, real_pdf_path):
        """Test that document name is intelligently extracted"""
        if not real_pdf_path.exists():
            pytest.skip("PDF file not found")

        result = ingestion.extract_text_from_pdf(real_pdf_path)

        doc_name = result["metadata"].document_name
        assert isinstance(doc_name, str)
        assert len(doc_name) > 0

    def test_handles_nonexistent_file(self, ingestion):
        """Test error handling for missing files"""
        fake_path = Path("nonexistent.pdf")

        with pytest.raises(FileNotFoundError):
            ingestion.extract_text_from_pdf(fake_path)

    def test_handles_invalid_pdf(self, ingestion):
        """Test error handling for corrupted PDF"""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(b"not a real pdf file")
            tmp_path = Path(tmp.name)

        try:
            with pytest.raises(ValueError, match="Error reading PDF"):
                ingestion.extract_text_from_pdf(tmp_path)
        finally:
            os.unlink(tmp_path)

    def test_real_pdf_contains_expected_content(self, ingestion, real_pdf_path):
        """Test that known content from one page.pdf (Ismay testimony) is extracted"""
        if not real_pdf_path.exists():
            pytest.skip("PDF file not found")

        result = ingestion.extract_text_from_pdf(real_pdf_path)
        text = result["text"].lower()

        # The one page.pdf contains Ismay's testimony
        assert "ismay" in text
        assert "senator" in text

    def test_extract_metadata_includes_all_fields(self, ingestion, real_pdf_path):
        """Test that all required metadata fields are present"""
        if not real_pdf_path.exists():
            pytest.skip("PDF file not found")

        result = ingestion.extract_text_from_pdf(real_pdf_path)
        metadata = result["metadata"]

        assert hasattr(metadata, 'document_name')
        assert hasattr(metadata, 'source_type')
        assert hasattr(metadata, 'total_pages')
        assert hasattr(metadata, 'extraction_date')
        assert hasattr(metadata, 'file_path')

        assert isinstance(metadata.document_name, str)
        assert isinstance(metadata.source_type, str)
        assert isinstance(metadata.total_pages, int)
        assert isinstance(metadata.file_path, str)

    def test_batch_process_multiple_pdfs(self, ingestion):
        """Test processing multiple PDFs at once"""
        pdf_paths = [
            root_dir / "Text" / "one page.pdf",
            root_dir / "Text" / "USInq.pdf"
        ]

        existing_paths = [p for p in pdf_paths if p.exists()]

        if existing_paths:
            results = ingestion.batch_process_documents(existing_paths)

            assert isinstance(results, list)
            assert len(results) == len(existing_paths)

            for result in results:
                assert "text" in result
                assert "metadata" in result
                assert len(result["text"]) > 0

    def test_text_cleaning_removes_artifacts(self, ingestion, real_pdf_path):
        """Test that extracted text is cleaned of PDF artifacts"""
        if not real_pdf_path.exists():
            pytest.skip("PDF file not found")

        result = ingestion.extract_text_from_pdf(real_pdf_path)
        text = result["text"]

        # Should not have excessive whitespace
        assert not text.startswith(" " * 10)
        assert not text.endswith(" " * 10)

        # Should have some structure (sentences)
        assert "." in text or "?" in text

        # Should contain expected content
        assert len(text.strip()) > 50

    def test_extract_pages_returns_per_page_text(self, ingestion, real_pdf_path):
        """Test per-page extraction for witness index attribution"""
        if not real_pdf_path.exists():
            pytest.skip("PDF file not found")

        page_texts = ingestion.extract_pages_from_pdf(real_pdf_path)

        assert isinstance(page_texts, dict)
        assert 1 in page_texts
        assert len(page_texts[1]) > 0
