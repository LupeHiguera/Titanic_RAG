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
    def sample_pdf_path(self):
        return root_dir / "Text" / "one page.pdf"

    def test_extract_text_from_pdf_returns_text_with_metadata(self, ingestion, sample_pdf_path):
        if not sample_pdf_path.exists():
            pytest.skip("PDF file not found")

        result = ingestion.extract_text_from_pdf(sample_pdf_path)

        assert isinstance(result, dict)
        assert "text" in result
        assert "metadata" in result
        assert isinstance(result["text"], str)
        assert len(result["text"]) > 0
        assert isinstance(result["metadata"], DocumentMetadata)

    def test_extract_text_preserves_page_count(self, ingestion, sample_pdf_path):
        if not sample_pdf_path.exists():
            pytest.skip("PDF file not found")

        result = ingestion.extract_text_from_pdf(sample_pdf_path)

        assert result["metadata"].total_pages >= 1
        assert isinstance(result["metadata"].total_pages, int)

    def test_extract_text_identifies_document_type(self, ingestion, sample_pdf_path):
        if not sample_pdf_path.exists():
            pytest.skip("PDF file not found")

        result = ingestion.extract_text_from_pdf(sample_pdf_path)

        assert result["metadata"].source_type in ["us_inquiry", "british_inquiry", "other"]

    def test_extract_text_handles_nonexistent_file(self, ingestion):
        with pytest.raises(FileNotFoundError):
            ingestion.extract_text_from_pdf(Path("nonexistent.pdf"))

    def test_extract_text_handles_invalid_pdf(self, ingestion):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(b"not a pdf")
            tmp_path = Path(tmp.name)

        try:
            with pytest.raises(ValueError, match="Error reading PDF"):
                ingestion.extract_text_from_pdf(tmp_path)
        finally:
            os.unlink(tmp_path)

    def test_extract_pages_from_pdf(self, ingestion, sample_pdf_path):
        if not sample_pdf_path.exists():
            pytest.skip("PDF file not found")

        page_texts = ingestion.extract_pages_from_pdf(sample_pdf_path)

        assert isinstance(page_texts, dict)
        assert len(page_texts) >= 1
        # Pages are 1-indexed
        assert 1 in page_texts
        assert isinstance(page_texts[1], str)
        assert len(page_texts[1]) > 0

    def test_text_cleaning_removes_artifacts(self, ingestion, sample_pdf_path):
        if not sample_pdf_path.exists():
            pytest.skip("PDF file not found")

        result = ingestion.extract_text_from_pdf(sample_pdf_path)
        text = result["text"]

        # Should not have excessive whitespace
        assert not text.startswith(" " * 10)
        assert not text.endswith(" " * 10)

        # Should have some structure (sentences)
        assert "." in text or "?" in text

        # Should contain expected content (Ismay testimony)
        assert "ismay" in text.lower()
        assert "senator" in text.lower()

    def test_batch_process_documents_handles_multiple_files(self, ingestion):
        pdf_paths = [
            root_dir / "Text" / "one page.pdf",
            root_dir / "Text" / "USInq.pdf"
        ]

        existing_paths = [p for p in pdf_paths if p.exists()]
        if not existing_paths:
            pytest.skip("No PDF files found")

        results = ingestion.batch_process_documents(existing_paths)

        assert isinstance(results, list)
        assert len(results) == len(existing_paths)

        for result in results:
            assert "text" in result
            assert "metadata" in result
            assert len(result["text"]) > 0

    def test_metadata_includes_all_fields(self, ingestion, sample_pdf_path):
        if not sample_pdf_path.exists():
            pytest.skip("PDF file not found")

        result = ingestion.extract_text_from_pdf(sample_pdf_path)
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

    def test_british_pdf_source_detection(self, ingestion):
        british_pdf = root_dir / "Text" / "British_Data.pdf"
        if not british_pdf.exists():
            pytest.skip("British PDF not found")

        result = ingestion.extract_text_from_pdf(british_pdf)

        assert result["metadata"].source_type == "british_inquiry"
        assert len(result["text"]) > 0
        # British PDF should have properly spaced text (pymupdf fix)
        assert "Attorney-General" in result["text"] or "attorney" in result["text"].lower()
