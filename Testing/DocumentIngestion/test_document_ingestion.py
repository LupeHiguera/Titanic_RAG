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
        return Path("../Text/USInq.pdf")
    
    @pytest.fixture
    def expected_metadata(self):
        return DocumentMetadata(
            document_name="US Senate Inquiry - Day 1",
            source_type="inquiry",
            page_number=1,
            total_pages=100
        )
    
    def test_extract_text_from_pdf_returns_text_with_metadata(self, ingestion, sample_pdf_path):
        result = ingestion.extract_text_from_pdf(sample_pdf_path)
        
        assert isinstance(result, dict)
        assert "text" in result
        assert "metadata" in result
        assert isinstance(result["text"], str)
        assert len(result["text"]) > 0
        assert isinstance(result["metadata"], DocumentMetadata)
    
    def test_extract_text_preserves_page_numbers(self, ingestion, sample_pdf_path):
        result = ingestion.extract_text_from_pdf(sample_pdf_path)
        
        assert "page_number" in result["metadata"].__dict__
        assert result["metadata"].page_number >= 1
    
    def test_extract_text_identifies_document_type(self, ingestion, sample_pdf_path):
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
            with pytest.raises(ValueError, match="Invalid PDF"):
                ingestion.extract_text_from_pdf(tmp_path)
        finally:
            os.unlink(tmp_path)
    
    def test_identify_witness_names_in_text(self, ingestion):
        sample_text = """
        CHARLES HERBERT LIGHTOLLER, recalled.
        Examined by MR. BUTLER ASPINALL.
        
        Q: What was your position on the Titanic?
        A: I was Second Officer.
        
        HAROLD SYDNEY BRIDE, sworn.
        Examined by the ATTORNEY-GENERAL.
        """
        
        witnesses = ingestion.identify_witness_names(sample_text)
        
        assert "Charles Herbert Lightoller" in witnesses
        assert "Harold Sydney Bride" in witnesses
        assert len(witnesses) == 2
    
    def test_extract_witness_context_maintains_speaker_identity(self, ingestion):
        sample_text = """
        CHARLES HERBERT LIGHTOLLER, recalled.
        Q: What happened to the lifeboats?
        A: We lowered them in order, women and children first.
        
        Q: Were there enough lifeboats?
        A: No, there were not sufficient for all passengers.
        """
        
        contexts = ingestion.extract_witness_contexts(sample_text)
        
        assert len(contexts) > 0
        assert all("witness" in context for context in contexts)
        assert any("Charles Herbert Lightoller" in context["witness"] for context in contexts)
        assert any("lifeboats" in context["testimony"].lower() for context in contexts)
    
    def test_process_document_returns_structured_data(self, ingestion, sample_pdf_path):
        result = ingestion.process_document(sample_pdf_path)
        
        assert isinstance(result, dict)
        assert "raw_text" in result
        assert "metadata" in result
        assert "witness_contexts" in result
        assert "witnesses_identified" in result
        
        assert isinstance(result["witness_contexts"], list)
        assert isinstance(result["witnesses_identified"], list)
    
    def test_batch_process_documents_handles_multiple_files(self, ingestion):
        file_paths = [Path("../Text/USInq.pdf"), Path("Text/BritishInq.pdf")]
        
        results = ingestion.batch_process_documents(file_paths)
        
        assert isinstance(results, list)
        assert len(results) == len([p for p in file_paths if p.exists()])
        
        for result in results:
            assert "raw_text" in result
            assert "metadata" in result
            assert "witness_contexts" in result