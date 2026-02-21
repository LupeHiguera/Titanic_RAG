import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import fitz  # pymupdf — better text extraction than PyPDF2


@dataclass
class DocumentMetadata:
    document_name: str
    source_type: str  # "us_inquiry", "british_inquiry", "other"
    total_pages: int
    extraction_date: str
    file_path: str


class DocumentIngestion:
    def __init__(self):
        # Known OCR spacing errors: broken word -> correct word
        # Most of these are no longer needed with pymupdf, but kept as safety net
        self._known_ocr_fixes = {
            'Britis h': 'British',
            'britis h': 'british',
            'COTTA M': 'COTTAM',
            'Cotta m': 'Cottam',
        }

    def extract_text_from_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract text and metadata from a PDF file."""
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        try:
            doc = fitz.open(str(pdf_path))
            total_pages = len(doc)

            # Extract text from all pages
            text = ""
            for page in doc:
                text += page.get_text() + "\n"
            doc.close()

            text = self._clean_extracted_text(text)

            # Create metadata
            metadata = self._create_document_metadata(text, pdf_path, total_pages)

            return {
                "text": text,
                "metadata": metadata
            }

        except Exception as e:
            raise ValueError(f"Error reading PDF {pdf_path}: {e}")

    def extract_pages_from_pdf(self, pdf_path: Path) -> Dict[int, str]:
        """Extract per-page text from a PDF file (1-indexed page numbers).

        Returns a dict mapping page number -> raw page text.
        Use this for page-level witness attribution via WitnessIndex.
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        doc = fitz.open(str(pdf_path))
        page_texts = {}
        for i, page in enumerate(doc):
            page_texts[i + 1] = page.get_text()
        doc.close()
        return page_texts

    def _clean_extracted_text(self, text: str) -> str:
        """Clean extracted PDF text of artifacts and formatting issues."""
        text = self._remove_control_characters(text)
        text = self._fix_bold_artifacts(text)
        text = self._fix_ocr_spacing(text)
        text = self._normalize_whitespace(text)
        return text

    def _remove_control_characters(self, text: str) -> str:
        """Remove form feed and other control characters."""
        text = text.replace('\f', '\n')
        text = text.replace('\x0c', '\n')
        return text

    def _fix_bold_artifacts(self, text: str) -> str:
        """Fix markdown-like bold artifacts from PDF extraction."""
        # Fix words broken by bold formatting: "ano**THE**r" -> "another"
        broken_word_patterns = [
            (r'ano\*\*THE\*\*r', 'another'),
            (r'far\*\*THE\*\*r', 'farther'),
            (r'o\*\*THE\*\*r', 'other'),
            (r'mo\*\*THE\*\*r', 'mother'),
            (r'bro\*\*THE\*\*r', 'brother'),
            (r'wea\*\*THE\*\*r', 'weather'),
            (r'ga\*\*THE\*\*r', 'gather'),
            (r'fea\*\*THE\*\*r', 'feather'),
            (r'lea\*\*THE\*\*r', 'leather'),
        ]

        for pattern, replacement in broken_word_patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # Remove all remaining ** bold formatting
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        text = re.sub(r'\*\*', '', text)
        return text

    def _fix_ocr_spacing(self, text: str) -> str:
        """Fix residual spacing errors not handled by pymupdf."""
        # Fix broken hyphenated words: "brother -in-law" -> "brother-in-law"
        text = re.sub(r'\b(\w+) - ?(\w+)\b', r'\1-\2', text)

        # Apply known fixes (safety net for edge cases)
        for wrong, correct in self._known_ocr_fixes.items():
            text = text.replace(wrong, correct)

        return text

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace, punctuation spacing, and strip edges."""
        # Collapse multiple spaces
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n +', '\n', text)
        text = re.sub(r' +\n', '\n', text)
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Clean up punctuation spacing
        text = re.sub(r' +([,.!?;:])', r'\1', text)
        text = re.sub(r'([.!?]) +([A-Z])', r'\1 \2', text)

        # Fix quotation marks spacing
        text = re.sub(r' +"', '"', text)
        text = re.sub(r'" +', '" ', text)

        return text.strip()

    def _create_document_metadata(self, text: str, file_path: Path, total_pages: int) -> DocumentMetadata:
        """Create metadata from extracted text and file info."""
        document_name = self._extract_document_name(text)
        source_type = self._determine_source_type(text, file_path.name)

        metadata = DocumentMetadata(
            document_name=document_name,
            source_type=source_type,
            total_pages=total_pages,
            extraction_date=datetime.now().isoformat(),
            file_path=str(file_path)
        )

        return metadata

    def _extract_document_name(self, text: str) -> str:
        """Extract document name from text content."""
        patterns = [
            r'(U\.?S\.? Senate.*?Inquiry)',
            r'(British.*?Inquiry)',
            r'(Wreck Commissioner.*?Inquiry)',
            r'(Testimony.*?Titanic)',
            r'(Hearing.*?Titanic)'
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return matches[0].strip()

        # Check for Board of Trade (British Inquiry)
        if 'board of trade' in text.lower():
            return "British Wreck Commissioner's Inquiry - Titanic"

        if any(word in text.lower() for word in ['senate', 'inquiry', 'testimony']):
            if 'senate' in text.lower():
                return "US Senate Inquiry - Titanic Disaster"
            else:
                return "Titanic Inquiry Testimony"

        return "Titanic Document"

    def _determine_source_type(self, text: str, filename: str) -> str:
        """Determine the type of inquiry document."""
        text_lower = text.lower()
        filename_lower = filename.lower()

        us_indicators = ['senate', 'senator', 'american', 'washington']
        if any(indicator in text_lower or indicator in filename_lower for indicator in us_indicators):
            return "us_inquiry"

        british_indicators = ['british', 'wreck commissioner', 'london', 'board of trade']
        if any(indicator in text_lower or indicator in filename_lower for indicator in british_indicators):
            return "british_inquiry"

        return "other"

    def batch_process_documents(self, pdf_paths: List[Path]) -> List[Dict[str, Any]]:
        """Process multiple PDF documents."""
        results = []

        for pdf_path in pdf_paths:
            try:
                if pdf_path.exists():
                    result = self.extract_text_from_pdf(pdf_path)
                    results.append(result)
            except Exception as e:
                print(f"Error processing {pdf_path}: {e}")
                continue

        return results
