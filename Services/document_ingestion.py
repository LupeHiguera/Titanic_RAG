import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import PyPDF2


@dataclass
class DocumentMetadata:
    document_name: str
    source_type: str  # "us_inquiry", "british_inquiry", "other"
    total_pages: int
    extraction_date: str
    file_path: str


class DocumentIngestion:
    def __init__(self):
        pass
    
    def extract_text_from_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract text and metadata from a PDF file."""
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                
                # Extract text from all pages
                text = ""
                for page in reader.pages:
                    page_text = page.extract_text()
                    text += page_text + "\n"
                
                text = self._clean_extracted_text(text)
                
                # Create metadata
                metadata = self._create_document_metadata(text, pdf_path, len(reader.pages))
                
                return {
                    "text": text,
                    "metadata": metadata
                }
                
        except Exception as e:
            raise ValueError(f"Error reading PDF {pdf_path}: {e}")
    
    def _clean_extracted_text(self, text: str) -> str:
        """Clean extracted PDF text of artifacts."""
        # Remove excessive whitespace
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r' +', ' ', text)
        
        # Remove form feed characters and other artifacts
        text = text.replace('\f', '\n')
        text = text.replace('\x0c', '\n')
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def _create_document_metadata(self, text: str, file_path: Path, total_pages: int) -> DocumentMetadata:
        """Create metadata from extracted text and file info."""
        # Determine document name from content
        document_name = self._extract_document_name(text)
        
        # Determine source type
        source_type = self._determine_source_type(text, file_path.name)
        
        # Create metadata
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
        # Look for common inquiry patterns
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
        
        # If no specific pattern found, look for general indicators
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
        
        # Check for US Senate indicators
        us_indicators = ['senate', 'senator', 'american', 'washington']
        if any(indicator in text_lower or indicator in filename_lower for indicator in us_indicators):
            return "us_inquiry"
        
        # Check for British inquiry indicators  
        british_indicators = ['british', 'wreck commissioner', 'london', 'board of trade']
        if any(indicator in text_lower or indicator in filename_lower for indicator in british_indicators):
            return "british_inquiry"
        
        return "other"
    
    def identify_witness_names(self, text: str) -> List[str]:
        """Identify witness names from inquiry text."""
        witnesses = []
        
        # Patterns for witness introduction lines
        patterns = [
            r'Mr\.\s+([A-Z][A-Za-z\s]+),?\s*being duly sworn',
            r'Mr\.\s+([A-Z\s]+)\.\s*[A-Z]',
            r'([A-Z][A-Z\s]+),?\s*recalled',
            r'([A-Z][A-Z\s]+),?\s*sworn',
            r'Mr\.\s+([A-Z][A-Za-z\s]+)\s*testified',
            # Handle spaced names like "I SMAY" -> "Ismay"
            r'Mr\.\s+([I]\s+[A-Z]+[A-Z\s]*)\.',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                name = match.strip().rstrip(',')
                # Clean up the name and fix spacing issues
                name = self._fix_spaced_name(name)
                name = re.sub(r'\s+', ' ', name)
                if len(name) > 2 and name not in witnesses:
                    witnesses.append(name)
        
        # Remove duplicates while preserving order
        unique_witnesses = []
        for witness in witnesses:
            if witness not in unique_witnesses:
                unique_witnesses.append(witness)
        
        return unique_witnesses
    
    def _fix_spaced_name(self, name: str) -> str:
        """Fix names that have been spaced out by PDF extraction."""
        # Handle common spacing issues
        fixes = {
            'I SMAY': 'Ismay',
            'I S MAY': 'Ismay',
            'B R U C E': 'Bruce',
            'L I G H T O L L E R': 'Lightoller',
        }
        
        name_upper = name.upper()
        for spaced, fixed in fixes.items():
            if spaced in name_upper:
                return fixed
        
        # General fix for single letter followed by spaced letters
        if re.match(r'^[A-Z]\s+[A-Z\s]+$', name):
            # Try to reconstruct the name
            letters = re.findall(r'[A-Z]', name)
            reconstructed = ''.join(letters)
            
            # Known name mappings
            known_names = {
                'ISMAY': 'Ismay',
                'BRUCE': 'Bruce',
                'LIGHTOLLER': 'Lightoller',
            }
            
            return known_names.get(reconstructed, name)
        
        return name
    
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