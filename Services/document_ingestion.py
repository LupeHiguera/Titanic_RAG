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
        """Clean extracted PDF text of OCR artifacts and formatting issues."""
        
        # Step 1: Remove form feed and control characters
        text = text.replace('\f', '\n')
        text = text.replace('\x0c', '\n')
        
        # Step 2: Fix markdown-like artifacts (** bold formatting)
        # Simply remove ** and handle the specific "IDID" case later
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        
        # Step 3: Fix broken names with spaces (common OCR issue)
        # Names like "N EWLANDS", "S MITH", "L IGHTOLLER", "I SMAY"
        # Be more specific to avoid merging regular words
        text = re.sub(r'\b([A-Z]) ([A-Z][A-Z]+)\b', r'\1\2', text)  # Only ALL CAPS surnames
        
        # Step 4: Fix broken words with spaces in middle - be more conservative
        # "brother -in-law" → "brother-in-law" (hyphenated words)
        text = re.sub(r'\b(\w+) - ?(\w+)\b', r'\1-\2', text)
        
        # Fix specific broken words we know about
        broken_words = {
            ' at tention': ' attention',
            ' at tention ': ' attention ',
            'at tention': 'attention',
        }
        for wrong, correct in broken_words.items():
            text = text.replace(wrong, correct)
        
        # Step 5: Fix random capitalization of common words and specific artifacts
        # Keep proper nouns but fix obvious errors like "DID"
        # Use word boundaries to avoid changing proper nouns
        common_word_fixes = [
            (r'\bIDID\b', 'I did'),   # Fix "IDID" artifact from "I **DID**"
            (r'\bI DID\b', 'I did'),  # Fix "I DID" 
            (r'\bDID\b', 'did'),
            (r'\bNOT\b(?!\s+[A-Z])', 'not'),  # Don't change "NOT GUILTY" etc.
            (r'\bTHE\b(?!\s+[A-Z])', 'the'),
            (r'\bAND\b(?!\s+[A-Z])', 'and'),
            (r'\bOR\b(?!\s+[A-Z])', 'or'),
            (r'\bBUT\b(?!\s+[A-Z])', 'but'),
            (r'\bWAS\b(?!\s+[A-Z])', 'was'),
            (r'\bWERE\b(?!\s+[A-Z])', 'were'),
            (r'\bHAD\b(?!\s+[A-Z])', 'had'),
            (r'\bHAVE\b(?!\s+[A-Z])', 'have'),
            (r'\bHAS\b(?!\s+[A-Z])', 'has'),
            (r'\bWOULD\b(?!\s+[A-Z])', 'would'),
            (r'\bCOULD\b(?!\s+[A-Z])', 'could'),
            (r'\bSHOULD\b(?!\s+[A-Z])', 'should'),
        ]
        
        for pattern, replacement in common_word_fixes:
            text = re.sub(pattern, replacement, text)
        
        # Step 6: Fix specific ALL CAPS words that should be normal case
        # LIFEBOAT, SHIP, etc. - but be careful not to change names
        caps_words_to_fix = [
            (r'\bLIFEBOAT\b', 'lifeboat'),
            (r'\bSHIP\b(?!\s+[A-Z])', 'ship'),
            (r'\bBOAT\b(?!\s+[A-Z])', 'boat'),
            (r'\bWATER\b(?!\s+[A-Z])', 'water'),
        ]
        
        for pattern, replacement in caps_words_to_fix:
            text = re.sub(pattern, replacement, text)
        
        # Step 7: Normalize excessive whitespace but preserve paragraph structure
        text = re.sub(r' +', ' ', text)  # Multiple spaces to single space
        text = re.sub(r'\n +', '\n', text)  # Remove leading spaces on lines
        text = re.sub(r' +\n', '\n', text)  # Remove trailing spaces on lines
        text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 consecutive newlines
        
        # Step 8: Clean up punctuation spacing
        text = re.sub(r' +([,.!?;:])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([.!?]) +([A-Z])', r'\1 \2', text)  # Ensure space after sentence end
        
        # Step 9: Fix quotation marks spacing
        text = re.sub(r' +"', '"', text)  # Remove space before opening quote
        text = re.sub(r'" +', '" ', text)  # Ensure space after closing quote
        
        # Step 10: Strip leading/trailing whitespace
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
        """Identify witness names from US Senate Inquiry Q&A format."""
        witnesses = []
        
        # Step 1: Find testimony sections marked by brackets
        testimony_sections = self._find_testimony_sections(text)
        
        # Step 2: Extract witnesses from Q&A dialogue in each section
        for section in testimony_sections:
            section_witnesses = self._extract_witnesses_from_qa_section(section)
            for witness in section_witnesses:
                if witness not in witnesses:
                    witnesses.append(witness)
        
        # Step 3: Handle recalled witnesses format
        recalled_witnesses = self._find_recalled_witnesses(text)
        for witness in recalled_witnesses:
            if witness not in witnesses:
                witnesses.append(witness)
        
        return witnesses
    
    def _find_testimony_sections(self, text: str) -> List[str]:
        """Find testimony sections marked by bracketed headers."""
        sections = []
        
        # Pattern: [Testimony taken before Senator...]
        section_pattern = r'\[([Tt]estimony taken[^]]*)\](.*?)(?=\[[Tt]estimony taken|$)'
        matches = re.findall(section_pattern, text, re.DOTALL)
        
        for header, content in matches:
            if content.strip():
                sections.append(content.strip())
        
        # If no bracketed sections found, treat entire text as one section
        if not sections and text.strip():
            sections = [text]
        
        return sections
    
    def _extract_witnesses_from_qa_section(self, section_text: str) -> List[str]:
        """Extract witness names from Q&A dialogue section."""
        witnesses = []
        
        # Look for pattern: The witness was sworn by... followed by Q&A
        # Then find name in responses like "Mr. LOWE. Harold Godfrey Lowe."
        
        # Pattern 1: Extract name from direct name responses only
        # Look for specific pattern: "Mr. SURNAME. FirstName MiddleName LastName"
        lines = section_text.split('\n')
        
        for line in lines:
            line = line.strip()
            # Skip senator lines
            if line.startswith('Senator'):
                continue
            
            # Very specific pattern: "Mr. SURNAME. Full Name" where Full Name is 2-4 words of proper names
            name_match = re.match(r'(Mr\.|Captain)\s+([A-Z\s]+)\.\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s*\.?\s*$', line)
            if name_match:
                title, surname, full_name = name_match.groups()
                
                # Validate that full_name looks like a real name
                name_words = full_name.split()
                
                # Enhanced validation for real names
                invalid_words = ['able', 'bodied', 'seaman', 'officer', 'years', 'old', 'street', 'road', 
                               'managing', 'director', 'york', 'company', 'line', 'limited', 'corporation',
                               'department', 'service', 'station', 'building', 'office', 'united', 'states']
                
                if (len(name_words) >= 2 and 
                    all(word[0].isupper() and word[1:].islower() for word in name_words) and
                    not any(word.lower() in invalid_words for word in name_words) and
                    # Additional check: at least one word should be 3+ characters (real names)
                    any(len(word) >= 3 for word in name_words)):
                    
                    clean_name = self._clean_witness_name(full_name)
                    if clean_name and clean_name not in witnesses:
                        witnesses.append(clean_name)
        
        # Pattern 2: If no full names found, extract from surname and guess
        if not witnesses:
            witnesses_from_context = self._extract_witnesses_from_qa_context(section_text)
            witnesses.extend(witnesses_from_context)
        
        return witnesses
    
    def _extract_witnesses_from_qa_context(self, section_text: str) -> List[str]:
        """Extract witnesses by analyzing Q&A context to avoid senators."""
        witnesses = []
        lines = section_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Skip any line that starts with "Senator" 
            if line.startswith('Senator'):
                continue
            
            # Check for witness responses (Mr./Captain/etc.) but NOT senator responses
            witness_match = re.match(r'(Mr\.|Captain)\s+([A-Z\s]+)\.', line)
            if witness_match:
                title, surname = witness_match.groups()
                clean_surname = self._clean_witness_name(surname)
                
                # Additional filtering: common senator surnames to avoid
                senator_surnames = ['SMITH', 'BOURNE', 'FLETCHER', 'PERKINS']
                if clean_surname.upper() in senator_surnames:
                    continue
                
                if (clean_surname and len(clean_surname.strip()) > 2):
                    # Map to full name if possible
                    full_name = self._map_surname_to_full_name(clean_surname)
                    if full_name and full_name not in witnesses:
                        witnesses.append(full_name)
        
        return witnesses
    
    def _find_recalled_witnesses(self, text: str) -> List[str]:
        """Find witnesses in 'recalled' format: HAROLD GODFREY LOWE, recalled."""
        witnesses = []
        
        # Pattern: "NAME, recalled" or "NAME (recalled)"
        recalled_patterns = [
            r'([A-Z][A-Z\s]+),\s*recalled',
            r'([A-Z][A-Z\s]+)\s*\(recalled\)',
            r'([A-Z][A-Z\s]+),?\s*recalled\.',
        ]
        
        for pattern in recalled_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                clean_name = self._clean_witness_name(match)
                if clean_name and clean_name not in witnesses:
                    witnesses.append(clean_name)
        
        return witnesses
    
    def _clean_witness_name(self, name: str) -> str:
        """Clean and standardize witness names."""
        if not name:
            return ""
        
        # Remove extra spaces and punctuation
        name = re.sub(r'[,.]', '', name.strip())
        name = re.sub(r'\s+', ' ', name)
        
        # Fix OCR spacing issues like "C HARLES HERBERT LIGHTOLLER"
        name = self._fix_spaced_name(name)
        
        # Convert to title case if all caps
        if name.isupper():
            name = name.title()
        
        return name.strip()
    
    def _map_surname_to_full_name(self, surname: str) -> str:
        """Map surname to full name using known witness list."""
        # Known mappings from witness.pdf
        surname_mapping = {
            'ISMAY': 'Bruce Ismay',
            'LIGHTOLLER': 'Charles Herbert Lightoller', 
            'LOWE': 'Harold Godfrey Lowe',
            'BOXHALL': 'Joseph Groves Boxhall',
            'PITMAN': 'Herbert John Pitman',
            'FLEET': 'Frederick Fleet',
            'CLENCH': 'Frederick Clench',
            'ROSTRON': 'Arthur Henry Rostron',
            'COTTAM': 'Harold Thomas Cottam',
            'BRIDE': 'Harold Sydney Bride',
            'CRAWFORD': 'Alfred Crawford',
            'BUCKLEY': 'Daniel Buckley',
            'ETCHES': 'Henry Samuel Etches',
            'CROWE': 'George Frederick Crowe',
            'BULEY': 'Edward John Buley',
            'EVANS': 'Cyril Furmstone Evans',
            'HAINES': 'Albert Haines',
            'HARDY': 'John Hardy',
            'JONES': 'Thomas Jones',
            'WHEELTON': 'Edward Wheelton',
            'WARD': 'William Ward',
            'PERKIS': 'Walter John Perkis',
            'GILL': 'Ernest Gill',
            'COLLINS': 'John Collins',
            'ABELSETH': 'Olaus Abelseth',
            'WIDGERY': 'James Widgery',
            'CUNNINGHAM': 'Andrew Cunningham',
            'BURKE': 'William Burke',
            'DAULER': 'Frederick Dauler',
        }
        
        surname_clean = surname.upper().strip()
        return surname_mapping.get(surname_clean, surname)
    
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