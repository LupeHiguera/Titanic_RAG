from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import re


@dataclass
class ChunkMetadata:
    document_name: str
    source_type: str  # "us_inquiry", "british_inquiry", "other"
    page_number: int
    credibility_score: float
    chunk_index: int
    total_chunks_for_witness: int


@dataclass
class WitnessChunk:
    content: str
    witness_name: str
    metadata: ChunkMetadata


class IntelligentChunker:
    def __init__(self, chunk_size: int = 800, overlap_size: int = 80, preserve_witness_context: bool = True):
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.preserve_witness_context = preserve_witness_context
    
    def chunk_witness_contexts(self, witness_contexts: List[Any]) -> List[WitnessChunk]:
        """Chunk witness contexts into manageable pieces while preserving context."""
        chunks = []
        
        for i, context in enumerate(witness_contexts):
            # Handle both dict and object contexts
            if isinstance(context, dict):
                witness_name = context['witness']
                testimony = context['testimony']
                page_num = context.get('page_number', 1)
                doc_name = context.get('document_name', 'Unknown')
            else:
                # Handle mock objects and dataclass objects
                witness_name = getattr(context, 'witness', 'Unknown')
                testimony = getattr(context, 'testimony', '')
                page_num = getattr(context, 'page_number', 1)
                doc_name = getattr(context, 'document_name', 'Unknown')
            
            # Determine credibility score based on witness role
            credibility_score = self._calculate_credibility_score(witness_name)
            
            # Split testimony into chunks while preserving Q&A structure
            text_chunks = self._split_text_preserving_context(testimony)
            
            for j, chunk_text in enumerate(text_chunks):
                metadata = ChunkMetadata(
                    document_name=doc_name,
                    source_type=self._determine_source_type(doc_name),
                    page_number=page_num,
                    credibility_score=credibility_score,
                    chunk_index=j,
                    total_chunks_for_witness=len(text_chunks)
                )
                
                chunk = WitnessChunk(
                    content=chunk_text,
                    witness_name=witness_name,
                    metadata=metadata
                )
                chunks.append(chunk)
        
        return chunks
    
    def _split_text_preserving_context(self, text: str) -> List[str]:
        """Split text into chunks while preserving Q&A context and sentence boundaries."""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        
        # First, try to split on Q&A boundaries
        qa_splits = self._split_on_qa_boundaries(text)
        
        for qa_section in qa_splits:
            # If Q&A section is still too long, split by sentences
            if len(qa_section) <= self.chunk_size:
                chunks.append(qa_section)
            else:
                sentence_chunks = self._split_by_sentences(qa_section)
                chunks.extend(sentence_chunks)
        
        # Add overlap between chunks
        return self._add_overlap(chunks)
    
    def _split_on_qa_boundaries(self, text: str) -> List[str]:
        """Split text on Q&A boundaries to preserve question-answer pairs."""
        # Pattern to match Q: ... A: ... structure
        qa_pattern = r'(Q:\s*[^A]*?A:\s*[^Q]*?)(?=Q:|$)'
        matches = re.findall(qa_pattern, text, re.DOTALL | re.IGNORECASE)
        
        if matches:
            return [match.strip() for match in matches if match.strip()]
        else:
            # If no Q&A pattern found, return as single section
            return [text]
    
    def _split_by_sentences(self, text: str) -> List[str]:
        """Split text by sentences while respecting chunk size limits."""
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Check if adding this sentence would exceed chunk size
            potential_chunk = current_chunk + " " + sentence + "."
            potential_chunk = potential_chunk.strip()
            
            if len(potential_chunk) <= self.chunk_size:
                current_chunk = potential_chunk
            else:
                # Save current chunk and start new one
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence + "."
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """Add overlap between consecutive chunks."""
        if len(chunks) <= 1:
            return chunks
        
        overlapped_chunks = [chunks[0]]  # First chunk unchanged
        
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i-1]
            current_chunk = chunks[i]
            
            # Take last few words from previous chunk as overlap
            prev_words = prev_chunk.split()
            overlap_words = prev_words[-min(self.overlap_size//10, len(prev_words)//2):]
            
            if overlap_words:
                overlap_text = " ".join(overlap_words)
                overlapped_chunk = overlap_text + " " + current_chunk
            else:
                overlapped_chunk = current_chunk
            
            overlapped_chunks.append(overlapped_chunk)
        
        return overlapped_chunks
    
    def _calculate_credibility_score(self, witness_name: str) -> float:
        """Calculate credibility score based on witness role."""
        name_lower = witness_name.lower()
        
        # Officers and ship personnel have highest credibility
        if any(title in name_lower for title in ['officer', 'captain', 'commander']):
            return 0.9
        elif any(title in name_lower for title in ['crew', 'steward', 'engineer']):
            return 0.8
        elif any(title in name_lower for title in ['mr.', 'mrs.', 'miss']):
            return 0.7
        else:
            return 0.6
    
    def _determine_source_type(self, document_name: str) -> str:
        """Determine inquiry type from document name."""
        doc_lower = document_name.lower()
        if 'british' in doc_lower or 'wreck commissioner' in doc_lower:
            return 'british_inquiry'
        elif 'senate' in doc_lower or 'american' in doc_lower or 'us' in doc_lower:
            return 'us_inquiry'
        else:
            return 'other'
    
    def group_chunks_by_topic(self, chunks: List[WitnessChunk]) -> Dict[str, List[WitnessChunk]]:
        """Group chunks by topic keywords."""
        topics = {
            'lifeboats': [],
            'officers': [],
            'passengers': [],
            'crew': [],
            'collision': [],
            'evacuation': []
        }
        
        for chunk in chunks:
            content_lower = chunk.content.lower()
            
            # Simple keyword matching for topic classification
            if any(word in content_lower for word in ['lifeboat', 'boat', 'davit']):
                topics['lifeboats'].append(chunk)
            if any(word in content_lower for word in ['officer', 'captain', 'commander']):
                topics['officers'].append(chunk)
            if any(word in content_lower for word in ['passenger', 'traveler']):
                topics['passengers'].append(chunk)
            if any(word in content_lower for word in ['crew', 'steward', 'engineer']):
                topics['crew'].append(chunk)
            if any(word in content_lower for word in ['collision', 'iceberg', 'impact']):
                topics['collision'].append(chunk)
            if any(word in content_lower for word in ['evacuation', 'abandon', 'emergency']):
                topics['evacuation'].append(chunk)
        
        return topics
    
    def find_potential_contradictions(self, chunks: List[WitnessChunk]) -> List[Dict[str, Any]]:
        """Find potential contradictions between witness statements."""
        contradictions = []
        topics = self.group_chunks_by_topic(chunks)
        
        for topic, topic_chunks in topics.items():
            if len(topic_chunks) >= 2:
                # Group chunks by different witnesses
                witnesses_statements = {}
                for chunk in topic_chunks:
                    witness = chunk.witness_name
                    if witness not in witnesses_statements:
                        witnesses_statements[witness] = []
                    witnesses_statements[witness].append(chunk)
                
                # If we have statements from multiple witnesses on same topic, it's a potential contradiction
                if len(witnesses_statements) >= 2:
                    witness_names = list(witnesses_statements.keys())
                    contradiction = {
                        'topic': topic,
                        'conflicting_chunks': [
                            witnesses_statements[witness_names[0]],
                            witnesses_statements[witness_names[1]]
                        ],
                        'confidence_score': self._calculate_contradiction_confidence(
                            witnesses_statements[witness_names[0]],
                            witnesses_statements[witness_names[1]]
                        )
                    }
                    contradictions.append(contradiction)
        
        return contradictions
    
    def _calculate_contradiction_confidence(self, chunks1: List[WitnessChunk], chunks2: List[WitnessChunk]) -> float:
        """Calculate confidence score for contradiction detection."""
        # Simple heuristic based on credibility scores and topic overlap
        avg_cred1 = sum(c.metadata.credibility_score for c in chunks1) / len(chunks1)
        avg_cred2 = sum(c.metadata.credibility_score for c in chunks2) / len(chunks2)
        
        # Higher confidence if both witnesses have high credibility
        credibility_factor = (avg_cred1 + avg_cred2) / 2
        
        # Base confidence for having multiple witnesses on same topic
        base_confidence = 0.6
        
        return min(0.9, base_confidence + (credibility_factor * 0.3))
    
    def chunk_document_by_witness(self, text: str, document_metadata) -> List[WitnessChunk]:
        """Extract witnesses from document text and chunk by witness testimony."""
        
        # Step 1: Extract witness contexts from raw text
        witness_contexts = self._extract_witness_contexts_from_text(text, document_metadata)
        
        # Step 2: Chunk the witness contexts
        return self.chunk_witness_contexts(witness_contexts)
    
    def _extract_witness_contexts_from_text(self, text: str, document_metadata) -> List[Dict[str, Any]]:
        """Extract witness testimonies from raw document text."""
        contexts = []
        
        # Pattern to find testimony sections
        # Look for "TESTIMONY OF [NAME]" or similar patterns
        testimony_pattern = r'TESTIMONY OF ([A-Z\s]+?)\.?\s*\n(.*?)(?=TESTIMONY OF|$)'
        matches = re.findall(testimony_pattern, text, re.DOTALL | re.IGNORECASE)
        
        for witness_name, testimony in matches:
            witness_name = witness_name.strip()
            testimony = testimony.strip()
            
            if len(testimony) > 100:  # Only include substantial testimonies
                contexts.append({
                    'witness': witness_name,
                    'testimony': testimony,
                    'page_number': 1,  # We don't have page-level info from raw text
                    'document_name': document_metadata.document_name
                })
        
        # If no formal testimony sections found, try question-answer format
        if not contexts:
            contexts = self._extract_qa_contexts_from_text(text, document_metadata)
        
        return contexts
    
    def _extract_qa_contexts_from_text(self, text: str, document_metadata) -> List[Dict[str, Any]]:
        """Extract witness contexts from US Senate Q&A format text."""
        contexts = []
        
        # Find testimony sections marked by brackets
        testimony_sections = self._find_testimony_sections(text)
        
        for section in testimony_sections:
            section_contexts = self._extract_witnesses_from_section(section, document_metadata)
            contexts.extend(section_contexts)
        
        return contexts
    
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
    
    def _extract_witnesses_from_section(self, section_text: str, document_metadata) -> List[Dict[str, Any]]:
        """Extract witness contexts from a single testimony section."""
        contexts = []
        current_witness = None
        current_testimony = []
        witness_full_name = None
        
        lines = section_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for "The witness was sworn by..." pattern
            if re.search(r'[Tt]he witness was sworn by', line):
                continue
            
            # Look for speaker patterns: "Senator SMITH." or "Mr. LIGHTOLLER."
            speaker_match = re.match(r'(Senator|Mr\.|Mrs\.|Miss|Captain)\s+([A-Z][A-Z\s]*[A-Z]*)\.\s*(.*)', line)
            
            if speaker_match:
                title, name, content = speaker_match.groups()
                
                # If this is a new witness (not Senator), save previous and start new
                if title != 'Senator':
                    # Save previous witness if exists
                    if current_witness and current_testimony:
                        contexts.append({
                            'witness': witness_full_name or current_witness,
                            'testimony': ' '.join(current_testimony),
                            'page_number': 1,
                            'document_name': document_metadata.document_name
                        })
                    
                    # Start new witness
                    current_witness = self._clean_witness_name(name.strip())
                    current_testimony = []
                    witness_full_name = None
                    
                    # Check if content contains the full name
                    if content:
                        full_name = self._extract_full_name_from_response(content)
                        if full_name:
                            witness_full_name = full_name
                        current_testimony.append(f"{title} {name}. {content}")
                else:
                    # This is a Senator question, add to current testimony
                    if current_testimony is not None:
                        current_testimony.append(line)
            else:
                # Continuation of testimony
                if current_testimony is not None:
                    current_testimony.append(line)
        
        # Don't forget the last witness
        if current_witness and current_testimony:
            contexts.append({
                'witness': witness_full_name or current_witness,
                'testimony': ' '.join(current_testimony),
                'page_number': 1,
                'document_name': document_metadata.document_name
            })
        
        return contexts
    
    def _extract_full_name_from_response(self, response_text: str) -> str:
        """Extract full name from witness response."""
        # Pattern for full names: "Harold Godfrey Lowe" or "Charles Herbert Lightoller"
        name_patterns = [
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*\s+[A-Z][a-z]+)',  # "Harold Godfrey Lowe"
            r'([A-Z][A-Z\s]+)',  # "HAROLD GODFREY LOWE" (all caps)
        ]
        
        for pattern in name_patterns:
            matches = re.findall(pattern, response_text)
            for match in matches:
                clean_name = self._clean_witness_name(match)
                # Validate it looks like a full name (at least 2 words)
                if clean_name and len(clean_name.split()) >= 2:
                    return clean_name
        
        return None
    
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
    
    def _fix_spaced_name(self, name: str) -> str:
        """Fix names with OCR spacing issues."""
        # Common OCR fixes
        fixes = {
            'C Harles Herbert Lightoller': 'Charles Herbert Lightoller',
            'L Ightoller': 'Lightoller',
            'I Smay': 'Ismay',
            'B Oxhall': 'Boxhall',
            'R Ostron': 'Rostron',
        }
        
        for broken, fixed in fixes.items():
            if broken.lower() in name.lower():
                return fixed
        
        # General fix for single spaced letters: "C HARLES" -> "CHARLES"
        if re.match(r'^[A-Z]\s+[A-Z\s]+$', name):
            return re.sub(r'\s+', '', name)
        
        return name