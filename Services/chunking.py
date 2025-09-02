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
    def __init__(self, chunk_size: int = 500, overlap_size: int = 50, preserve_witness_context: bool = True):
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