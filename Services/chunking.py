from abc import ABC, abstractmethod
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


class BoundarySplitter(ABC):
    """Strategy for splitting raw testimony text on inquiry-specific boundaries
    (Senate Q&A turns, British numbered Q&A, etc.). Implementations return a
    list of coherent sections; the chunker further splits any section that
    exceeds chunk_size by sentence."""

    @abstractmethod
    def split(self, text: str) -> List[str]:
        ...


class SenateBoundarySplitter(BoundarySplitter):
    """US Senate Inquiry format: `Senator SMITH.` / `Mr. LOWE.` speaker turns.
    A Q&A pair is a Senator-led turn plus all witness responses until the next
    Senator turn.
    """

    _SPEAKER_PATTERN = r'(?:Senator|Mr\.|Mrs\.|Miss|Captain)\s+[A-Z][A-Z\s]*[A-Z]*\.'

    def split(self, text: str) -> List[str]:
        positions = [m.start() for m in re.finditer(self._SPEAKER_PATTERN, text)]
        if len(positions) < 2:
            return [text]

        sections = []
        i = 0
        while i < len(positions):
            start = positions[i]
            next_senator = None
            for j in range(i + 1, len(positions)):
                segment = text[positions[j]:positions[j] + 30]
                if segment.startswith('Senator'):
                    if j > i + 1 or not text[start:start + 30].startswith('Senator'):
                        next_senator = j
                        break
                    else:
                        next_senator = j
                        break

            if next_senator is not None:
                section = text[start:positions[next_senator]].strip()
                if section:
                    sections.append(section)
                i = next_senator
            else:
                section = text[start:].strip()
                if section:
                    sections.append(section)
                break

        return sections or [text]


class BritishBoundarySplitter(BoundarySplitter):
    """British Wreck Commissioner's Inquiry format: numbered Q&A like
    `190. Question? - Answer`. Each Q&A is a coherent unit. Splits on the
    `\\nN.` anchor between numbered exchanges.

    Strips embedded `Page N` markers (inquiry transcript pagination
    surfaces as standalone lines mid-testimony) before splitting.
    """

    # Match a Q&A number anywhere (word-boundary anchored), since the transcript
    # often has two Q&As on the same line: "... - Not any.  42. Just tell us..."
    # Lookahead requires a capital letter or `(` (examiner label) after the
    # period to avoid catching dates, list items, or section refs.
    _QA_NUMBER = re.compile(r'\b(\d+)\.\s+(?=[A-Z(])')
    _PAGE_MARKER = re.compile(r'^\s*Page\s+\d+\s*$', re.MULTILINE)

    def split(self, text: str) -> List[str]:
        text = self._PAGE_MARKER.sub('', text)

        positions = [m.start() for m in self._QA_NUMBER.finditer(text)]
        if not positions:
            return [text]

        sections = []
        for idx, start in enumerate(positions):
            end = positions[idx + 1] if idx + 1 < len(positions) else len(text)
            section = text[start:end].strip()
            if section:
                sections.append(section)

        return sections or [text]


class IntelligentChunker:
    def __init__(self, chunk_size: int = 800, overlap_size: int = 80,
                 preserve_witness_context: bool = True,
                 splitter: Optional[BoundarySplitter] = None):
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.preserve_witness_context = preserve_witness_context
        self.splitter: BoundarySplitter = splitter or SenateBoundarySplitter()

    def chunk_witness_contexts(self, witness_contexts: List[Any]) -> List[WitnessChunk]:
        """Chunk witness contexts into manageable pieces while preserving context."""
        chunks = []

        for i, context in enumerate(witness_contexts):
            if isinstance(context, dict):
                witness_name = context['witness']
                testimony = context['testimony']
                page_num = context.get('page_number', 1)
                doc_name = context.get('document_name', 'Unknown')
            else:
                witness_name = getattr(context, 'witness', 'Unknown')
                testimony = getattr(context, 'testimony', '')
                page_num = getattr(context, 'page_number', 1)
                doc_name = getattr(context, 'document_name', 'Unknown')

            credibility_score = 0.0
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
        if len(text) <= self.chunk_size:
            return [text]

        sections = self.splitter.split(text)

        chunks = []
        for section in sections:
            if len(section) <= self.chunk_size:
                chunks.append(section)
            else:
                chunks.extend(self._split_by_sentences(section))

        return self._add_overlap(chunks)

    def _split_by_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            potential_chunk = current_chunk + " " + sentence + "."
            potential_chunk = potential_chunk.strip()

            if len(potential_chunk) <= self.chunk_size:
                current_chunk = potential_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence + "."

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _add_overlap(self, chunks: List[str]) -> List[str]:
        if len(chunks) <= 1:
            return chunks

        overlapped_chunks = [chunks[0]]

        for i in range(1, len(chunks)):
            prev_chunk = chunks[i - 1]
            current_chunk = chunks[i]

            prev_words = prev_chunk.split()
            overlap_words = prev_words[-min(self.overlap_size // 10, len(prev_words) // 2):]

            if overlap_words:
                overlap_text = " ".join(overlap_words)
                overlapped_chunk = overlap_text + " " + current_chunk
            else:
                overlapped_chunk = current_chunk

            overlapped_chunks.append(overlapped_chunk)

        return overlapped_chunks

    def _determine_source_type(self, document_name: str) -> str:
        doc_lower = document_name.lower()
        if 'british' in doc_lower or 'wreck commissioner' in doc_lower:
            return 'british_inquiry'
        elif 'senate' in doc_lower or 'american' in doc_lower or 'us' in doc_lower:
            return 'us_inquiry'
        else:
            return 'other'
