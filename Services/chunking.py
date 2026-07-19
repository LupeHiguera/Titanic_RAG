from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import re


@dataclass
class ChunkMetadata:
    document_name: str
    source_type: str  # "us_inquiry", "british_inquiry", "other"
    page_number: int  # printed/transcript page (citable), not raw PDF page
    credibility_score: float
    chunk_index: int
    total_chunks_for_witness: int
    role: str = ""            # e.g. "2nd Officer, Titanic"
    ship: str = ""            # e.g. "Titanic", "Californian"
    witness_type: str = ""    # Officer / Crew / Passenger / Technical / Executive


# Inline page markers: ingestion embeds ⟦p:N⟧ where the printed page changes
# so each chunk can cite the page it actually came from. The glyphs never
# occur in 1912 OCR text. Stripped from content after page assignment.
PAGE_TAG = re.compile(r'⟦p:(\d+)⟧')


def page_tag(page: int) -> str:
    return f'⟦p:{page}⟧'


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

        # A section starts at each Senator-led turn; witness turns (Mr. LOWE.)
        # belong to the Senator turn that prompted them. The first speaker
        # position starts a section regardless, and text before it (swearing-in
        # narration, page tags) is kept as its own leading section.
        boundaries = [p for p in positions if text.startswith('Senator', p)]
        if not boundaries or boundaries[0] != positions[0]:
            boundaries.insert(0, positions[0])

        sections = []
        lead = text[:boundaries[0]].strip()
        if lead:
            sections.append(lead)
        for idx, start in enumerate(boundaries):
            end = boundaries[idx + 1] if idx + 1 < len(boundaries) else len(text)
            section = text[start:end].strip()
            if section:
                sections.append(section)

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
        lead = text[:positions[0]].strip()
        if lead:
            sections.append(lead)  # swearing-in narration before the first Q
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
                role = context.get('role', '')
                ship = context.get('ship', '')
                witness_type = context.get('witness_type', '')
            else:
                witness_name = getattr(context, 'witness', 'Unknown')
                testimony = getattr(context, 'testimony', '')
                page_num = getattr(context, 'page_number', 1)
                doc_name = getattr(context, 'document_name', 'Unknown')
                role = getattr(context, 'role', '')
                ship = getattr(context, 'ship', '')
                witness_type = getattr(context, 'witness_type', '')

            credibility_score = 0.0
            text_chunks = self._split_text_preserving_context(testimony)
            paged_chunks = self._assign_pages(text_chunks, page_num)

            for j, (chunk_text, chunk_page) in enumerate(paged_chunks):
                metadata = ChunkMetadata(
                    document_name=doc_name,
                    source_type=self._determine_source_type(doc_name),
                    page_number=chunk_page,
                    credibility_score=credibility_score,
                    chunk_index=j,
                    total_chunks_for_witness=len(paged_chunks),
                    role=role,
                    ship=ship,
                    witness_type=witness_type,
                )

                chunk = WitnessChunk(
                    content=chunk_text,
                    witness_name=witness_name,
                    metadata=metadata
                )
                chunks.append(chunk)

        return chunks

    @staticmethod
    def _assign_pages(text_chunks: List[str], start_page: int) -> List[tuple]:
        """Resolve each chunk's citable page from inline ⟦p:N⟧ tags, then strip
        the tags from the content. A chunk's page is the page in effect where it
        starts; tags inside it advance the running page for later chunks.
        Chunks that were only tags/whitespace are dropped."""
        current_page = start_page
        result = []
        for chunk_text in text_chunks:
            lead = PAGE_TAG.match(chunk_text.strip())
            tags = PAGE_TAG.findall(chunk_text)
            chunk_page = int(lead.group(1)) if lead else current_page
            if tags:
                current_page = int(tags[-1])
            content = PAGE_TAG.sub(' ', chunk_text)
            content = re.sub(r'[ \t]{2,}', ' ', content).strip()
            if content:
                result.append((content, chunk_page))
        return result

    def _split_text_preserving_context(self, text: str) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text]

        sections = self.splitter.split(text)

        # Split oversized sections by sentence, then pack adjacent pieces back
        # up toward chunk_size. Q&As in these transcripts are short (~140-200
        # chars); one-per-chunk starves the embedder of context, so a chunk
        # holds as many *whole* consecutive Q&As as fit.
        pieces = []
        for section in sections:
            if len(section) <= self.chunk_size:
                pieces.append(section)
            else:
                pieces.extend(self._split_by_sentences(section))

        return self._add_overlap(self._pack_pieces(pieces))

    def _pack_pieces(self, pieces: List[str]) -> List[str]:
        """Greedily merge consecutive pieces up to chunk_size, never splitting
        a piece (Q&A boundaries stay intact)."""
        chunks: List[str] = []
        buffer = ""
        for piece in pieces:
            candidate = f"{buffer}\n{piece}" if buffer else piece
            if len(candidate) <= self.chunk_size:
                buffer = candidate
            else:
                if buffer:
                    chunks.append(buffer)
                buffer = piece
        if buffer:
            chunks.append(buffer)
        return chunks

    # Sentence boundary: terminal punctuation, whitespace, then an uppercase
    # letter or opening quote. Keeps decimal times ("11.40") and initials
    # intact — the old [.!?]+ split shredded them and rewrote ?/! as periods.
    _SENTENCE_BOUNDARY = re.compile(r'(?<=[.!?])\s+(?=[A-Z"“(])')

    def _split_by_sentences(self, text: str) -> List[str]:
        sentences = self._SENTENCE_BOUNDARY.split(text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            potential_chunk = f"{current_chunk} {sentence}".strip()
            if len(potential_chunk) <= self.chunk_size:
                current_chunk = potential_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence

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
