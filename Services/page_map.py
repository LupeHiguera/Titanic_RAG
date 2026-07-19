"""PDF-page → printed-page mapping for inquiry transcripts.

Both inquiry PDFs embed the original printed pagination as standalone
`Page N` lines. Witness indexes are keyed on those printed pages, so
attribution must translate PDF page numbers through this map:

- US Senate: printed pages drift 4-10 behind PDF pages (front matter,
  inserted exhibits), so raw PDF-page attribution is off by up to a whole
  witness near session boundaries.
- British: ~2.5 PDF pages per transcript page.

Markers are noisy — front matter contains lines like `Page 603` that are
references, not pagination. accept/reject rules:

1. A printed page can never exceed its PDF page (covers precede page 1).
2. Printed pages are monotonic non-decreasing.
3. A forward jump can't outrun the PDF pages elapsed by more than a small
   slack (printed pagination advances at most ~1 page per PDF page).

Pages without an accepted marker inherit the last accepted printed page
(carry-forward), matching how a reader would cite them.
"""
from __future__ import annotations

from typing import Dict, Optional

import fitz
import re

_PAGE_MARKER = re.compile(r"^\s*Page\s+(\d+)\s*$", re.MULTILINE)

# A printed page number may exceed the PDF pages elapsed since the last
# accepted marker by at most this much (headers/footers make single markers
# wobble by a page or two).
_JUMP_SLACK = 5

# A body page shows at most a leftover footer plus a header (2-3 markers).
# Front-matter contents pages list `Page N` per witness entry — dozens of
# markers — and none of them are pagination.
_MAX_MARKERS_PER_PAGE = 3


def build_page_map(pdf_path: str) -> Dict[int, int]:
    """Scan a PDF for `Page N` markers and return a
    pdf_page (1-indexed) → printed_page mapping.

    PDF pages before the first accepted marker are omitted; pages without
    their own marker inherit the last accepted printed page.
    """
    pdf = fitz.open(pdf_path)
    mapping: Dict[int, int] = {}
    current: Optional[int] = None
    current_pdf_page: Optional[int] = None

    try:
        for i in range(pdf.page_count):
            pdf_page = i + 1
            markers = [int(m) for m in _PAGE_MARKER.findall(pdf[i].get_text())]
            if len(markers) > _MAX_MARKERS_PER_PAGE:
                markers = []  # reference listing (TOC/index), not pagination
            accepted = [m for m in markers if _plausible(m, pdf_page, current, current_pdf_page)]
            if accepted:
                # Highest surviving marker: a page can show the leftover
                # footer of one printed page and the header of the next.
                current = max(accepted)
                current_pdf_page = pdf_page
            if current is not None:
                mapping[pdf_page] = current
    finally:
        pdf.close()

    return mapping


def _plausible(marker: int, pdf_page: int, current: Optional[int],
               current_pdf_page: Optional[int]) -> bool:
    if marker > pdf_page:  # printed pagination never runs ahead of the PDF
        return False
    if current is None:
        return True
    if marker < current:  # monotonic non-decreasing
        return False
    return marker - current <= (pdf_page - current_pdf_page) + _JUMP_SLACK
