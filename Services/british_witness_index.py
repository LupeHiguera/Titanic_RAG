"""
British Witness Index - Structured data from the British Wreck Commissioner's
Inquiry (1912) witness list.

Keyed on **transcript page numbers** (as they appear in the printed inquiry's
own pagination), not PDF page numbers. PDF→transcript page translation lives
in get_witness_by_pdf_page(), which consumes a pre-built offset map.

Mirrors the public API of WitnessIndex (US Senate) so consumers can be polymorphic.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from Services.witness_index import Witness


# Transcript-page → witness, in source order from the British Inquiry TOC (PDF pp.3-6).
# Tuple format: (name, role, starting_transcript_page).
# Multiple entries per name = recalled witnesses (same shape as US WitnessIndex).
# Roles are hand-curated for major witnesses; "Unknown" for the rest.
_BRITISH_WITNESS_DATA: List[Dict[str, Any]] = [
    {"name": "Archie Jewell", "role": "Lookout, Titanic", "page": 17},
    {"name": "Joseph Scarrott", "role": "Seaman, Titanic", "page": 24},
    {"name": "George Beauchamp", "role": "Fireman, Titanic", "page": 34},
    {"name": "Robert Hitchins", "role": "Quartermaster, Titanic", "page": 39},
    {"name": "William Lucas", "role": "Seaman, Titanic", "page": 49},
    {"name": "Frederick Barrett", "role": "Leading Fireman, Titanic", "page": 57},
    {"name": "Frederick Barrett", "role": "Leading Fireman, Titanic", "page": 66},  # recalled
    {"name": "Reginald Lee", "role": "Lookout, Titanic", "page": 72},
    {"name": "John Poingdestre", "role": "Seaman, Titanic", "page": 80},
    {"name": "James Johnson", "role": "Steward, Titanic", "page": 89},
    {"name": "James Johnson", "role": "Steward, Titanic", "page": 94},  # recalled
    {"name": "Thomas Dillon", "role": "Trimmer, Titanic", "page": 98},
    {"name": "Thomas Ranger", "role": "Greaser, Titanic", "page": 103},
    {"name": "George Cavell", "role": "Trimmer, Titanic", "page": 106},
    {"name": "Alfred Shiers", "role": "Fireman, Titanic", "page": 111},
    {"name": "Charles Hendrickson", "role": "Leading Fireman, Titanic", "page": 116},
    {"name": "Frank Morris", "role": "Steward, Titanic", "page": 126},
    {"name": "Frederick Scott", "role": "Greaser, Titanic", "page": 130},
    {"name": "Charles Joughin", "role": "Chief Baker, Titanic", "page": 139},
    {"name": "Samuel Rule", "role": "Bath Steward, Titanic", "page": 148},
    {"name": "Stanley Lord", "role": "Captain, Californian", "page": 156},
    {"name": "James Gibson", "role": "Apprentice, Californian", "page": 171},
    {"name": "Herbert Stone", "role": "2nd Officer, Californian", "page": 177},
    {"name": "Charles V. Stone", "role": "Steward, Titanic", "page": 186},
    {"name": "George F. Stewart", "role": "Chief Officer, Californian", "page": 194},
    {"name": "Charles V. Stone", "role": "Steward, Titanic", "page": 201},  # recalled
    {"name": "Cyril Evans", "role": "Marconi Operator, Californian", "page": 201},
    {"name": "James Moore", "role": "Captain, Mount Temple", "page": 207},
    {"name": "John Durrant", "role": "Marconi Operator, Mount Temple", "page": 211},
    {"name": "John Durrant", "role": "Marconi Operator, Mount Temple", "page": 216},  # recalled
    {"name": "Samuel Rule", "role": "Bath Steward, Titanic", "page": 216},  # recalled
    {"name": "John E. Hart", "role": "Steward, Titanic", "page": 221},
    {"name": "Albert Pearcey", "role": "Steward, Titanic", "page": 230},
    {"name": "Edward Brown", "role": "Steward, Titanic", "page": 233},
    {"name": "Charles MacKay", "role": "Steward, Titanic", "page": 236},
    {"name": "Joseph Wheat", "role": "Asst. 2nd Steward, Titanic", "page": 240},
    {"name": "Charles Hendrickson", "role": "Leading Fireman, Titanic", "page": 249},  # recalled
    {"name": "George Symons", "role": "Lookout, Titanic", "page": 253},
    {"name": "James Taylor", "role": "Fireman, Titanic", "page": 268},
    {"name": "James Barr", "role": "Captain, Caronia", "page": 273},
    {"name": "Albert Horswill", "role": "Seaman, Titanic", "page": 274},
    {"name": "Cosmo Duff-Gordon", "role": "1st Class passenger, Titanic", "page": 276},
    {"name": "Cosmo Duff-Gordon", "role": "1st Class passenger, Titanic", "page": 282},  # recalled
    {"name": "Lady Duff-Gordon", "role": "1st Class passenger, Titanic", "page": 290},
    {"name": "Samuel Collins", "role": "Fireman, Titanic", "page": 292},
    {"name": "Frederick Sheath", "role": "Trimmer, Titanic", "page": 294},
    {"name": "Robert Pusey", "role": "Fireman, Titanic", "page": 294},
    {"name": "Elizabeth Leather", "role": "Stewardess, Titanic", "page": 296},
    {"name": "Joseph Wheat", "role": "Asst. 2nd Steward, Titanic", "page": 297},  # recalled
    {"name": "Annie Robinson", "role": "Stewardess, Titanic", "page": 298},
    {"name": "Walter Wynn", "role": "Quartermaster, Titanic", "page": 299},
    {"name": "Charles Lightoller", "role": "2nd Officer, Titanic", "page": 301},
    {"name": "Charles Lightoller", "role": "2nd Officer, Titanic", "page": 312},  # recalled
    {"name": "Herbert Pitman", "role": "3rd Officer, Titanic", "page": 346},
    {"name": "Joseph Boxhall", "role": "4th Officer, Titanic", "page": 354},
    {"name": "Harold Lowe", "role": "5th Officer, Titanic", "page": 366},
    {"name": "George Turnbull", "role": "Deputy Manager, Marconi Co.", "page": 371},
    {"name": "George Turnbull", "role": "Deputy Manager, Marconi Co.", "page": 374},  # recalled
    {"name": "Harold Bride", "role": "Marconi Operator, Titanic", "page": 383},
    {"name": "Charles Lightoller", "role": "2nd Officer, Titanic", "page": 394},  # recalled
    {"name": "Joseph Boxhall", "role": "4th Officer, Titanic", "page": 397},  # recalled
    {"name": "Herbert Pitman", "role": "3rd Officer, Titanic", "page": 400},  # recalled
    {"name": "Harold Lowe", "role": "5th Officer, Titanic", "page": 401},  # recalled
    {"name": "Harold Cottam", "role": "Marconi Operator, Carpathia", "page": 404},
    {"name": "Frederick Fleet", "role": "Lookout, Titanic", "page": 409},
    {"name": "George Hogg", "role": "Lookout, Titanic", "page": 415},
    {"name": "George Rowe", "role": "Quartermaster, Titanic", "page": 417},
    {"name": "Samuel Hemmings", "role": "Lamp Trimmer, Titanic", "page": 421},
    {"name": "Wilfred Seward", "role": "Chief Pantryman, Titanic", "page": 422},
    {"name": "Alfred Crawford", "role": "Steward, Titanic", "page": 426},
    {"name": "Edward Buley", "role": "Seaman, Titanic", "page": 431},
    {"name": "Ernest Archer", "role": "Seaman, Titanic", "page": 432},
    {"name": "Ernest Gill", "role": "Donkeyman, Californian", "page": 432},
    {"name": "J. Bruce Ismay", "role": "Managing Director White Star Line, Titanic passenger", "page": 435},
    {"name": "J. Bruce Ismay", "role": "Managing Director White Star Line, Titanic passenger", "page": 456},  # recalled
    {"name": "Harold Sanderson", "role": "Director, White Star Line", "page": 466},
    {"name": "Harold Sanderson", "role": "Director, White Star Line", "page": 482},  # recalled
    {"name": "Edward Wilding", "role": "Naval Architect, Harland & Wolff", "page": 498},
    {"name": "Paul Mauge", "role": "Kitchen Clerk, Titanic", "page": 508},
    {"name": "Edward Wilding", "role": "Naval Architect, Harland & Wolff", "page": 510},  # recalled
    {"name": "Edward Wilding", "role": "Naval Architect, Harland & Wolff", "page": 534},  # recalled
    {"name": "Leonard Peskett", "role": "Naval Architect, Cunard", "page": 544},
    {"name": "Alexander Carlisle", "role": "Former Chief Designer, Harland & Wolff", "page": 550},
    {"name": "Leonard Peskett", "role": "Naval Architect, Cunard", "page": 558},  # recalled
    {"name": "Charles Bartlett", "role": "Marine Superintendent, White Star Line", "page": 561},
    {"name": "Bertram Hayes", "role": "Captain, Adriatic", "page": 568},
    {"name": "Frederick Passow", "role": "Captain, St. Paul", "page": 571},
    {"name": "Francis Miller", "role": "Captain, Empress of Britain", "page": 572},
    {"name": "Benjamin Steel", "role": "Marine Superintendent, White Star Line", "page": 574},
    {"name": "Stanley Adams", "role": "Marconi Operator, Mesaba", "page": 576},
    {"name": "Walter Howell", "role": "Asst. Secretary, Board of Trade", "page": 579},
    {"name": "Walter Howell", "role": "Asst. Secretary, Board of Trade", "page": 592},  # recalled
    {"name": "Walter Howell", "role": "Asst. Secretary, Board of Trade", "page": 620},  # recalled
    {"name": "Alfred Chalmers", "role": "Nautical Adviser, Board of Trade", "page": 629},
    {"name": "Alfred Young", "role": "Professional Officer, Board of Trade", "page": 639},
    {"name": "Alfred Young", "role": "Professional Officer, Board of Trade", "page": 651},  # recalled
    {"name": "Richard Jones", "role": "Master Mariner", "page": 663},
    {"name": "Edwin Cannons", "role": "Master Mariner", "page": 666},
    {"name": "Frank Carruthers", "role": "Engineer Surveyor, Board of Trade", "page": 671},
    {"name": "Frank Carruthers", "role": "Engineer Surveyor, Board of Trade", "page": 676},  # recalled
    {"name": "William Chantler", "role": "Engineer Surveyor, Board of Trade", "page": 676},
    {"name": "Alfred Peacock", "role": "Engineer Surveyor, Board of Trade", "page": 677},
    {"name": "Maurice Clarke", "role": "Asst. Emigration Officer, Board of Trade", "page": 678},
    {"name": "William Archer", "role": "Principal Ship Surveyor, Board of Trade", "page": 681},
    {"name": "Alstander Boyle", "role": "Engineer Surveyor, Board of Trade", "page": 698},
    {"name": "Eben Sharpe", "role": "Engineer Surveyor, Board of Trade", "page": 698},
    {"name": "Joseph Harvey", "role": "Engineer Surveyor, Board of Trade", "page": 699},
    {"name": "Norman Hill", "role": "Chairman, Liverpool Steamship Owners' Assoc.", "page": 700},
    {"name": "Guglielmo Marconi", "role": "Chairman, British Marconi Co.", "page": 713},
    {"name": "Joseph Ranson", "role": "Captain, Baltic", "page": 717},
    {"name": "Ernest Shackleton", "role": "Polar Explorer", "page": 719},
    {"name": "Riversdale French", "role": "Naval Architect", "page": 724},
    {"name": "John Pritchard", "role": "Captain, Mauretania", "page": 732},
    {"name": "Hugh Young", "role": "Master Mariner", "page": 733},
    {"name": "William Stewart", "role": "Master Mariner", "page": 733},
    {"name": "John Fairfull", "role": "Master Mariner", "page": 734},
    {"name": "Andrew Braes", "role": "Master Mariner", "page": 734},
    {"name": "Edward Wilding", "role": "Naval Architect, Harland & Wolff", "page": 735},  # recalled
    {"name": "Arthur Rostron", "role": "Captain, Carpathia", "page": 740},
    {"name": "Gerald Affeld", "role": "1st Class passenger, Titanic", "page": 746},
    {"name": "Arthur Tride", "role": "Captain, Manitou", "page": 748},
]


class BritishWitnessIndex:
    """Witness index for the British Wreck Commissioner's Inquiry (1912).

    API mirrors WitnessIndex (US). Page lookups expect **transcript pages**
    (as printed in the inquiry's own pagination). For lookups by PDF page,
    supply a `pdf_to_transcript` map and call get_witness_by_pdf_page().
    """

    # Transcript-page bounds of actual witness testimony. Anything outside
    # this range is opening statements (pre-17) or closing arguments / final
    # report (post-748), which should not be attributed to any witness.
    FIRST_WITNESS_PAGE = 17
    LAST_WITNESS_PAGE = 748

    def __init__(self, pdf_to_transcript: Optional[Dict[int, int]] = None):
        self.witnesses: List[Witness] = [Witness(**data) for data in _BRITISH_WITNESS_DATA]
        self.recalled_witnesses: Dict[str, List[int]] = self._identify_recalled_witnesses()
        self.pdf_to_transcript: Dict[int, int] = pdf_to_transcript or {}

    def _identify_recalled_witnesses(self) -> Dict[str, List[int]]:
        witness_pages: Dict[str, List[int]] = {}
        for witness in self.witnesses:
            witness_pages.setdefault(witness.name, []).append(witness.page)
        return {name: pages for name, pages in witness_pages.items() if len(pages) > 1}

    def get_witness_by_page(self, transcript_page: int) -> Optional[Witness]:
        for witness in self.witnesses:
            if witness.page == transcript_page:
                return witness
        return None

    def get_witness_by_page_range(self, transcript_page: int) -> Optional[Witness]:
        """Get the witness whose testimony covers this transcript page.

        Returns None for pages outside [FIRST_WITNESS_PAGE, LAST_WITNESS_PAGE]
        — i.e., opening statements and closing arguments are not attributed.
        """
        if transcript_page < self.FIRST_WITNESS_PAGE or transcript_page > self.LAST_WITNESS_PAGE:
            return None
        # Latest applicable TOC entry wins; `>=` gives same-page ties to the
        # later-listed witness (see WitnessIndex.get_witness_by_page_range).
        best = None
        for w in self.witnesses:
            if w.page <= transcript_page and (best is None or w.page >= best.page):
                best = w
        return best

    def get_witness_by_pdf_page(self, pdf_page: int) -> Optional[Witness]:
        """Get the witness for a PDF page, via the transcript-page mapping."""
        transcript_page = self.pdf_to_transcript.get(pdf_page)
        if transcript_page is None:
            return None
        return self.get_witness_by_page_range(transcript_page)

    def get_witnesses_by_ship(self, ship: str) -> List[Witness]:
        return [w for w in self.witnesses if w.ship_affiliation.lower() == ship.lower()]

    def get_witnesses_by_type(self, witness_type: str) -> List[Witness]:
        return [w for w in self.witnesses if w.witness_type.lower() == witness_type.lower()]

    def get_unique_witnesses(self) -> List[Witness]:
        seen = set()
        unique = []
        for witness in self.witnesses:
            if witness.name not in seen:
                unique.append(witness)
                seen.add(witness.name)
        return unique

    def get_recalled_testimonies(self, witness_name: str) -> List[Witness]:
        return [w for w in self.witnesses if w.name == witness_name]

    def get_statistics(self) -> Dict[str, Any]:
        unique = self.get_unique_witnesses()
        return {
            'total_testimonies': len(self.witnesses),
            'unique_witnesses': len(unique),
            'recalled_witnesses': len(self.recalled_witnesses),
            'ships': len(set(w.ship_affiliation for w in unique)),
            'witness_types': len(set(w.witness_type for w in unique)),
            'page_range': f"{min(w.page for w in self.witnesses)}-{max(w.page for w in self.witnesses)}",
        }


# Maps British-TOC names → US-canonical names for witnesses who testified in
# BOTH inquiries under different name strings. Used for the "same person"
# hint at display time; NOT applied at ingest (per-inquiry names are what let
# the contradiction detector pair a witness against their own other-inquiry
# testimony). Witnesses whose names match exactly across inquiries (Ismay,
# Stanley Lord, Fleet, Barrett, Gill, Crawford, Ernest Archer) need no entry —
# name equality plus differing source_type already identifies them.
BRITISH_TO_US_CANONICAL: Dict[str, str] = {
    "Charles Lightoller": "Charles Herbert Lightoller",
    "Herbert Pitman": "Herbert John Pitman",
    "Joseph Boxhall": "Joseph Groves Boxhall",
    "Harold Lowe": "Harold Godfrey Lowe",
    "Harold Bride": "Harold S. Bride",
    "George Symons": "G. Symons",
    "George Hogg": "G. A. Hogg",
    "Arthur Rostron": "Arthur Henry Rostron",
    "Harold Cottam": "Harold Thomas Cottam",
    "Robert Hitchins": "Robert Hichens",       # spelling drift between TOCs
    "George Rowe": "George Thomas Rowe",
    "Edward Buley": "Edward John Buley",
    "Samuel Hemmings": "Samuel S. Hemming",
    "James Moore": "James Henry Moore",
    "Cyril Evans": "Cyril Furmstone Evans",
}


def canonical_witness_name(british_name: str) -> str:
    """Translate a British-TOC witness name to its US-canonical form when the
    witness testified in both inquiries; otherwise return the name unchanged."""
    return BRITISH_TO_US_CANONICAL.get(british_name, british_name)


def build_pdf_to_transcript_map(pdf_path: str) -> Dict[int, int]:
    """Build a pdf_page (1-indexed) → transcript_page mapping for a British
    Inquiry PDF. Thin wrapper over Services.page_map.build_page_map, which
    adds marker noise-filtering (monotonicity, plausibility, TOC pages).
    """
    from Services.page_map import build_page_map

    return build_page_map(pdf_path)


british_witness_index = BritishWitnessIndex()
