"""
Witness Index - Structured data from US Senate Inquiry witness list.
This replaces regex-based witness extraction with precise page-based attribution.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set
import re


@dataclass
class Witness:
    name: str
    role: str
    page: int
    ship_affiliation: str = None
    witness_type: str = None
    
    def __post_init__(self):
        """Auto-classify witness based on role."""
        if not self.ship_affiliation:
            self.ship_affiliation = self._extract_ship_affiliation()
        if not self.witness_type:
            self.witness_type = self._classify_witness_type()
    
    def _extract_ship_affiliation(self) -> str:
        """Extract ship affiliation from role."""
        role_lower = self.role.lower()
        if 'titanic' in role_lower:
            return 'Titanic'
        elif 'carpathia' in role_lower:
            return 'Carpathia' 
        elif 'californian' in role_lower:
            return 'Californian'
        elif 'olympic' in role_lower:
            return 'Olympic'
        elif 'mount temple' in role_lower:
            return 'Mount Temple'
        else:
            return 'Other'
    
    def _classify_witness_type(self) -> str:
        """Classify witness by type."""
        role_lower = self.role.lower()
        if any(title in role_lower for title in ['officer', 'captain', 'commander']):
            return 'Officer'
        elif any(title in role_lower for title in ['steward', 'seaman', 'lookout', 'quartermaster', 'fireman', 'cook', 'boatswain']):
            return 'Crew'
        elif 'passenger' in role_lower:
            return 'Passenger'
        elif any(title in role_lower for title in ['marconi', 'operator', 'engineer']):
            return 'Technical'
        elif any(title in role_lower for title in ['managing director', 'vice president', 'chairman']):
            return 'Executive'
        else:
            return 'Other'


class WitnessIndex:
    """Manages the complete witness index from US Senate Inquiry.

    Page lookups expect **printed** inquiry pages (as embedded in the PDF's
    `Page N` markers), not raw PDF page numbers — the two drift apart by
    4-10 pages through the document. Use Services.page_map.build_page_map
    to translate PDF pages before calling get_witness_by_page_range.
    """

    # Printed-page bounds of live witness testimony. Page 1 is the session
    # opening; pages after 1142 are affidavits, letters, and the digest of
    # testimony, which must not be attributed to the last witness (Barrett).
    FIRST_WITNESS_PAGE = 2
    LAST_WITNESS_PAGE = 1142

    def __init__(self):
        self.witnesses = self._load_witness_index()
        self.recalled_witnesses = self._identify_recalled_witnesses()
    
    def _load_witness_index(self) -> List[Witness]:
        """Load the complete witness index."""
        witness_data = [
            {"name": "J. Bruce Ismay", "role": "Managing Director White Star Line, Titanic passenger", "page": 2},
            {"name": "Arthur Henry Rostron", "role": "Captain, Carpathia", "page": 18},
            {"name": "Guglielmo Marconi", "role": "Chairman, British Marconi Co.", "page": 37},
            {"name": "Charles Herbert Lightoller", "role": "2nd Officer, Titanic", "page": 46},
            {"name": "Harold Thomas Cottam", "role": "Marconi Operator, Carpathia", "page": 95},
            {"name": "Alfred Crawford", "role": "Steward, Titanic", "page": 111},
            {"name": "Harold Thomas Cottam", "role": "Marconi Operator, Carpathia", "page": 121},  # recalled
            {"name": "Harold S. Bride", "role": "Marconi Operator, Titanic", "page": 133},
            {"name": "Harold Thomas Cottam", "role": "Marconi Operator, Carpathia", "page": 154},  # recalled
            {"name": "Harold S. Bride", "role": "Marconi Operator, Titanic", "page": 154},  # recalled
            {"name": "Herbert John Pitman", "role": "3rd Officer, Titanic", "page": 166},
            {"name": "Philip A. S. Franklin", "role": "Vice President, IMM", "page": 169},
            {"name": "Joseph Groves Boxhall", "role": "4th Officer, Titanic", "page": 209},
            {"name": "Herbert John Pitman", "role": "3rd Officer, Titanic", "page": 259},  # recalled
            {"name": "Frederick Fleet", "role": "Lookout, Titanic", "page": 315},
            {"name": "Major Arthur G. Peuchen", "role": "1st Class passenger, Titanic", "page": 329},
            {"name": "Frederick Fleet", "role": "Lookout, Titanic", "page": 357},  # recalled
            {"name": "Harold Godfrey Lowe", "role": "5th Officer, Titanic", "page": 368},
            {"name": "Charles Herbert Lightoller", "role": "2nd Officer, Titanic", "page": 421},  # recalled
            {"name": "Robert Hichens", "role": "Quartermaster, Titanic", "page": 449},
            {"name": "Guglielmo Marconi", "role": "Chairman, British Marconi Co", "page": 463},  # recalled
            {"name": "Harold Thomas Cottam", "role": "Marconi Operator, Carpathia", "page": 494},  # recalled
            {"name": "Guglielmo Marconi", "role": "Chairman, British Marconi Co", "page": 515},  # recalled
            {"name": "George Thomas Rowe", "role": "Quartermaster, Titanic", "page": 519},
            {"name": "Alfred Olliver", "role": "Quartermaster, Titanic", "page": 526},
            {"name": "Frank Osman", "role": "Seaman, Titanic", "page": 537},
            {"name": "Edward Wheelton", "role": "Steward, Titanic", "page": 543},
            {"name": "W. H. Taylor", "role": "Fireman, Titanic", "page": 550},
            {"name": "George Moore", "role": "Seaman, Titanic", "page": 559},
            {"name": "Thomas Jones", "role": "Seaman, Titanic", "page": 566},
            {"name": "G. Symons", "role": "Lookout, Titanic", "page": 573},
            {"name": "G. A. Hogg", "role": "Lookout, Titanic", "page": 577},
            {"name": "Walter John Perkis", "role": "Quartermaster, Titanic", "page": 580},
            {"name": "G. A. Hogg", "role": "Lookout, Titanic", "page": 583},  # recalled
            {"name": "G. Symons", "role": "Lookout, Titanic", "page": 584},  # recalled
            {"name": "John Hardy", "role": "Steward, Titanic", "page": 587},
            {"name": "William Ward", "role": "Seaman", "page": 595},
            {"name": "John Hardy", "role": "Steward, Titanic", "page": 601},  # recalled
            {"name": "James Widgery", "role": "Steward, Titanic", "page": 601},
            {"name": "Edward John Buley", "role": "Seaman, Titanic", "page": 603},
            {"name": "George Frederick Crowe", "role": "Steward, Titanic", "page": 613},
            {"name": "C. E. Andrews", "role": "Steward, Titanic", "page": 622},
            {"name": "John Collins", "role": "Cook, Titanic", "page": 627},
            {"name": "Frederick Clench", "role": "Seaman, Titanic", "page": 634},
            {"name": "Ernest Archer", "role": "Seaman, Titanic", "page": 643},
            {"name": "W. Brice", "role": "Seaman, Titanic", "page": 648},
            {"name": "Albert Haines", "role": "Boatswain's Mate, Titanic", "page": 655},
            {"name": "Samuel S. Hemming", "role": "Seaman, Titanic", "page": 662},
            {"name": "Frank Oliver Evans", "role": "Seaman, Titanic", "page": 673},
            {"name": "Philip A. S. Franklin", "role": "Vice President, IMM", "page": 688},  # recalled
            {"name": "Ernest Gill", "role": "Donkeyman, Californian", "page": 710},
            {"name": "Stanley Lord", "role": "Captain, Californian", "page": 714},
            {"name": "Cyril Furmstone Evans", "role": "Marconi Operator, Californian", "page": 733},
            {"name": "Frank Oliver Evans", "role": "Seaman, Titanic", "page": 749},  # recalled
            {"name": "Charles Herbert Lightoller", "role": "2nd Officer, Titanic", "page": 755},  # recalled
            {"name": "James Henry Moore", "role": "Captain, Mount Temple", "page": 757},
            {"name": "Charles Herbert Lightoller", "role": "2nd Officer, Titanic", "page": 785},  # recalled
            {"name": "Philip A. S. Franklin", "role": "Vice President, IMM", "page": 787},  # recalled
            {"name": "Andrew Cunningham", "role": "Steward, Titanic", "page": 790},
            {"name": "Frederick D. Ray", "role": "Steward, Titanic", "page": 798},
            {"name": "Henry Samuel Etches", "role": "Steward, Titanic", "page": 810},
            {"name": "William Burke", "role": "Steward, Titanic", "page": 821},
            {"name": "Alfred Crawford", "role": "Steward, Titanic", "page": 826},  # recalled
            {"name": "Arthur John Bright", "role": "Quartermaster, Titanic", "page": 831},
            {"name": "Alfred Crawford", "role": "Steward, Titanic", "page": 842},  # recalled
            {"name": "Guglielmo Marconi", "role": "Chairman, British Marconi Co", "page": 845},  # recalled
            {"name": "Frederick M. Sammis", "role": "Chief Engineer, Marconi Wireless Telegraph Co. of America.", "page": 845},
            {"name": "Hugh Woolner", "role": "1st Class passenger, Titanic", "page": 860},
            {"name": "Harold S. Bride", "role": "Marconi Operator, Titanic", "page": 896},  # recalled
            {"name": "Joseph Groves Boxhall", "role": "4th Officer, Titanic", "page": 907},  # recalled
            {"name": "Harold Thomas Cottam", "role": "Marconi Operator, Carpathia", "page": 918},  # recalled
            {"name": "Joseph Groves Boxhall", "role": "4th Officer, Titanic", "page": 930},  # recalled
            {"name": "Edward J. Dunn", "role": "Salesman", "page": 935},
            {"name": "Charles H. Morgan", "role": "Deputy United States Marshal", "page": 937},
            {"name": "J. Bruce Ismay", "role": "Managing Director White Star Line, Titanic passenger", "page": 938},  # recalled
            {"name": "C. E. Henry Stengel", "role": "1st Class passenger, Titanic", "page": 970},
            {"name": "J. Bruce Ismay", "role": "Managing Director White Star Line, Titanic passenger", "page": 981},  # recalled
            {"name": "Archibald Gracie", "role": "1st Class passenger, Titanic", "page": 989},
            {"name": "Helen W. Bishop", "role": "1st Class passenger, Titanic", "page": 998},
            {"name": "Dickinson H. Bishop", "role": "1st Class passenger, Titanic", "page": 1000},
            {"name": "Archibald Gracie", "role": "1st Class passenger, Titanic", "page": 1004},  # recalled
            {"name": "Mrs. J. Stuart White", "role": "1st Class passenger, Titanic", "page": 1005},
            {"name": "John Bottomley", "role": "Vice president, Marconi Wireless Telegraph Co. of America.", "page": 1010},
            {"name": "Daniel Buckley", "role": "3rd Class passenger, Titanic", "page": 1019},
            {"name": "Melville E. Stone", "role": "General Manager, Associated Press", "page": 1023},
            {"name": "George A. Harder", "role": "1st Class passenger, Titanic", "page": 1028},
            {"name": "John R. Binns", "role": "Ex-Marconi Operator, Republic", "page": 1032},
            {"name": "Olaus Abelseth", "role": "3rd Class passenger, Titanic", "page": 1036},
            {"name": "Norman Campbell Chambers", "role": "1st Class passenger, Titanic", "page": 1041},
            {"name": "Frederick Dauler", "role": "Clerk, Western Union Telegraph Co.", "page": 1047},
            {"name": "Harold S. Bride", "role": "Marconi Operator, Titanic", "page": 1051},  # recalled
            {"name": "Berk Pickard", "role": "3rd Class passenger, Titanic", "page": 1054},
            {"name": "Gilbert William Balfour", "role": "Inspector, Marconi Co.", "page": 1056},
            {"name": "Maurice L. Farrell", "role": "Managing News Editor, Dow Jones Co.", "page": 1065},
            {"name": "Benjamin Campbell", "role": "Vice President, New York, New Haven & Hartford Railroad Co.", "page": 1103},
            {"name": "John J. Knapp", "role": "United States Navy, Hydrographer", "page": 1111},
            {"name": "Herbert James Haddock", "role": "Captain, Olympic", "page": 1127},
            {"name": "Frederick Barrett", "role": "Fireman, Titanic", "page": 1140}
        ]
        
        return [Witness(**data) for data in witness_data]
    
    def _identify_recalled_witnesses(self) -> Dict[str, List[int]]:
        """Identify witnesses with multiple testimonies."""
        witness_pages = {}
        for witness in self.witnesses:
            name = witness.name
            if name not in witness_pages:
                witness_pages[name] = []
            witness_pages[name].append(witness.page)
        
        # Return only witnesses with multiple pages
        return {name: pages for name, pages in witness_pages.items() if len(pages) > 1}
    
    def get_witness_by_page(self, page_number: int) -> Optional[Witness]:
        """Get witness for a specific page number."""
        for witness in self.witnesses:
            if witness.page == page_number:
                return witness
        return None
    
    def get_witness_by_page_range(self, page_number: int) -> Optional[Witness]:
        """Get witness for a printed page number within their testimony range.

        Returns None outside [FIRST_WITNESS_PAGE, LAST_WITNESS_PAGE] — the
        opening session text and the appendices are not witness testimony.
        """
        if not (self.FIRST_WITNESS_PAGE <= page_number <= self.LAST_WITNESS_PAGE):
            return None
        # Latest applicable TOC entry wins. `>=` (not `>`) means that when two
        # witnesses start on the same page, the one listed later — the one
        # whose testimony continues onto the following pages — takes the tie.
        # (max() with a key would return the FIRST tie, silently attributing
        # e.g. all of Widgery's testimony to Hardy.)
        best = None
        for w in self.witnesses:
            if w.page <= page_number and (best is None or w.page >= best.page):
                best = w
        return best
    
    def get_witnesses_by_ship(self, ship: str) -> List[Witness]:
        """Get all witnesses affiliated with a specific ship."""
        return [w for w in self.witnesses if w.ship_affiliation.lower() == ship.lower()]
    
    def get_witnesses_by_type(self, witness_type: str) -> List[Witness]:
        """Get all witnesses of a specific type."""
        return [w for w in self.witnesses if w.witness_type.lower() == witness_type.lower()]
    
    def get_unique_witnesses(self) -> List[Witness]:
        """Get unique witnesses (first appearance only)."""
        seen_names = set()
        unique_witnesses = []
        for witness in self.witnesses:
            if witness.name not in seen_names:
                unique_witnesses.append(witness)
                seen_names.add(witness.name)
        return unique_witnesses
    
    def get_recalled_testimonies(self, witness_name: str) -> List[Witness]:
        """Get all testimonies for a recalled witness."""
        return [w for w in self.witnesses if w.name == witness_name]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the witness index."""
        unique = self.get_unique_witnesses()
        return {
            'total_testimonies': len(self.witnesses),
            'unique_witnesses': len(unique),
            'recalled_witnesses': len(self.recalled_witnesses),
            'ships': len(set(w.ship_affiliation for w in unique)),
            'witness_types': len(set(w.witness_type for w in unique)),
            'page_range': f"{min(w.page for w in self.witnesses)}-{max(w.page for w in self.witnesses)}"
        }


# Global instance for easy access
witness_index = WitnessIndex()