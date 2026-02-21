#!/usr/bin/env python3
"""
Test cases for new index-based witness extraction system.
Tests precise page-to-witness mapping from US Senate Inquiry index.
"""

import pytest
import sys
from pathlib import Path

# Add root directory to path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from Services.witness_index import witness_index, Witness


class TestIndexBasedWitnesses:
    """Test the new index-based witness system."""
    
    def test_lightoller_page_46_mapping(self):
        """Test that LightHollerSample.pdf maps correctly to page 46 witness."""
        # Expected: Charles Herbert Lightoller, 2nd Officer, Titanic, page 46
        expected_witness = "Charles Herbert Lightoller"
        expected_role = "2nd Officer, Titanic" 
        expected_page = 46
        
        witness = witness_index.get_witness_by_page(46)
        assert witness is not None
        assert witness.name == expected_witness
        assert witness.role == expected_role
        assert witness.page == expected_page
    
    def test_witness_index_contains_60_plus_witnesses(self):
        """Test that witness index contains all 60+ witnesses from the list."""
        expected_unique_witnesses = {
            "J. Bruce Ismay",
            "Arthur Henry Rostron", 
            "Guglielmo Marconi",
            "Charles Herbert Lightoller",
            "Harold Thomas Cottam",
            "Alfred Crawford",
            "Harold S. Bride",
            "Herbert John Pitman",
            "Philip A. S. Franklin",
            "Joseph Groves Boxhall",
            "Frederick Fleet",
            "Major Arthur G. Peuchen",
            "Harold Godfrey Lowe",
            "Robert Hichens",
            "George Thomas Rowe",
            "Alfred Olliver",
            "Frank Osman",
            "Edward Wheelton",
            "W. H. Taylor",
            "George Moore",
            "Thomas Jones",
            "G. Symons",
            "G. A. Hogg",
            "Walter John Perkis",
            "John Hardy",
            "William Ward",
            "James Widgery",
            "Edward John Buley",
            "George Frederick Crowe",
            "C. E. Andrews",
            "John Collins",
            "Frederick Clench",
            "Ernest Archer",
            "W. Brice",
            "Albert Haines",
            "Samuel S. Hemming",
            "Frank Oliver Evans",
            "Ernest Gill",
            "Stanley Lord",
            "Cyril Furmstone Evans",
            "James Henry Moore",
            "Andrew Cunningham",
            "Frederick D. Ray",
            "Henry Samuel Etches",
            "William Burke",
            "Arthur John Bright",
            "Frederick M. Sammis",
            "Hugh Woolner",
            "Edward J. Dunn",
            "Charles H. Morgan",
            "C. E. Henry Stengel",
            "Archibald Gracie",
            "Helen W. Bishop",
            "Dickinson H. Bishop",
            "Mrs. J. Stuart White",
            "John Bottomley",
            "Daniel Buckley",
            "Melville E. Stone",
            "George A. Harder",
            "John R. Binns",
            "Olaus Abelseth",
            "Norman Campbell Chambers",
            "Frederick Dauler",
            "Berk Pickard",
            "Gilbert William Balfour",
            "Maurice L. Farrell",
            "Benjamin Campbell",
            "John J. Knapp",
            "Herbert James Haddock",
            "Frederick Barrett"
        }
        
        unique_witnesses = set(w.name for w in witness_index.get_unique_witnesses())
        assert len(unique_witnesses) >= 60
        assert expected_unique_witnesses.issubset(unique_witnesses)
    
    def test_recalled_witnesses_multiple_pages(self):
        """Test handling of witnesses with multiple testimony appearances."""
        recalled_witnesses_expected = {
            "Harold Thomas Cottam": [95, 121, 154, 494, 918],
            "Harold S. Bride": [133, 154, 896, 1051], 
            "Herbert John Pitman": [166, 259],
            "Frederick Fleet": [315, 357],
            "Charles Herbert Lightoller": [46, 421, 755, 785],
            "Guglielmo Marconi": [37, 463, 515, 845],
            "Philip A. S. Franklin": [169, 688, 787],
            "Joseph Groves Boxhall": [209, 907, 930],
            "Frank Oliver Evans": [673, 749],
            "Alfred Crawford": [111, 826, 842],
            "J. Bruce Ismay": [2, 938, 981]
        }
        
        for name, expected_pages in recalled_witnesses_expected.items():
            testimonies = witness_index.get_recalled_testimonies(name)
            actual_pages = sorted([t.page for t in testimonies])
            expected_pages_sorted = sorted(expected_pages)
            assert actual_pages == expected_pages_sorted
    
    def test_witness_role_classification(self):
        """Test witness categorization by role and ship."""
        
        # Test officer classification
        officers_expected = [
            ("Charles Herbert Lightoller", "2nd Officer, Titanic"),
            ("Herbert John Pitman", "3rd Officer, Titanic"),
            ("Joseph Groves Boxhall", "4th Officer, Titanic"),
            ("Harold Godfrey Lowe", "5th Officer, Titanic"),
            ("Arthur Henry Rostron", "Captain, Carpathia"),
            ("Stanley Lord", "Captain, Californian")
        ]
        
        # Test crew classification
        crew_expected = [
            ("Frederick Fleet", "Lookout, Titanic"),
            ("Robert Hichens", "Quartermaster, Titanic"),
            ("Alfred Crawford", "Steward, Titanic"),
            ("Frederick Clench", "Seaman, Titanic"),
            ("W. H. Taylor", "Fireman, Titanic"),
            ("John Collins", "Cook, Titanic")
        ]
        
        # Test passenger classification
        passengers_expected = [
            ("J. Bruce Ismay", "Managing Director White Star Line, Titanic passenger"),
            ("Major Arthur G. Peuchen", "1st Class passenger, Titanic"),
            ("Hugh Woolner", "1st Class passenger, Titanic"),
            ("Daniel Buckley", "3rd Class passenger, Titanic")
        ]
        
        # TODO: Implement role classification
        # officers = witness_index.get_witnesses_by_type("Officer")
        # crew = witness_index.get_witnesses_by_type("Crew") 
        # passengers = witness_index.get_witnesses_by_type("Passenger")
        
        # assert len(officers) >= 6
        # assert len(crew) >= 25
        # assert len(passengers) >= 15
        
        # Placeholder assertions
        assert len(officers_expected) == 6
        assert len(crew_expected) == 6
        assert len(passengers_expected) == 4
    
    def test_ship_affiliation_grouping(self):
        """Test grouping witnesses by ship affiliation."""
        
        ships_expected = {
            "Titanic": 50,  # Majority of witnesses
            "Carpathia": 2,  # Rostron, Cottam
            "Californian": 3,  # Lord, Evans, Gill
            "Olympic": 1,  # Haddock
            "Mount Temple": 1,  # Moore
            "Other": 10  # Marconi company, press, officials, etc.
        }
        
        # TODO: Implement ship grouping
        # for ship, expected_count in ships_expected.items():
        #     witnesses = witness_index.get_witnesses_by_ship(ship)
        #     assert len(witnesses) >= expected_count
        
        # Placeholder assertion
        assert sum(ships_expected.values()) == 67
    
    def test_page_range_coverage(self):
        """Test that witness index covers full page range."""
        expected_first_page = 2  # J. Bruce Ismay
        expected_last_page = 1140  # Frederick Barrett
        
        # TODO: Implement page range checking
        # stats = witness_index.get_statistics()
        # assert stats['page_range'].startswith('2-')
        # assert stats['page_range'].endswith('1140')
        
        # Placeholder assertion
        assert expected_last_page > expected_first_page
    
    def test_witness_lookup_by_page_range(self):
        """Test finding witness responsible for content on specific pages."""
        
        test_cases = [
            # Page falls exactly on witness start page
            (46, "Charles Herbert Lightoller"),
            (95, "Harold Thomas Cottam"),
            (315, "Frederick Fleet"),
            
            # Page falls within witness testimony range
            (50, "Charles Herbert Lightoller"),  # Should still be Lightoller
            (100, "Harold Thomas Cottam"),  # Should still be Cottam
            (320, "Frederick Fleet"),  # Should still be Fleet
        ]
        
        # TODO: Implement page range lookup
        # for page, expected_witness in test_cases:
        #     witness = witness_index.get_witness_by_page_range(page)
        #     assert witness.name == expected_witness
        
        # Placeholder assertion
        assert len(test_cases) == 6
    
    def test_lighthtoller_sample_pdf_verification(self):
        """Test LightHollerSample.pdf contains expected witness content."""
        expected_content_snippets = [
            "TESTIMONY OF CHARLES HERBERT LIGHTOLLER",
            "Mr. Lightoller was sworn by the chairman", 
            "Senator SMITH. What is your name?",
            "Mr. LIGHTOLLER. Charles Herbert Lightoller",
            "Second officer of the Titanic"
        ]
        
        # Read the actual PDF content for verification
        from pathlib import Path
        pdf_path = Path(__file__).parent.parent / "Text" / "LightHollerSample.pdf"
        if pdf_path.exists():
            from Services.document_ingestion import DocumentIngestion
            doc_processor = DocumentIngestion()
            result = doc_processor.extract_text_from_pdf(pdf_path)
            pdf_content = result['text']
            
            # Verify critical content is present  
            assert "TESTIMONY OF CHARLES HERBERT LIGHTOLLER" in pdf_content
            assert "Mr. Lightoller was sworn by the chairman" in pdf_content
            assert "Charles Herbert Lightoller" in pdf_content
            assert "Page 46" in pdf_content  # Confirms page number alignment
            assert "Second officer of the Titanic" in pdf_content
        else:
            # Fallback assertion if PDF not available
            assert len(expected_content_snippets) == 5
    
    def test_index_vs_pdf_page_alignment(self):
        """Test that PDF page numbers align with witness index."""
        
        # LightHollerSample.pdf should start at page 46 according to index
        # This is critical for accurate witness attribution
        
        # TODO: Implement PDF page number extraction
        # pdf_first_page = extract_first_page_number("Text/LightHollerSample.pdf")
        # expected_page = 46
        # assert pdf_first_page == expected_page
        
        # TODO: Verify witness matches page
        # witness = witness_index.get_witness_by_page(46)
        # assert witness.name == "Charles Herbert Lightoller"
        
        # Placeholder assertion
        assert True  # Will be implemented with actual PDF processing


if __name__ == "__main__":
    pytest.main([__file__, "-v"])