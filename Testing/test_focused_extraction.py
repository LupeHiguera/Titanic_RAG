#!/usr/bin/env python3
"""
Focused test to debug why we're getting too many false positives.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from Services.document_ingestion import DocumentIngestion

def test_focused_extraction():
    """Test extraction on a small sample to debug issues."""
    
    print("🔍 Focused Test: Debugging False Positives")
    print("=" * 50)
    
    ingestion = DocumentIngestion()
    
    # Test on a very small sample first
    sample_text = """
[Testimony taken before Senator Bourne on behalf of the subcommittee.]

The witness was sworn by Senator Bourne.
Senator BOURNE. Kindly state your age, residence, and occupation.
Mr. CLENCH. Able-bodied seaman; I live at No. 10, the Flats, Chantry Road, Southampton.
Senator BOURNE. How long have you followed the sea?
Mr. CLENCH. About 19 years now, sir.
Senator BOURNE. What experience have you had on ships?
Mr. CLENCH. About seven years altogether on various ships.

[Testimony taken separately before Senator William Alden Smith, chairman of the subcommittee.]

The witness was sworn by Senator Smith.
Senator SMITH. Mr. Lowe, will you give your full name to the reporter?
Mr. LOWE. Harold Godfrey Lowe.
Senator SMITH. What is your position?
Mr. LOWE. I was fifth officer of the Titanic.
"""
    
    print("Testing on sample text...")
    witnesses = ingestion.identify_witness_names(sample_text)
    
    print(f"\nWitnesses found: {len(witnesses)}")
    for i, witness in enumerate(witnesses, 1):
        print(f"  {i}. {witness}")
    
    # Now test individual methods to see which one is causing issues
    print("\n" + "="*50)
    print("Testing individual extraction methods:")
    
    sections = ingestion._find_testimony_sections(sample_text)
    print(f"\nSections found: {len(sections)}")
    
    for i, section in enumerate(sections, 1):
        print(f"\n--- Section {i} ---")
        print(f"First 200 chars: {section[:200]}...")
        
        section_witnesses = ingestion._extract_witnesses_from_qa_section(section)
        print(f"Witnesses from this section: {section_witnesses}")
    
    recalled = ingestion._find_recalled_witnesses(sample_text)
    print(f"\nRecalled witnesses: {recalled}")

if __name__ == "__main__":
    test_focused_extraction()