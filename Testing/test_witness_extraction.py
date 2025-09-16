#!/usr/bin/env python3
"""
Test script for enhanced witness extraction.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Services.document_ingestion import DocumentIngestion
from Services.chunking import IntelligentChunker
from dataclasses import dataclass

@dataclass
class MockDocumentMetadata:
    document_name: str = "US Senate Inquiry - Test"
    source_type: str = "us_inquiry"

def test_witness_extraction():
    """Test the enhanced witness extraction against our test cases."""
    
    # Initialize the extraction services
    ingestion = DocumentIngestion()
    chunker = IntelligentChunker()
    mock_metadata = MockDocumentMetadata()
    
    # Test Case 1: Standard Senate Q&A Format
    test_case_1 = """
[Testimony taken before Senator Bourne on behalf of the subcommittee.]

The witness was sworn by Senator Bourne.
Senator BOURNE. Kindly state your age, residence, and occupation.
Mr. CLENCH. Able-bodied seaman; I live at No. 10, the Flats, Chantry Road, Southampton.
Senator BOURNE. How long have you followed the sea?
Mr. CLENCH. About 19 years now, sir.
"""
    
    print("=== TEST CASE 1: Standard Senate Q&A Format ===")
    witnesses_1 = ingestion.identify_witness_names(test_case_1)
    contexts_1 = chunker._extract_qa_contexts_from_text(test_case_1, mock_metadata)
    
    print(f"Witnesses found: {witnesses_1}")
    print(f"Contexts found: {len(contexts_1)}")
    if contexts_1:
        print(f"First context witness: {contexts_1[0]['witness']}")
        print(f"Testimony preview: {contexts_1[0]['testimony'][:200]}...")
    print()
    
    # Test Case 2: Chairman Format
    test_case_2 = """
The witness was sworn by the chairman.
Senator SMITH. Will you give your full name to the reporter?
Mr. LOWE. Harold Godfrey Lowe.
Senator SMITH. I would like to have you turn your chair so you are facing the reporter.
Mr. LOWE. Yes, sir.
"""
    
    print("=== TEST CASE 2: Chairman Format ===")
    witnesses_2 = ingestion.identify_witness_names(test_case_2)
    contexts_2 = chunker._extract_qa_contexts_from_text(test_case_2, mock_metadata)
    
    print(f"Witnesses found: {witnesses_2}")
    print(f"Contexts found: {len(contexts_2)}")
    if contexts_2:
        print(f"First context witness: {contexts_2[0]['witness']}")
    print()
    
    # Test Case 3: OCR Artifacts Format
    test_case_3 = """
The witness was sworn by the chairman.
Senator S MITH. Will you give your name?
Mr. L IGHTOLLER. Charles Herbert Lightoller.
Senator S MITH. What is your position?
Mr. L IGHTOLLER. I was first officer of the Titanic.
"""
    
    print("=== TEST CASE 3: OCR Artifacts Format ===")
    witnesses_3 = ingestion.identify_witness_names(test_case_3)
    contexts_3 = chunker._extract_qa_contexts_from_text(test_case_3, mock_metadata)
    
    print(f"Witnesses found: {witnesses_3}")
    print(f"Contexts found: {len(contexts_3)}")
    if contexts_3:
        print(f"First context witness: {contexts_3[0]['witness']}")
    print()
    
    # Test Case 4: Captain Title
    test_case_4 = """
The witness was sworn by the chairman.
Senator SMITH. State your name and position.
Captain ROSTRON. Arthur Henry Rostron, Captain of the steamship Carpathia.
Senator SMITH. How long have you been captain?
Captain ROSTRON. About 13 years.
"""
    
    print("=== TEST CASE 4: Captain Title ===")
    witnesses_4 = ingestion.identify_witness_names(test_case_4)
    contexts_4 = chunker._extract_qa_contexts_from_text(test_case_4, mock_metadata)
    
    print(f"Witnesses found: {witnesses_4}")
    print(f"Contexts found: {len(contexts_4)}")
    if contexts_4:
        print(f"First context witness: {contexts_4[0]['witness']}")
    print()
    
    # Test Case 5: Recalled Format
    test_case_5 = """
HAROLD GODFREY LOWE, recalled.

Senator SMITH. Mr. Lowe, I want to ask you about the distress signals.
Mr. LOWE. Yes, sir.
"""
    
    print("=== TEST CASE 5: Recalled Format ===")
    witnesses_5 = ingestion.identify_witness_names(test_case_5)
    contexts_5 = chunker._extract_qa_contexts_from_text(test_case_5, mock_metadata)
    
    print(f"Witnesses found: {witnesses_5}")
    print(f"Contexts found: {len(contexts_5)}")
    if contexts_5:
        print(f"First context witness: {contexts_5[0]['witness']}")
    print()
    
    # Summary
    all_witnesses = set()
    all_witnesses.update(witnesses_1)
    all_witnesses.update(witnesses_2) 
    all_witnesses.update(witnesses_3)
    all_witnesses.update(witnesses_4)
    all_witnesses.update(witnesses_5)
    
    print("=== SUMMARY ===")
    print(f"Total unique witnesses extracted: {len(all_witnesses)}")
    print(f"Witnesses: {sorted(all_witnesses)}")
    
    # Expected witnesses from our test cases
    expected = {
        "Frederick Clench",
        "Harold Godfrey Lowe", 
        "Charles Herbert Lightoller",
        "Arthur Henry Rostron"
    }
    
    found_expected = all_witnesses.intersection(expected)
    missing_expected = expected - all_witnesses
    
    print(f"Expected witnesses found: {len(found_expected)}/{len(expected)}")
    if missing_expected:
        print(f"Missing expected witnesses: {missing_expected}")
    
    success_rate = len(found_expected) / len(expected) * 100
    print(f"Success rate: {success_rate:.1f}%")

if __name__ == "__main__":
    test_witness_extraction()