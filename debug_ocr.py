#!/usr/bin/env python3
"""Debug OCR pattern detection"""

import sys
from pathlib import Path

root_dir = Path(__file__).parent
sys.path.append(str(root_dir))

from Services.document_ingestion import DocumentIngestion

def debug_specific_patterns():
    """Debug specific OCR patterns"""
    
    test_cases = [
        "Under the Britis h post-office authorities",
        "Mr. COTTA M. On the Carpathia",
        "No dir ection by the Marconi Co",
        "How mu ch of the time next day"
    ]
    
    doc_ingestion = DocumentIngestion()
    
    for test in test_cases:
        print(f"TESTING: {test}")
        cleaned = doc_ingestion._clean_extracted_text(test)
        print(f"RESULT:  {cleaned}")
        print(f"CHANGED: {test != cleaned}")
        print()

if __name__ == "__main__":
    debug_specific_patterns()