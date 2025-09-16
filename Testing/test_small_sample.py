#!/usr/bin/env python3
"""
Test improved extraction on a small sample of USInq.pdf to validate quality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from Services.document_ingestion import DocumentIngestion

def test_small_sample():
    """Test extraction on first few pages of USInq.pdf."""
    
    print("🔍 Testing on Small Sample of USInq.pdf")
    print("=" * 50)
    
    ingestion = DocumentIngestion()
    
    # Extract just first few pages to test quality
    usinq_path = Path("Text/USInq.pdf")
    
    if not usinq_path.exists():
        print(f"❌ USInq.pdf not found")
        return
    
    try:
        import PyPDF2
        with open(usinq_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            
            # Read first 10 pages only
            sample_text = ""
            for page_num in range(min(10, len(reader.pages))):
                page_text = reader.pages[page_num].extract_text()
                sample_text += page_text + "\n"
        
        print(f"✅ Extracted {len(sample_text):,} characters from first 10 pages")
        
        # Test extraction
        witnesses = ingestion.identify_witness_names(sample_text)
        
        print(f"\n📈 RESULTS:")
        print(f"🎯 Witnesses found: {len(witnesses)}")
        
        if witnesses:
            print(f"\n📋 EXTRACTED WITNESSES:")
            for i, witness in enumerate(sorted(witnesses), 1):
                print(f"   {i:2d}. {witness}")
        
        # Check quality - should have reasonable number, not thousands
        if len(witnesses) <= 20:
            print(f"\n✅ GOOD: Reasonable number of witnesses (≤20)")
        elif len(witnesses) <= 50:
            print(f"\n⚠️  FAIR: Moderate number of witnesses (21-50)")
        else:
            print(f"\n❌ POOR: Too many witnesses (>50) - likely false positives")
        
        return witnesses
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    test_small_sample()