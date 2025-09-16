#!/usr/bin/env python3
"""
Test improved witness extraction on real USInq.pdf document.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from Services.document_ingestion import DocumentIngestion

def test_real_document_extraction():
    """Test our improved extraction on the actual USInq.pdf."""
    
    print("🔍 Testing Improved Witness Extraction on Real USInq.pdf")
    print("=" * 60)
    
    # Initialize extraction service
    ingestion = DocumentIngestion()
    
    # Path to the real document
    usinq_path = Path("Text/USInq.pdf")
    
    if not usinq_path.exists():
        print(f"❌ USInq.pdf not found at {usinq_path}")
        return
    
    print(f"📄 Processing document: {usinq_path}")
    print("⏳ This may take a moment for a large PDF...")
    
    try:
        # Extract text and metadata
        result = ingestion.extract_text_from_pdf(usinq_path)
        text = result['text']
        
        print(f"✅ Document processed successfully")
        print(f"📊 Text length: {len(text):,} characters")
        
        # Test our improved witness extraction
        print("\n🔍 Extracting witnesses with improved patterns...")
        witnesses = ingestion.identify_witness_names(text)
        
        print(f"\n📈 EXTRACTION RESULTS:")
        print(f"🎯 Witnesses found: {len(witnesses)}")
        print(f"🎯 Target witnesses: 77 (from witness.pdf)")
        
        if witnesses:
            print(f"\n📋 EXTRACTED WITNESSES:")
            for i, witness in enumerate(sorted(witnesses), 1):
                print(f"   {i:2d}. {witness}")
        
        # Load expected witnesses from witness.pdf reference
        expected_witnesses = get_expected_witnesses()
        
        print(f"\n🔍 COMPARISON WITH EXPECTED WITNESSES:")
        
        # Find matches
        found_expected = []
        missing_expected = []
        
        for expected in expected_witnesses:
            # Check for exact match or partial match
            found = False
            for extracted in witnesses:
                if (expected.lower() in extracted.lower() or 
                    extracted.lower() in expected.lower() or
                    expected.split()[-1].lower() == extracted.split()[-1].lower()):  # Same surname
                    found_expected.append((expected, extracted))
                    found = True
                    break
            
            if not found:
                missing_expected.append(expected)
        
        # Find unexpected witnesses (false positives)
        expected_surnames = {name.split()[-1].lower() for name in expected_witnesses}
        unexpected_witnesses = []
        for extracted in witnesses:
            surname = extracted.split()[-1].lower()
            if surname not in expected_surnames:
                unexpected_witnesses.append(extracted)
        
        print(f"✅ Found expected: {len(found_expected)}/{len(expected_witnesses)} ({len(found_expected)/len(expected_witnesses)*100:.1f}%)")
        print(f"❌ Missing expected: {len(missing_expected)}")
        print(f"⚠️  Unexpected: {len(unexpected_witnesses)}")
        
        if found_expected:
            print(f"\n✅ SUCCESSFULLY MATCHED WITNESSES:")
            for expected, extracted in found_expected[:10]:  # Show first 10
                print(f"   ✓ {expected} → {extracted}")
            if len(found_expected) > 10:
                print(f"   ... and {len(found_expected) - 10} more")
        
        if missing_expected:
            print(f"\n❌ MISSING EXPECTED WITNESSES (first 10):")
            for missing in missing_expected[:10]:
                print(f"   ✗ {missing}")
            if len(missing_expected) > 10:
                print(f"   ... and {len(missing_expected) - 10} more")
        
        if unexpected_witnesses:
            print(f"\n⚠️  UNEXPECTED WITNESSES (possible false positives):")
            for unexpected in unexpected_witnesses[:5]:
                print(f"   ? {unexpected}")
            if len(unexpected_witnesses) > 5:
                print(f"   ... and {len(unexpected_witnesses) - 5} more")
        
        # Calculate success metrics
        success_rate = len(found_expected) / len(expected_witnesses) * 100
        precision = len(found_expected) / len(witnesses) * 100 if witnesses else 0
        
        print(f"\n📊 PERFORMANCE METRICS:")
        print(f"🎯 Recall (found/expected): {success_rate:.1f}%")
        print(f"🎯 Precision (correct/total): {precision:.1f}%")
        
        # Status assessment
        if success_rate >= 95:
            print(f"🎉 EXCELLENT: Ready for production!")
        elif success_rate >= 80:
            print(f"✅ GOOD: Minor improvements needed")
        elif success_rate >= 60:
            print(f"⚠️  FAIR: Significant improvements needed")
        else:
            print(f"❌ POOR: Major extraction issues")
            
        return {
            'witnesses_found': len(witnesses),
            'witnesses_expected': len(expected_witnesses),
            'success_rate': success_rate,
            'precision': precision,
            'found_expected': found_expected,
            'missing_expected': missing_expected,
            'unexpected_witnesses': unexpected_witnesses
        }
        
    except Exception as e:
        print(f"❌ Error processing document: {e}")
        return None

def get_expected_witnesses():
    """Get list of expected witnesses from witness.pdf analysis."""
    # Based on witness.pdf, these are some of the key witnesses we should find
    return [
        "J. Bruce Ismay",
        "Charles Herbert Lightoller", 
        "Harold Godfrey Lowe",
        "Joseph Groves Boxhall",
        "Herbert John Pitman",
        "Frederick Fleet",
        "Arthur Henry Rostron",
        "Harold Thomas Cottam",
        "Harold Sydney Bride",
        "Alfred Crawford",
        "Daniel Buckley",
        "Frederick Clench",
        "Henry Samuel Etches",
        "George Frederick Crowe",
        "Edward John Buley",
        "Cyril Furmstone Evans",
        "Albert Haines",
        "John Hardy",
        "Thomas Jones",
        "Edward Wheelton",
        "William Ward",
        "Walter John Perkis",
        "Ernest Gill",
        "John Collins",
        "Olaus Abelseth",
        "James Widgery",
        "Andrew Cunningham",
        "William Burke",
        "Frederick Dauler",
        "Archibald Gracie",
        # Add more from the full 77 witness list as needed
    ]

if __name__ == "__main__":
    test_real_document_extraction()