#!/usr/bin/env python3
"""
Test cases for PDF text cleaning and parsing.
Focus on fixing OCR artifacts, formatting issues, and broken text.
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the root directory to path
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

from Services.document_ingestion import DocumentIngestion


def test_broken_text_samples():
    """Test cleaning of real broken text samples from PDF."""
    print("=== TESTING PDF TEXT CLEANING ===")
    
    # Sample broken text from your PDF
    broken_samples = [
        # Sample 1: Broken words and weird spacing
        """he other boats, to get close to them. We pulled toward a light, 
but we **DID** not seem to get any closer to it, until daybreak. A lady back of me complained of the cold, 
and I took my coat off and gave it to her.""",
        
        # Sample 2: Random capitalization and formatting
        """We sighted the Carpathia and put the boat about and pulled 
toward her. We got alongside the Carpathia and I made the rope fast on the offside of the **LIFEBOAT**. That was hanging from the Carpathia , that rope, and I stood by until the boat was unloaded and the 
officer shouted "Come up. " """,
        
        # Sample 3: Name formatting issues
        """Senator N EWLANDS. How many boats **DID** you see loaded. Mr. W HEELTON. They were lowering No. 5 when I left to go to the storeroom, and I saw No. 7 and 
No. 9. I went away in No. 11, sir. Senator N EWLANDS. What was Mr. **ISMAY** doing. Mr. W HEELTON. He was standing aft, sir""",
        
        # Sample 4: Broken dialog and spacing
        """[Mr. Go i nto the **LIFEBOAT** and get saved. " He put his hand on her shoulder and I think he said: "Please 
get into the **LIFEBOAT** and get saved. " She replied: "No; let me stay with you. " I could not say who it 
was, but I saw that he was an old man.""",
        
        # Sample 5: Technical artifacts
        """I **DID** not pay much at tention to him, because I **DID** not know 
him. I was standing there, and I asked my brother -in-law if he could swim and he said no. I asked my 
cousin if he could swim and he said no. So we could see the water coming up, the bow of the ship was 
going down, a nd there was a kind of an explosion."""
    ]
    
    doc_ingestion = DocumentIngestion()
    
    print("\\n📋 Testing current text cleaning:")
    for i, sample in enumerate(broken_samples, 1):
        print(f"\\n--- Sample {i} ---")
        print("BEFORE:", repr(sample[:100] + "..."))
        
        cleaned = doc_ingestion._clean_extracted_text(sample)
        print("AFTER: ", repr(cleaned[:100] + "..."))
        
        # Check for remaining issues
        issues = []
        if "**" in cleaned:
            issues.append("Still has ** artifacts")
        if " DID " in cleaned:
            issues.append("Still has weird DID capitalization")
        if "LIFEBOAT" in cleaned:
            issues.append("Still has ALL CAPS words")
        if " N EWLANDS" in cleaned or " W HEELTON" in cleaned:
            issues.append("Still has broken names")
        if "- in-" in cleaned or "at tention" in cleaned:
            issues.append("Still has broken words")
            
        if issues:
            print("❌ Issues found:", ", ".join(issues))
        else:
            print("✅ No obvious issues detected")


def test_witness_name_extraction():
    """Test extraction and cleaning of witness names."""
    print("\\n=== TESTING WITNESS NAME EXTRACTION ===")
    
    test_cases = [
        "Senator N EWLANDS. How many boats did you see loaded.",
        "Mr. W HEELTON. They were lowering No. 5 when I left",
        "Senator S MITH. What is your name. Mr. L IGHTOLLER. Charles Herbert Lightoller.",
        "Mr. I SMAY. I am managing director of the White Star Line.",
        "TESTIMONY OF C HARLES HERBERT LIGHTOLLER ."
    ]
    
    for case in test_cases:
        print(f"\\nText: {case}")
        # This would test witness name extraction logic
        # We'll implement this after fixing the basic text cleaning


def test_dialog_formatting():
    """Test proper formatting of question-answer dialogs."""
    print("\\n=== TESTING DIALOG FORMATTING ===")
    
    dialog_sample = """Senator S MITH. What is your name. Mr. L IGHTOLLER. Charles Herbert Lightoller. Senator S MITH. Mr. Lightoller, where do you reside. Mr. L IGHTOLLER. Netley Abbey, Hants, England."""
    
    doc_ingestion = DocumentIngestion()
    cleaned = doc_ingestion._clean_extracted_text(dialog_sample)
    
    print("BEFORE:", dialog_sample)
    print("AFTER: ", cleaned)
    
    # Should properly separate questions and answers
    expected_patterns = [
        "Senator SMITH:",
        "Mr. LIGHTOLLER:",
        "Proper sentence breaks"
    ]
    
    print("Expected improvements needed:", expected_patterns)


if __name__ == "__main__":
    test_broken_text_samples()
    test_witness_name_extraction() 
    test_dialog_formatting()
    
    print("\\n" + "="*60)
    print("RECOMMENDATIONS FOR IMPROVEMENT:")
    print("1. Fix ** artifacts (markdown-like formatting)")
    print("2. Fix broken words (at tention → attention)")
    print("3. Fix broken names (N EWLANDS → NEWLANDS)")
    print("4. Fix random capitalization (DID → did)")
    print("5. Improve dialog formatting")
    print("6. Handle OCR spacing issues")
    print("="*60)