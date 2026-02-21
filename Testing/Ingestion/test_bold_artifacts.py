#!/usr/bin/env python3
"""
Test case specifically for the bold word artifacts issue you reported.
"""

import sys
from pathlib import Path

# Add the root directory to path
root_dir = Path(__file__).parent
sys.path.append(str(root_dir))

from Services.document_ingestion import DocumentIngestion

def test_your_specific_issue():
    """Test the exact text you provided with bold word artifacts."""
    
    # Your exact text with the issues
    problematic_text = """that came to you **WAS** under sail. Mr. EVANS. After we left **THE** wreckage we made sail to ano**THE**r boat that **WAS** in distress, far**THE**r
over. Senator SMITH. That **WAS** Lowe's boat, **WAS** it not. Mr. EVANS. Yes. Senator SMITH. **WHEN** you picked up **THE**se four men, that left you 13 people in your boat. Mr. EVANS. Thirteen; yes, sir. Senator SMITH. Did you see o**THE**r people in **THE** water, or hear **THE**ir cries. Mr. EVAN S. No, sir; none whatsoever, sir, o**THE**r than **THE**se four persons we picked up. Senator SMITH. Did you not hear **THE** cries of anyone in distress. Mr. EVANS. No, sir. For help. Mr. EVANS. In **THE** first place, **WHEN** **THE** **SHIP** sank I **WAS** in No. 10 bo at, **THE**n, sir. Senator SMITH. **WHEN** **THE** **SHIP** sank you heard **THE**se cries. Mr. EVANS."""
    
    print("=== TESTING YOUR SPECIFIC BOLD ARTIFACTS ISSUE ===")
    print("BEFORE:")
    print(problematic_text[:200] + "...")
    print()
    
    doc_ingestion = DocumentIngestion()
    cleaned = doc_ingestion._clean_extracted_text(problematic_text)
    
    print("AFTER:")
    print(cleaned[:200] + "...")
    print()
    
    # Debug: Show the actual transformation for specific words
    print("SPECIFIC TRANSFORMATIONS:")
    if "ano**THE**r" in problematic_text:
        print(f"ano**THE**r found in original")
    if "another" in cleaned:
        print(f"another found in cleaned text")
    if "ano" in cleaned and "r boat" in cleaned:
        print(f"Broken: found 'ano' and 'r boat' separately")
    
    # Check if issues remain
    remaining_issues = []
    
    if "**" in cleaned:
        bold_words = [word for word in cleaned.split() if "**" in word]
        remaining_issues.append(f"Bold artifacts remain: {bold_words[:5]}")
    
    if "**THE**" in cleaned:
        remaining_issues.append("Still has **THE**")
    
    if "**WAS**" in cleaned:
        remaining_issues.append("Still has **WAS**")
        
    if "**WHEN**" in cleaned:
        remaining_issues.append("Still has **WHEN**")
        
    if "**SHIP**" in cleaned:
        remaining_issues.append("Still has **SHIP**")
    
    # Check for broken words like "ano**THE**r" → should become "another"
    if "ano" in cleaned and "r boat" in cleaned:
        remaining_issues.append("Broken word 'another' not fixed")
        
    if "far" in cleaned and "r over" in cleaned:
        remaining_issues.append("Broken word 'farther' not fixed")
        
    if "o" in cleaned and "r people" in cleaned:
        remaining_issues.append("Broken word 'other' not fixed")
    
    print("ANALYSIS:")
    if remaining_issues:
        print("❌ Issues still present:")
        for issue in remaining_issues:
            print(f"   - {issue}")
    else:
        print("✅ All bold artifacts successfully cleaned!")
    
    return cleaned, remaining_issues

if __name__ == "__main__":
    cleaned_text, issues = test_your_specific_issue()
    
    if issues:
        print("\n" + "="*60)
        print("NEED TO IMPROVE TEXT CLEANING FOR:")
        print("1. Individual bold words: **THE**, **WAS**, **WHEN**, **SHIP**")
        print("2. Broken words with bold in middle: ano**THE**r → another")
        print("3. Make sure bold removal doesn't break adjacent text")
        print("="*60)
    else:
        print("\n✅ Text cleaning working perfectly!")