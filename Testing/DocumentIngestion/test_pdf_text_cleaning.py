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
    
    # Sample broken text from your PDF and real Pinecone data
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
going down, a nd there was a kind of an explosion.""",
        
        # Sample 6: REAL PINECONE DATA - Cottam testimony with OCR spacing issues
        """The witness was sworn by the chairman. Senator SMITH. Mr. Cottam, w hat is your full name. Mr. COTTAM. Harold Thomas Cottam. Senator SMITH. Where do you reside. Mr. COTTAM. Liverpool, England. Senator SMITH. How old are you. Mr. COTTAM. Twenty-one. Senator SMITH. What is your business. Mr. COTTAM. Marconi tel egraphist. Senator SMITH. How long have you been engaged in that business. Mr. COTTAM. Three years. Senator SMITH. Where have you been employed. Mr. COTTAM. The Marconi Co. all the time.""",
        
        # Sample 7: REAL PINECONE DATA - More OCR issues
        """on one of their land stations. Senator SMITH. Under the Britis h post-office authorities. Mr. COTTAM. Yes, sir. Senator SMITH. Where. Mr. COTTAM. Liverpool. Senator SMITH. How long were you thus employed. Mr. COTTAM. About 14 to 16 months. Senator SMITH. Then what did you do. Mr. COTTAM. I was taken off there and went away to sea again, on the Australian run.""",
        
        # Sample 8: REAL PINECONE DATA - Technical terms and name breaking
        """kind of apparatus was there on the Medic. Mr. COTTAM. A Marconi, sir. Senator SMITH. What type of instrument or equipment. Mr. COTTAM. A one and a half watt set, sir. Senator SMITH. What was the maximum wave length. Mr. COTTAM. A sta ndard wave length, sir; 2,000 feet. Senator SMITH. What was your next employment. Mr. COTTA M. On the Carpathia, sir.""",
        
        # Sample 9: NEW PINECONE DATA - Additional OCR spacing issues
        """of the British Government. Mr. COTTAM. No, sir. Senator SMITH. No dir ection by the Marconi Co. Mr. COTTAM. No, sir; but you are more or less responsible for communications which are\nexpected. Senator SMITH. You are responsible for communication. Mr. COTTAM. Yes, sir; if there is a ship expected, sir. If a ship is exp ected to pass at 3 o'clock in\nthe morning you should be at duty at that time to establish communication. Senator SMITH. Has it been your custom to go to the apparatus at regular times. Mr. COTTAM. No, sir. Senator SMITH. Are you employed at anything else on the boat. Mr. COTTAM. No, sir. Senator SMITH. What wages do you receive. Mr. COTTAM. Four pounds ten a month. Senator SMITH. Four pounds ten shillings a month. Mr. COTTAM. Yes, sir. Senator SMITH. And board. Mr. COTTAM. Yes, sir. Sena tor SMITH. And room. Mr. COTTAM.""",
        
        # Sample 10: ADDITIONAL PINECONE DATA - More OCR patterns
        """the captain. Mr. COTTAM. Yes, sir. Senator SMITH. Then, after you had taken this message to the captain, you came back to your\ninstrument and sent the message that you have just described. Mr. COTTAM. Yes, sir. Senator SMITH. And to that you received no reply. Mr. COTTAM. No, sir. Senator SMITH. And you never received any other reply. Mr. COTTAM. No, sir. Senator SMITH. Or any other word from the ship. Mr. COTTAM. No, sir. Senator SMITH. After the Carpathia had picked up these lifeboats and started for New York, did\nyou receive messages. Mr. COTTAM. Yes, sir. Senator SMITH. How long did you remain at your post that night. Mr. COTTAM. All the night, sir. Senator SMITH. How mu ch of the time next day. Mr. COTTAM. All the day, sir. Senator SMITH. That was Sunday and Monday; how about Monday night. Mr. COTTAM. I was on all night again, sir."""
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
        
        # Check for OCR spacing patterns automatically using regex
        import re
        
        # Find potential OCR spacing errors: word + space + 1-3 letters
        ocr_patterns = re.findall(r'\b([a-z]{2,}) ([a-z]{1,3})\b', cleaned.lower())
        
        # Filter out legitimate phrases
        legitimate_phrases = [
            'i am', 'to go', 'we had', 'he was', 'it was', 'you are', 'on a', 'in a', 'at a',
            'of a', 'for a', 'by a', 'up to', 'as to', 'so to', 'go to', 'do you', 'if you',
            'or no', 'yes or', 'is no', 'was no', 'had no', 'get up', 'put up', 'set up',
            'the us', 'was on', 'get on', 'put on', 'go on', 'come on', 'hold on'
        ]
        
        suspicious_patterns = []
        for word1, word2 in ocr_patterns:
            phrase = f"{word1} {word2}"
            if phrase not in legitimate_phrases and len(word2) <= 2:
                suspicious_patterns.append(phrase)
        
        # Remove duplicates
        suspicious_patterns = list(set(suspicious_patterns))
        
        if suspicious_patterns:
            issues.append(f"Potential OCR spacing errors: {suspicious_patterns[:5]}")  # Show first 5
            
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