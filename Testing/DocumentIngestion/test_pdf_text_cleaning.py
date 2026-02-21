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

    # Sample broken text from PDF and real Pinecone data
    broken_samples = [
        # Sample 1: Bold artifacts
        """he other boats, to get close to them. We pulled toward a light,
but we **DID** not seem to get any closer to it, until daybreak. A lady back of me complained of the cold,
and I took my coat off and gave it to her.""",

        # Sample 2: Bold formatting
        """We sighted the Carpathia and put the boat about and pulled
toward her. We got alongside the Carpathia and I made the rope fast on the offside of the **LIFEBOAT**. That was hanging from the Carpathia , that rope, and I stood by until the boat was unloaded and the
officer shouted "Come up. " """,

        # Sample 3: Broken hyphenated words
        """I did not pay much attention to him, because I did not know
him. I was standing there, and I asked my brother -in-law if he could swim and he said no.""",

        # Sample 4: OCR spacing errors
        """The witness was sworn by the chairman. Senator SMITH. Mr. Cottam, what is your full name. Mr. COTTAM. Harold Thomas Cottam.""",
    ]

    doc_ingestion = DocumentIngestion()

    print("\nTesting current text cleaning:")
    for i, sample in enumerate(broken_samples, 1):
        print(f"\n--- Sample {i} ---")
        print("BEFORE:", repr(sample[:100] + "..."))

        cleaned = doc_ingestion._clean_extracted_text(sample)
        print("AFTER: ", repr(cleaned[:100] + "..."))

        # Check for remaining issues
        issues = []
        if "**" in cleaned:
            issues.append("Still has ** artifacts")
        if "- in-" in cleaned:
            issues.append("Still has broken hyphenated words")

        if issues:
            print("Issues found:", ", ".join(issues))
        else:
            print("No obvious issues detected")


def test_bold_artifact_removal():
    """Test that ** bold artifacts are properly cleaned."""
    print("\n=== TESTING BOLD ARTIFACT REMOVAL ===")

    doc_ingestion = DocumentIngestion()

    test_cases = [
        ("ano**THE**r ship", "another ship"),
        ("his bro**THE**r was there", "his brother was there"),
        ("the wea**THE**r was clear", "the weather was clear"),
        ("**LIFEBOAT** capacity", "LIFEBOAT capacity"),
        ("he **DID** not know", "he DID not know"),
    ]

    for broken, expected_substring in test_cases:
        cleaned = doc_ingestion._fix_bold_artifacts(broken)
        # Check the core word was fixed (case-insensitive)
        core_word = expected_substring.split()[0].lower()
        if core_word in cleaned.lower():
            print(f"  '{broken}' -> '{cleaned}' (fixed)")
        else:
            print(f"  '{broken}' -> '{cleaned}' (STILL BROKEN)")


def test_hyphenated_word_fix():
    """Test that broken hyphenated words are fixed."""
    print("\n=== TESTING HYPHENATED WORD FIX ===")

    doc_ingestion = DocumentIngestion()

    test_cases = [
        ("brother -in-law", "brother-in-law"),
        ("sister -in-law", "sister-in-law"),
    ]

    for broken, expected in test_cases:
        cleaned = doc_ingestion._fix_ocr_spacing(broken)
        if expected in cleaned:
            print(f"  '{broken}' -> '{cleaned}' (fixed)")
        else:
            print(f"  '{broken}' -> '{cleaned}' (STILL BROKEN, expected '{expected}')")


def test_real_pdf_extraction_quality():
    """Test extraction quality on actual PDF files."""
    print("\n=== TESTING REAL PDF EXTRACTION ===")

    doc_ingestion = DocumentIngestion()

    pdf_path = root_dir / "Text" / "one page.pdf"
    if not pdf_path.exists():
        print("one page.pdf not found, skipping")
        return

    result = doc_ingestion.extract_text_from_pdf(pdf_path)
    text = result["text"]

    print(f"Extracted {len(text)} chars from one page.pdf")

    # With pymupdf, names should not be broken
    issues = []
    if "S MITH" in text:
        issues.append("Broken name: S MITH")
    if "I SMAY" in text:
        issues.append("Broken name: I SMAY")
    if "hea ring" in text:
        issues.append("Broken word: hea ring")
    if "Southamp ton" in text:
        issues.append("Broken word: Southamp ton")

    if issues:
        print("Issues found:", ", ".join(issues))
    else:
        print("No broken names or words detected (pymupdf extraction is clean)")


if __name__ == "__main__":
    test_broken_text_samples()
    test_bold_artifact_removal()
    test_hyphenated_word_fix()
    test_real_pdf_extraction_quality()

    print("\n" + "=" * 60)
    print("TEXT CLEANING TESTS COMPLETE")
    print("=" * 60)
