"""Regression for bold OCR artifacts, including markup inside words."""
from Services.document_ingestion import DocumentIngestion


def test_bold_artifacts_preserve_whole_words():
    problematic_text = "that came to you **WAS** under sail. Mr. EVANS. After we left **THE** wreckage we made sail to ano**THE**r boat that **WAS** in distress, far**THE**r\nover. Senator SMITH. That **WAS** Lowe's boat, **WAS** it not. Mr. EVANS. Yes. Senator SMITH. **WHEN** you picked up **THE**se four men, that left you 13 people in your boat. Mr. EVANS. Thirteen; yes, sir. Senator SMITH. Did you see o**THE**r people in **THE** water, or hear **THE**ir cries. Mr. EVAN S. No, sir; none whatsoever, sir, o**THE**r than **THE**se four persons we picked up. Senator SMITH. Did you not hear **THE** cries of anyone in distress. Mr. EVANS. No, sir. For help. Mr. EVANS. In **THE** first place, **WHEN** **THE** **SHIP** sank I **WAS** in No. 10 bo at, **THE**n, sir. Senator SMITH. **WHEN** **THE** **SHIP** sank you heard **THE**se cries. Mr. EVANS."
    cleaned = DocumentIngestion()._clean_extracted_text(problematic_text)
    assert "**" not in cleaned
    assert "another boat" in cleaned
    assert "farther" in cleaned.split()
    assert "other people" in cleaned
    assert "these four persons" in cleaned.lower()
