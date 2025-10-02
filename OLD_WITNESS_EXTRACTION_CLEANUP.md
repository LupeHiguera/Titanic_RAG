# Old Witness Extraction System - Cleanup Guide

## 🔥 **TRANSITION TO INDEX-BASED WITNESS EXTRACTION**

This document tracks the removal of obsolete text-parsing witness extraction in favor of precise index-based attribution.

---

## **📋 FILES TO REMOVE COMPLETELY**

### **Obsolete Test Files**
```bash
# Remove these files - they test OCR/regex patterns we're abandoning
rm Testing/test_witness_extraction.py
rm Testing/WitnessExtraction/test_witness_extraction.py
rm Testing/test_focused_extraction.py
rm Testing/test_real_extraction.py  
rm Testing/test_small_sample.py
rm Testing/Ingestion/reprocess_all_witnesses.py
```

### **Why Removing:**
- Test regex pattern matching for witness names ❌ 
- Test OCR artifact cleaning ❌
- Test Q&A format parsing ❌
- **INDEX PROVIDES PERFECT WITNESS DATA** ✅

---

## **🔧 FILES TO UPDATE - REMOVE OBSOLETE CODE**

### **Services/chunking.py** 
**Remove these methods (lines 269-448):**
- `_extract_witness_contexts_from_text()` - Line 269
- `_extract_qa_contexts_from_text()` - Line 296  
- `_find_testimony_sections()` - Line 309
- `_extract_witnesses_from_section()` - Line 327
- `_extract_full_name_from_response()` - Line 393
- `_clean_witness_name()` - Line 411
- `_fix_spaced_name()` - Line 429

**Keep these methods:**
- `chunk_witness_contexts()` ✅
- `_split_text_preserving_context()` ✅ 
- All chunking/overlap logic ✅
- Metadata handling ✅

### **Testing/Chunking/test_chunking.py**
**Remove obsolete test methods:**
- Lines 74-84: `test_chunk_preserves_witness_identity()` (tests text parsing)
- Lines 120-133: `test_chunk_preserves_question_answer_pairs()` (tests Q&A parsing)
- Lines 134-143: `test_chunk_avoids_splitting_mid_sentence()` (sentence parsing logic)

**Keep and update:**
- Chunking size tests ✅
- Metadata tests ✅  
- Content processing tests ✅

---

## **✅ NEW INDEX-BASED SYSTEM**

### **Witness Index Data (from US Senate Inquiry)**
```python
WITNESS_INDEX = [
    {"name": "J. Bruce Ismay", "role": "Managing Director White Star Line, Titanic passenger", "page": 2},
    {"name": "Arthur Henry Rostron", "role": "Captain, Carpathia", "page": 18}, 
    {"name": "Guglielmo Marconi", "role": "Chairman, British Marconi Co.", "page": 37},
    {"name": "Charles Herbert Lightoller", "role": "2nd Officer, Titanic", "page": 46},
    {"name": "Harold Thomas Cottam", "role": "Marconi Operator, Carpathia", "page": 95},
    {"name": "Alfred Crawford", "role": "Steward, Titanic", "page": 111},
    {"name": "Harold S. Bride", "role": "Marconi Operator, Titanic", "page": 133},
    {"name": "Herbert John Pitman", "role": "3rd Officer, Titanic", "page": 166},
    {"name": "Philip A. S. Franklin", "role": "Vice President, IMM", "page": 169},
    {"name": "Joseph Groves Boxhall", "role": "4th Officer, Titanic", "page": 209},
    {"name": "Frederick Fleet", "role": "Lookout, Titanic", "page": 315},
    {"name": "Major Arthur G. Peuchen", "role": "1st Class passenger, Titanic", "page": 329},
    {"name": "Harold Godfrey Lowe", "role": "5th Officer, Titanic", "page": 368},
    {"name": "Robert Hichens", "role": "Quartermaster, Titanic", "page": 449},
    {"name": "George Thomas Rowe", "role": "Quartermaster, Titanic", "page": 519},
    {"name": "Alfred Olliver", "role": "Quartermaster, Titanic", "page": 526},
    {"name": "Frank Osman", "role": "Seaman, Titanic", "page": 537},
    {"name": "Edward Wheelton", "role": "Steward, Titanic", "page": 543},
    {"name": "W. H. Taylor", "role": "Fireman, Titanic", "page": 550},
    {"name": "George Moore", "role": "Seaman, Titanic", "page": 559},
    {"name": "Thomas Jones", "role": "Seaman, Titanic", "page": 566},
    {"name": "G. Symons", "role": "Lookout, Titanic", "page": 573},
    {"name": "G. A. Hogg", "role": "Lookout, Titanic", "page": 577},
    {"name": "Walter John Perkis", "role": "Quartermaster, Titanic", "page": 580},
    {"name": "John Hardy", "role": "Steward, Titanic", "page": 587},
    {"name": "William Ward", "role": "Seaman", "page": 595},
    {"name": "James Widgery", "role": "Steward, Titanic", "page": 601},
    {"name": "Edward John Buley", "role": "Seaman, Titanic", "page": 603},
    {"name": "George Frederick Crowe", "role": "Steward, Titanic", "page": 613},
    {"name": "C. E. Andrews", "role": "Steward, Titanic", "page": 622},
    {"name": "John Collins", "role": "Cook, Titanic", "page": 627},
    {"name": "Frederick Clench", "role": "Seaman, Titanic", "page": 634},
    {"name": "Ernest Archer", "role": "Seaman, Titanic", "page": 643},
    {"name": "W. Brice", "role": "Seaman, Titanic", "page": 648},
    {"name": "Albert Haines", "role": "Boatswain's Mate, Titanic", "page": 655},
    {"name": "Samuel S. Hemming", "role": "Seaman, Titanic", "page": 662},
    {"name": "Frank Oliver Evans", "role": "Seaman, Titanic", "page": 673},
    {"name": "Ernest Gill", "role": "Donkeyman, Californian", "page": 710},
    {"name": "Stanley Lord", "role": "Captain, Californian", "page": 714},
    {"name": "Cyril Furmstone Evans", "role": "Marconi Operator, Californian", "page": 733},
    {"name": "James Henry Moore", "role": "Captain, Mount Temple", "page": 757},
    {"name": "Andrew Cunningham", "role": "Steward, Titanic", "page": 790},
    {"name": "Frederick D. Ray", "role": "Steward, Titanic", "page": 798},
    {"name": "Henry Samuel Etches", "role": "Steward, Titanic", "page": 810},
    {"name": "William Burke", "role": "Steward, Titanic", "page": 821},
    {"name": "Arthur John Bright", "role": "Quartermaster, Titanic", "page": 831},
    {"name": "Frederick M. Sammis", "role": "Chief Engineer, Marconi Wireless Telegraph Co. of America.", "page": 845},
    {"name": "Hugh Woolner", "role": "1st Class passenger, Titanic", "page": 860},
    {"name": "Edward J. Dunn", "role": "Salesman", "page": 935},
    {"name": "Charles H. Morgan", "role": "Deputy United States Marshal", "page": 937},
    {"name": "C. E. Henry Stengel", "role": "1st Class passenger, Titanic", "page": 970},
    {"name": "Archibald Gracie", "role": "1st Class passenger, Titanic", "page": 989},
    {"name": "Helen W. Bishop", "role": "1st Class passenger, Titanic", "page": 998},
    {"name": "Dickinson H. Bishop", "role": "1st Class passenger, Titanic", "page": 1000},
    {"name": "Mrs. J. Stuart White", "role": "1st Class passenger, Titanic", "page": 1005},
    {"name": "John Bottomley", "role": "Vice president, Marconi Wireless Telegraph Co. of America.", "page": 1010},
    {"name": "Daniel Buckley", "role": "3rd Class passenger, Titanic", "page": 1019},
    {"name": "Melville E. Stone", "role": "General Manager, Associated Press", "page": 1023},
    {"name": "George A. Harder", "role": "1st Class passenger, Titanic", "page": 1028},
    {"name": "John R. Binns", "role": "Ex-Marconi Operator, Republic", "page": 1032},
    {"name": "Olaus Abelseth", "role": "3rd Class passenger, Titanic", "page": 1036},
    {"name": "Norman Campbell Chambers", "role": "1st Class passenger, Titanic", "page": 1041},
    {"name": "Frederick Dauler", "role": "Clerk, Western Union Telegraph Co.", "page": 1047},
    {"name": "Berk Pickard", "role": "3rd Class passenger, Titanic", "page": 1054},
    {"name": "Gilbert William Balfour", "role": "Inspector, Marconi Co.", "page": 1056},
    {"name": "Maurice L. Farrell", "role": "Managing News Editor, Dow Jones Co.", "page": 1065},
    {"name": "Benjamin Campbell", "role": "Vice President, New York, New Haven & Hartford Railroad Co.", "page": 1103},
    {"name": "John J. Knapp", "role": "United States Navy, Hydrographer", "page": 1111},
    {"name": "Herbert James Haddock", "role": "Captain, Olympic", "page": 1127},
    {"name": "Frederick Barrett", "role": "Fireman, Titanic", "page": 1140}
]
```

### **Recalled Witnesses (Multiple Appearances)**
```python
RECALLED_WITNESSES = [
    {"name": "Harold Thomas Cottam", "pages": [95, 121, 154, 494, 918]},
    {"name": "Harold S. Bride", "pages": [133, 154, 896, 1051]}, 
    {"name": "Herbert John Pitman", "pages": [166, 259]},
    {"name": "Frederick Fleet", "pages": [315, 357]},
    {"name": "Charles Herbert Lightoller", "pages": [46, 421, 755, 785]},
    {"name": "Guglielmo Marconi", "pages": [37, 463, 515, 845]},
    {"name": "Philip A. S. Franklin", "pages": [169, 688, 787]},
    {"name": "Joseph Groves Boxhall", "pages": [209, 907, 930]},
    {"name": "Frank Oliver Evans", "pages": [673, 749]},
    {"name": "Alfred Crawford", "pages": [111, 826, 842]},
    {"name": "J. Bruce Ismay", "pages": [2, 938, 981]}
]
```

---

## **🎯 NEW TEST CASES TO CREATE**

### **Test Case 1: LightHoller Sample Page Mapping**
```python
def test_lightoller_page_46_mapping():
    """Test that LightHollerSample.pdf maps correctly to page 46 witness."""
    # LightHollerSample.pdf should start at page 46
    # Should map to: Charles Herbert Lightoller, 2nd Officer, Titanic
    pass
```

### **Test Case 2: Index-Based Witness Attribution** 
```python
def test_index_based_witness_attribution():
    """Test witness attribution using page numbers from index."""
    # Page 95 -> Harold Thomas Cottam
    # Page 315 -> Frederick Fleet  
    # Page 449 -> Robert Hichens
    pass
```

### **Test Case 3: Recalled Witness Handling**
```python
def test_recalled_witness_testimonies():
    """Test handling of witnesses with multiple appearances."""
    # Charles Herbert Lightoller: pages 46, 421, 755, 785
    # Should create separate testimony contexts for each appearance
    pass
```

### **Test Case 4: Witness Role Classification**
```python
def test_witness_role_classification():
    """Test witness categorization by role."""
    officers = ["2nd Officer", "3rd Officer", "4th Officer", "5th Officer"] 
    crew = ["Steward", "Seaman", "Lookout", "Quartermaster"]
    passengers = ["1st Class passenger", "3rd Class passenger"]
    pass
```

---

## **📍 LIGHTHHOLLER SAMPLE VERIFICATION**

**Expected Mapping:**
- **File**: `Text/LightHollerSample.pdf` 
- **Witness**: Charles Herbert Lightoller
- **Role**: 2nd Officer, Titanic
- **Index Page**: 46
- **Should Contain**: "TESTIMONY OF CHARLES HERBERT LIGHTOLLER"

**Verification Steps:**
1. Extract page numbers from LightHollerSample.pdf
2. Confirm it starts at page 46  
3. Verify witness attribution matches index
4. Test new page-based chunking system

---

## **🚀 IMPLEMENTATION ORDER**

1. ✅ Create this cleanup documentation  
2. ⏳ Parse witness index into structured data
3. ⏳ Test LightHoller page mapping
4. ⏳ Remove obsolete code from chunking.py
5. ⏳ Create new index-based test cases  
6. ⏳ Update chunking system to use page-based attribution
7. ⏳ Remove obsolete test files
8. ⏳ Verify all tests pass with new system

---

**Status: Ready to implement index-based witness extraction system** 🎯