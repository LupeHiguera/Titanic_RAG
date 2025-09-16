# Witness Extraction Test Cases

Based on analysis of the real US Senate Inquiry format from our database content.

## Test Case 1: Standard Senate Q&A Format
**Input Text:**
```
[Testimony taken before Senator Bourne on behalf of the subcommittee.]

The witness was sworn by Senator Bourne.
Senator BOURNE. Kindly state your age, residence, and occupation.
Mr. CLENCH. Able-bodied seaman; I live at No. 10, the Flats, Chantry Road, Southampton.
Senator BOURNE. How long have you followed the sea?
Mr. CLENCH. About 19 years now, sir.
```

**Expected Output:**
- Witness Name: `Frederick Clench` (from witness.pdf we know CLENCH = Frederick Clench)
- Testimony Content: Full Q&A dialogue
- Introduction Pattern: `[Testimony taken before...]` + `The witness was sworn by...`

---

## Test Case 2: Chairman Format
**Input Text:**
```
The witness was sworn by the chairman.
Senator SMITH. Will you give your full name to the reporter?
Mr. LOWE. Harold Godfrey Lowe.
Senator SMITH. I would like to have you turn your chair so you are facing the reporter.
Mr. LOWE. Yes, sir.
```

**Expected Output:**
- Witness Name: `Harold Godfrey Lowe`
- Introduction Pattern: `The witness was sworn by the chairman.`

---

## Test Case 3: Separate Testimony Format
**Input Text:**
```
[Testimony taken separately before Senator William Alden Smith, chairman of the subcommittee.]

The witness was sworn by Senator Smith.
Senator SMITH. Mr. Buckley, where do you live?
Mr. BUCKLEY. 855 Trent Avenue, Bronx.
Senator SMITH. How old are you?
Mr. BUCKLEY. Twenty-one years old.
```

**Expected Output:**
- Witness Name: `Daniel Buckley` (from witness.pdf)
- Introduction Pattern: `[Testimony taken separately before...]`

---

## Test Case 4: OCR Artifacts Format
**Input Text:**
```
The witness was sworn by the chairman.
Senator S MITH. Will you give your name?
Mr. L IGHTOLLER. Charles Herbert Lightoller.
Senator S MITH. What is your position?
Mr. L IGHTOLLER. I was first officer of the Titanic.
```

**Expected Output:**
- Witness Name: `Charles Herbert Lightoller`
- Should handle: `S MITH` → `SMITH`, `L IGHTOLLER` → `LIGHTOLLER`

---

## Test Case 5: Multiple Witnesses in Same Text
**Input Text:**
```
[Testimony taken before Senator Bourne on behalf of the subcommittee.]

The witness was sworn by Senator Bourne.
Senator BOURNE. State your name.
Mr. CLENCH. Frederick Clench.

[Later in the document...]

The witness was sworn by Senator Smith.
Senator SMITH. Give your name to the reporter.
Mr. FLEET. Frederick Fleet.
```

**Expected Output:**
- Two separate witnesses: `Frederick Clench` and `Frederick Fleet`
- Each with their respective testimony sections

---

## Test Case 6: Title Variations
**Input Text:**
```
The witness was sworn by the chairman.
Senator SMITH. State your name and position.
Captain ROSTRON. Arthur Henry Rostron, Captain of the steamship Carpathia.
Senator SMITH. How long have you been captain?
Captain ROSTRON. About 13 years.
```

**Expected Output:**
- Witness Name: `Arthur Henry Rostron`
- Should handle `Captain` title instead of `Mr.`

---

## Test Case 7: Missing Names (Edge Case)
**Input Text:**
```
as possible, any repetition of such a disaster. Resolved further, That the committee shall inquire particularly into the number of lifeboats, rafts, and life preservers, and other equipment for the protection of the passengers and crew; the number of persons aboard the TITANIC, whether passenger or crew.
```

**Expected Output:**
- No witness extracted (this is document preamble text)
- Should not create false witnesses

---

## Test Case 8: Recalled Witness Format
**Input Text:**
```
HAROLD GODFREY LOWE, recalled.

Senator SMITH. Mr. Lowe, I want to ask you about the distress signals.
Mr. LOWE. Yes, sir.
```

**Expected Output:**
- Witness Name: `Harold Godfrey Lowe`
- Should handle `recalled` notation

---

## Current System Failures

Based on our analysis, our current system fails on:

1. **Q&A Format**: We look for `TESTIMONY OF [NAME]` but real format is `[Testimony taken before...]` + Q&A dialogue
2. **Name Extraction**: We don't extract names from `Mr. LOWE. Harold Godfrey Lowe.` responses  
3. **OCR Artifacts**: We don't handle `C HARLES HERBERT LIGHTOLLER` properly
4. **Multiple Formats**: We don't handle recalled witnesses, Captain titles, etc.

## Success Metrics

- **Target**: Extract all 77 witnesses from witness.pdf
- **Current**: Only extracting 30 witnesses (43% success rate)
- **Goal**: 95%+ success rate on witness extraction

## Key Patterns to Implement

1. `[Testimony taken before Senator...]` → New witness section
2. `The witness was sworn by...` → Witness introduction
3. `Mr./Captain/Senator NAME. Full Name Here.` → Name extraction
4. `WITNESS NAME, recalled.` → Recalled witness
5. OCR fix: `C HARLES` → `CHARLES`, `L IGHTOLLER` → `LIGHTOLLER`