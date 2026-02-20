# Titanic Historical RAG - Project Documentation

## Project Vision & Unique Value

### Core Mission
Build the first RAG system designed specifically to **highlight contradictions** between historical witness testimonies rather than hide them. This is fundamentally different from standard RAG systems that try to find "the truth" - we embrace conflicting accounts as features, not bugs.

**Killer Feature:** Automatic contradiction detection using **PyTorch + RoBERTa-MNLI transformer**
**Tagline:** "Google for Titanic primary sources, but it shows you contradictions instead of hiding them"

---

## Current Status (January 2026)

### What's Working
- **1237 Document Chunks** in Pinecone vector database
- **FastAPI Web Application** running at http://localhost:8000
- **Semantic Search** with OpenAI text-embedding-3-large (1024 dims)
- **Basic UI** with witness filtering and search

### What's Being Built
- **Contradiction Detection** using PyTorch + RoBERTa-MNLI (NLI)
- **AWS Lambda Container** deployment to higuera.io
- **British Inquiry** document parsing support

---

## Implementation Plan

w### Phase 0: Fix Search Quality (CRITICAL)

#### 0.1 REMOVE Keyword Matching Boost (5 min - HIGHEST PRIORITY)
- **File**: `Services/semantic_search.py:133-137`
- **Problem**: Keyword matching REVERSES semantic ranking from embeddings
- **Example**: "lifeboats" returns chunks with "LIFE" + "BOAT" as separate words
- **Solution**: DELETE the keyword boost code entirely
- **Expected improvement**: 30-40% better relevance immediately

```python
# DELETE THIS CODE (lines 133-137):
query_words = set(query.text.lower().split())
content_words = set(chunk.content.lower().split())
keyword_overlap = len(query_words.intersection(content_words))
if keyword_overlap > 0:
    relevance += 0.1 * keyword_overlap  # THIS IS HARMFUL - REMOVE IT
```

#### 0.2 Fix CORS for Production
- **File**: `app.py:18-24`
- **Change**: Replace `allow_origins=["*"]` with configurable domain list

#### 0.3 Lower Similarity Threshold
- **Files**: `app.py:55`, `Services/semantic_search.py:39`
- **Change**: Default threshold from 0.7 → 0.55

---

### Phase 1: Contradiction Detection (Core Feature)

#### 1.1 Create ContradictionDetector Service
- **New file**: `Services/contradiction_detector.py`
- **Dependencies**: `transformers`, `torch` (CPU-only), `scipy`
- **NLI Model**: `roberta-large-mnli` from Hugging Face

**Core class**:
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

@dataclass
class Contradiction:
    witness1: str
    witness2: str
    claim1: str
    claim2: str
    topic: str
    confidence_score: float
    detection_method: str  # "nli_transformer", "negation", "numerical"

class ContradictionDetector:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
        self.model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")
        self.model.eval()

    def find_contradictions(self, query: str, results: List[SearchResult]) -> List[Contradiction]
    def _check_nli_contradiction(self, claim1: str, claim2: str) -> Tuple[bool, float]
    def _check_negation_patterns(self, claim1: str, claim2: str) -> Tuple[bool, float]
    def _check_numerical_contradiction(self, claim1: str, claim2: str) -> Tuple[bool, float]
```

**Detection Strategy (Hybrid)**:
1. Group results by witness
2. For each pair of statements on same topic:
   - **Step 1**: Check negation patterns ("not", "never" vs affirmative) → high confidence, free
   - **Step 2**: Extract and compare numbers → high confidence, free
   - **Step 3**: Run through RoBERTa-MNLI transformer → get contradiction probability
3. Combine scores: `confidence = max(negation_score, number_score, nli_score)`

#### 1.2 Integrate into Search Engine
- **File**: `Services/semantic_search.py`
- **Modify**: `get_related_contradictions()` (currently returns empty list)
- Wire up to ContradictionDetector

#### 1.3 Add API Endpoints
- **File**: `app.py`
- **New endpoints**:
  - `POST /search/contradictions` - Search with contradiction analysis
  - `GET /witnesses/compare?witness1=X&witness2=Y&topic=Z` - Direct comparison
  - `GET /contradictions/topics` - List known contradiction topics

#### 1.4 Frontend Contradiction UI
- **File**: `static/index.html`
- **Add**:
  - "Show Contradictions" toggle in search form
  - Contradiction card with side-by-side witness display
  - Confidence badge (%, color-coded)
  - "Compare Testimonies" button

**UI Structure**:
```
┌─ Witness 1: Ismay ─────────────────┐
│ "We were never at full speed"      │
├────────── VS ──────────────────────┤
│ CONTRADICTION: 85% confidence      │
├─ Witness 2: Lightoller ────────────┤
│ "Ship was at nearly full speed"    │
└────────────────────────────────────┘
```

---

### Phase 2: Re-Index with Better Chunking (CRITICAL FOR SEARCH)

#### 2.1 Fix Chunking Strategy
- **File**: `Services/chunking.py`
- **Root cause**: 800-char chunks bundle unrelated Q&A pairs together
- **Fix**:
  - Reduce chunk size from 800 → 400-500 characters
  - Split on SINGLE Q&A pairs, not multiple
  - Add topic keywords extraction to metadata

**New chunk metadata**:
```python
metadata = {
    "witness_name": str,
    "source_type": str,
    "page_number": int,
    "topic_keywords": list,      # ["lifeboat", "evacuation"]
    "qa_question": str,          # The question being answered
}
```

#### 2.2 British Inquiry Parser
- **File**: `Services/document_ingestion.py`
- Add `_extract_witnesses_from_british_inquiry()` method
- British Inquiry uses numbered questions and different examiner format

#### 2.3 Use Index-Based Attribution
- Leverage existing `Services/witness_index.py` (77 witnesses with page numbers)
- Bypass regex parsing issues by mapping chunks to witnesses via page ranges

#### 2.4 Re-Ingest All Data
- Clear Pinecone index
- Re-chunk with smaller size + topic metadata
- Re-embed and upload
- Test with: "lifeboats", "ship speed", "ice warnings"

---

### Phase 3: Deployment to higuera.io (AWS Lambda Container)

**Why Lambda Container**: Fits PyTorch (~2GB), pay-per-request (~$0-5/month), serverless.
**Note**: Cold starts ~60-90s due to model loading. Fine for portfolio.

#### 3.1 Create Dockerfile for Lambda
```dockerfile
FROM public.ecr.aws/lambda/python:3.12

# Install PyTorch CPU-only (smaller)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Download and cache the model at build time
RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
    AutoTokenizer.from_pretrained('roberta-large-mnli'); \
    AutoModelForSequenceClassification.from_pretrained('roberta-large-mnli')"

# Copy app code
COPY . .

CMD ["app.handler"]
```

#### 3.2 Add Mangum Adapter
- **File**: `app.py`
```python
from mangum import Mangum
handler = Mangum(app)
```

#### 3.3 Deploy Steps
1. Install AWS SAM CLI: `brew install aws-sam-cli`
2. Build container: `sam build`
3. Deploy: `sam deploy --guided`
4. Set up custom domain in API Gateway console
5. Point titanic.higuera.io to API Gateway URL

---

## Execution Order

| Step | Task | Effort |
|------|------|--------|
| 1 | Phase 0.1: Remove keyword boost (CRITICAL) | 5 min |
| 2 | Phase 0.2-0.3: CORS + threshold fixes | 30 min |
| 3 | Phase 2.1: Fix chunking strategy | 1 day |
| 4 | Phase 2.2-2.3: British parser + index attribution | 1 day |
| 5 | Phase 2.4: Re-ingest all data | 1 day |
| 6 | Phase 1.1: ContradictionDetector with PyTorch | 1-2 days |
| 7 | Phase 1.2-1.4: Integration + API + UI | 2 days |
| 8 | Phase 3: Lambda container deploy | 1-2 days |

**Note**: Search quality fixes FIRST - good search is prerequisite for contradiction detection.

---

## Files to Modify

| File | Changes |
|------|---------|
| `app.py` | Fix threshold, fix CORS, add contradiction endpoints, add Mangum handler |
| `Services/semantic_search.py` | Fix threshold, integrate ContradictionDetector |
| `Services/contradiction_detector.py` | NEW - RoBERTa-MNLI NLI + negation/numerical detection |
| `static/index.html` | Add contradiction UI components |
| `requirements.txt` | Add `transformers`, `torch`, `mangum` |
| `Dockerfile` | NEW - Lambda container with PyTorch + model |
| `template.yaml` | NEW - AWS SAM template for container Lambda |
| `Services/document_ingestion.py` | Fix British Inquiry parsing (Phase 2) |

## New Dependencies

```
transformers>=4.30.0   # Hugging Face transformers (RoBERTa-MNLI)
torch                  # PyTorch (CPU-only via --index-url)
scipy                  # Required by transformers
mangum>=0.17.0         # FastAPI -> Lambda adapter
```

---

## Resume Keywords

- **PyTorch** - Deep learning framework
- **Transformers / Hugging Face** - NLP model library
- **RoBERTa-MNLI** - Pre-trained NLI model
- **Natural Language Inference (NLI)** - Text entailment/contradiction classification
- **AWS Lambda** - Serverless compute
- **Containerized ML** - Docker + ML models
- **RAG** - Retrieval-Augmented Generation
- **Pinecone** - Vector database
- **FastAPI** - Modern Python web framework

---

## Verification

1. **Local search works**: `curl -X POST http://localhost:8000/search -d '{"query": "iceberg", "similarity_threshold": 0.5}'`
2. **NLI model loads**: ContradictionDetector initializes RoBERTa-MNLI without errors
3. **Contradictions detected**: Query "ship speed" returns contradictions with NLI classification
4. **UI displays contradictions**: Side-by-side view with confidence score
5. **Container builds**: `sam build` completes successfully
6. **Production**: https://titanic.higuera.io loads and searches work

---

*Last Updated: January 2026*
*Status: Implementing Contradiction Detection with PyTorch + RoBERTa-MNLI*
*Deployment Target: AWS Lambda Container → titanic.higuera.io*