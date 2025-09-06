# Titanic Historical RAG - Project Documentation

## 🎯 **Project Vision & Unique Value**

### **Core Mission**
Build the first RAG system designed specifically to **highlight contradictions** between historical witness testimonies rather than hide them. This is fundamentally different from standard RAG systems that try to find "the truth" - we embrace conflicting accounts as features, not bugs.

**Killer Feature:** **Automatic contradiction detection and visual highlighting** between witness testimonies  
**Tagline:** "Google for Titanic primary sources, but it shows you contradictions instead of hiding them"

---

## 🚀 **CURRENT STATUS: FUNCTIONAL FOUNDATION, MISSING KILLER FEATURE**

### ✅ **What's Working (Foundation Complete)**
- **30 Unique Witnesses** indexed from US Senate Inquiry
- **381 Document Chunks** with proper witness attribution  
- **FastAPI Web Application** running at http://localhost:8001
- **Basic Search Engine** with witness filtering
- **Vector Storage** with ChromaDB (381 chunks stored)

### 🔥 **CRITICAL MISSING: The Contradiction Highlighting System**
**This is our unique differentiator and main value proposition - currently NOT implemented:**

```
❌ Automatic contradiction detection between witnesses
❌ Side-by-side contradiction visual display
❌ Contradiction confidence scoring  
❌ "Show me conflicting accounts" functionality
❌ Visual highlighting of contradictory statements
```

**Example of what we SHOULD be able to do:**
```
Query: "How many people were in Ismay's lifeboat?"
Expected Result:
┌─ Ismay: "About 45 people" ────────────┐  
│                                       │
│  CONTRADICTION DETECTED! ⚠️          │
│                                       │  
└─ Officer Lowe: "Only 12 people" ─────┘
```

---

## 🔧 **KNOWN ISSUES & IMPROVEMENT AREAS**

### **Priority 1: KILLER FEATURE MISSING (CRITICAL)**
- **Contradiction Detection**: Framework exists in `chunking.py` but not integrated into search
- **Visual Highlighting**: No UI component for showing conflicting testimonies
- **Comparison Engine**: Need to build witness account comparison system
- **This is what makes us different from every other RAG system!**

### **Priority 2: Document Quality Issues (High Impact)**  
- **OCR Artifacts**: Weird capital letters ("C HARLES HERBERT LIGHTOLLER")
- **Text Normalization**: Inconsistent formatting in extracted documents
- **Witness Name Parsing**: Some names malformed due to OCR issues

### **Priority 3: Search Result Quality (Medium Impact)**
- **Relevance**: Search results functional but not always optimal
- **False Positives**: Some irrelevant results appearing
- **Ranking**: Algorithm needs tuning for better result ordering

---

## 🏗️ **TECHNICAL ARCHITECTURE**

### **Current Pipeline (Missing Key Component)**
```
📄 PDF Documents → 🔍 Witness Extraction (OCR issues) → ✂️ Chunking → 
🧠 Embeddings → 💾 Vector Storage → 🔍 Basic Search →
❌ MISSING: Contradiction Detection & Highlighting
```

### **Services Status**
- **document_ingestion.py**: 🔧 Working but OCR quality issues
- **chunking.py**: ✅ Working, has contradiction framework (unused)
- **embeddings.py**: ✅ Solid implementation
- **vector_storage.py**: ✅ Reliable ChromaDB storage
- **semantic_search.py**: 🔧 Basic search works, needs contradiction integration
- **app.py**: ✅ FastAPI endpoints functional

### **What We Have vs. What We Need**
```
✅ Document Processing Pipeline
✅ Witness Attribution System  
✅ Vector Search Capability
✅ Web API Interface

❌ Contradiction Detection Engine
❌ Conflict Visualization UI
❌ Multi-Witness Comparison
❌ Contradiction Confidence Scoring
```

---

## 🎯 **DEVELOPMENT PRIORITIES - FOCUSED ON KILLER FEATURE**

### **Phase 1: Implement Contradiction Detection (2-3 weeks) - CRITICAL**

#### **Week 1: Core Contradiction Engine**
```python
# Enhance Services/semantic_search.py:
- Build contradiction detection algorithm
- Implement witness account comparison 
- Add contradiction confidence scoring
- Create conflict identification system
```

#### **Week 2: Contradiction API & Logic**
```python
# New API endpoints in app.py:
- POST /search/contradictions - Find conflicting accounts
- GET /witnesses/compare - Compare specific witnesses
- POST /analyze/conflicts - Analyze contradiction patterns
```

#### **Week 3: Contradiction UI Components**
```javascript
// Frontend components needed:
- Side-by-side witness comparison display
- Contradiction highlighting visualization
- Conflict confidence indicators
- "Show Contradictions" toggle in search
```

### **Phase 2: Quality Improvements (2 weeks)**
- Fix OCR artifacts in document processing
- Improve search result relevance
- Add more witness testimonies

### **Phase 3: Advanced Features (2-3 weeks)**  
- British Inquiry document integration
- Timeline contradictions
- Witness credibility analysis

---

## 💡 **THE KILLER FEATURE - DETAILED IMPLEMENTATION PLAN**

### **Contradiction Detection Algorithm**
```python
class ContradictionDetector:
    def find_conflicts(self, query: str) -> List[Contradiction]:
        """Find contradictory witness accounts for a query"""
        # 1. Get all relevant witnesses for query
        # 2. Extract key claims from each witness
        # 3. Compare claims for contradictions
        # 4. Score contradiction confidence
        # 5. Return conflicting pairs
        
    def compare_witnesses(self, witness1: str, witness2: str, topic: str):
        """Direct witness comparison on specific topic"""
        
    def score_contradiction(self, claim1: str, claim2: str) -> float:
        """Calculate how contradictory two claims are"""
```

### **Example User Experience**
```
User Query: "How fast was the ship going?"

Standard RAG Response: 
"The ship was traveling at normal speed"

Our Contradiction-Aware Response:
┌─ Ismay: "We were never at full speed" ────────┐
│                                               │  
│  ⚠️  CONTRADICTION DETECTED (85% confidence) │
│                                               │
├─ Lightoller: "Ship was at nearly full speed" ┤
│                                               │
└─ Fleet: "I don't know the exact speed" ───────┘

[Show Details] [Compare All Witnesses] [View Sources]
```

---

## 📊 **SUCCESS METRICS - REDEFINED**

### **Current Metrics (Foundation)**
```
✅ 30 witnesses indexed
✅ 381 chunks in vector database  
✅ Web application operational
✅ Basic search functionality working
```

### **Target Metrics (With Killer Feature)**
```
🎯 Contradiction detection for 10+ common topics
🎯 95% accuracy in identifying conflicting accounts
🎯 Sub-second response time for contradiction queries
🎯 Visual contradiction display working
🎯 User can easily compare witness accounts side-by-side
```

---

## 🚀 **PROJECT ROADMAP - CONTRADICTION-FOCUSED**

### **September 2025: Foundation Complete**
- ✅ Basic RAG pipeline operational
- ✅ 30 witnesses indexed
- ✅ Web application running

### **October 2025: KILLER FEATURE IMPLEMENTATION** 
- 🔥 **Build contradiction detection engine**
- 🔥 **Implement visual conflict highlighting** 
- 🔥 **Add witness comparison functionality**
- 🔥 **Create contradiction confidence scoring**

### **November 2025: Quality & Scale**
- 🔧 Fix document processing issues
- 📈 Add more witnesses and documents
- 🎨 Polish contradiction visualization UI
- 🚀 Prepare for production deployment

---

## 💎 **WHY THIS MATTERS - THE UNIQUE VALUE PROPOSITION**

### **Every Other RAG System:**
- Tries to find "the single truth"
- Hides conflicting information
- Normalizes contradictory sources
- Generic document search

### **Our System:**
- **Embraces contradictions as features**
- **Highlights conflicts between witnesses**  
- **Shows multiple perspectives simultaneously**
- **Designed specifically for historical research**

**This is what makes us special - we need to build it!**

---

## 📞 **IMMEDIATE ACTION ITEMS**

### **This Week:**
1. **Prioritize contradiction detection implementation**
2. Design contradiction visualization UI mockups
3. Plan witness comparison algorithm architecture
4. Start building contradiction detection engine

### **This Month:**
1. **Complete killer feature implementation**
2. Fix document processing quality issues
3. Add more witness testimonies
4. Test contradiction accuracy

**The foundation is solid. Now we need to build what makes us unique!**
Ok
---

*Last Updated: September 2025*  
*Status: 🔥 Ready to Build Killer Feature - Contradiction Detection*  
*Priority: Contradiction Highlighting System - This Is What Makes Us Special!*