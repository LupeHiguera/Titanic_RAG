# Titanic Historical RAG - 6-Week MVP

## Project Vision
Build a RAG system for exploring 2,000+ pages of Titanic historical documents (US/British inquiries) that **uniquely surfaces contradictions** between witness testimonies instead of hiding them.

**One Line:** "Google for Titanic primary sources, but it shows you contradictions instead of hiding them"

## Core Features (ONLY THESE 3)

### Feature 1: Core RAG Pipeline (Weeks 1-2)
**What:** Basic search that actually works

**Implementation Tasks:**
- [ ] Ingest 2,000 pages of inquiry documents (PDF → text)
- [ ] Chunk intelligently (preserve witness name + testimony context)
- [ ] Embed with OpenAI/Cohere embeddings
- [ ] Store in vector DB (Pinecone or Chroma for local dev)
- [ ] Build semantic search returning relevant passages
- [ ] Add basic LLM summarization with source references

**Success Test:** Query "Did the band play?" → Get actual witness testimony with page numbers

**Claude Code Sessions:**
```bash
# Week 1
claude-code "Build document ingestion pipeline for Titanic inquiry PDFs - extract text while preserving witness names and page numbers"

# Week 2  
claude-code "Implement RAG search: chunk documents keeping witness context, embed with OpenAI, store in Chroma, return relevant passages with sources"
```

### Feature 2: Contradiction Detection & Comparison (Weeks 3-4)
**What:** Surface conflicting accounts side-by-side

**Implementation Tasks:**
- [ ] Detect when witnesses disagree on same topic
- [ ] Extract conflicting claims into structured format
- [ ] Build side-by-side comparison UI component
- [ ] Flag confidence levels (crew vs passenger credibility)
- [ ] Group similar contradictions together

**Success Test:** Query "Lifeboat loading procedure" → See "Officer Lightoller says women and children only" vs "Passenger says men were allowed"

**Claude Code Sessions:**
```bash
# Week 3
claude-code "Build contradiction detector: analyze witness statements on same topic, identify conflicts, score by witness credibility (officer > crew > passenger)"

# Week 4
claude-code "Create comparison UI: side-by-side conflicting testimonies with context about each witness's position and vantage point"
```

### Feature 3: Citations & Production Deploy (Weeks 5-6)
**What:** Every claim is clickable + anyone can use it

**Implementation Tasks:**
- [ ] Link every AI statement to exact source passage
- [ ] Show original document page/line numbers
- [ ] Build clean web UI (FastAPI + simple frontend)
- [ ] Deploy to Vercel/Railway (free tier)
- [ ] Add basic error handling and rate limiting

**Success Test:** Send link to non-technical friend → They can research Titanic facts without help

**Claude Code Sessions:**
```bash
# Week 5
claude-code "Add citation system: every LLM response links to exact source passage with page numbers, build FastAPI endpoints"

# Week 6
claude-code "Create production-ready web UI and deploy to Railway with error handling and rate limits"
```

## Technical Stack

**Backend:** FastAPI  
**Vector DB:** Pinecone (production) or Chroma (local dev)  
**LLM:** GPT-4o-mini for summaries, OpenAI embeddings for search  
**Frontend:** Simple HTML/JS (Claude-generated)  
**Deploy:** Railway or Vercel (free tiers)

## Sample Data Format
```
Document: "British Wreck Commissioner's Inquiry - Day 5"
Witness: "Charles Lightoller, Second Officer"
Page: 247
Testimony: "The order was women and children first, and I interpreted that as women and children only..."

Document: "US Senate Inquiry - Day 12"  
Witness: "Hugh Woolner, First Class Passenger"
Page: 891
Testimony: "I saw men getting into lifeboats when there were no more women nearby..."
```

## What We're NOT Building (Yet)
- Knowledge graphs or timeline visualization
- Multi-language sources or OCR pipeline  
- Analytics dashboard or user accounts
- Advanced ML models or custom embeddings
- **Advanced witness credibility scoring** (currently using basic placeholder values)

## Definition of Done
A deployed website where users can:
1. Ask questions about the Titanic disaster
2. Get accurate answers from primary source testimonies
3. See when witnesses disagree on the same events
4. Click any claim to verify the original source
5. Share links with others for collaborative research

## Key Prompting Guidelines for Claude Code

**Domain Context:**
- Emphasize historical accuracy over smooth narratives
- Preserve contradictions as features, not bugs
- Account for witness bias (crew defending actions vs. passenger observations)
- British inquiry was more formal, American more aggressive

**Technical Priorities:**
- Chunking must preserve speaker identity with testimony
- Search should surface multiple perspectives, not single "truth"
- Citations are non-negotiable - every claim needs source
- UI should make contradictions obvious, not hidden

## Success Metrics
- **Week 2:** Can find and cite specific testimony passages
- **Week 4:** Shows conflicting accounts for known controversial topics
- **Week 6:** Non-technical users can independently research Titanic questions

## Development Approach
Start local (Chroma + free embeddings) → Scale to production (Pinecone + OpenAI) as needed. Deploy early and iterate based on actual historical research needs.