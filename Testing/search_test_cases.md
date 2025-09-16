# Search System Test Cases - Date2.pdf Content
*Generated: September 13, 2025*

## 🎯 **Purpose**
Test cases based on specific content from Date2.pdf (pages 207-212) to evaluate and improve our RAG search system's ability to find precise factual information.

## 📋 **Test Cases**

### **1. Ocean Navigation Details**
**Question**: "How many miles apart are the eastward and westward shipping tracks?"
**Expected Answer**: "60 miles" (from Franklin's testimony)
**Current Performance**: POOR - finds track information but wrong context

### **2. Officer Personal Details**
**Question**: "What is Joseph Groves Boxhall's age and years of experience?"
**Expected Answer**: "Twenty-eight years old, thirteen years experience at sea"
**Current Performance**: EXCELLENT - finds exact match

### **3. Corporate Structure**
**Question**: "What was the relationship between Ismay & Imrie and the White Star Line?"
**Expected Answer**: "Managing firm of the White Star Line, now just a trade name, empty shell"
**Current Performance**: POOR - finds Ismay mentions but not corporate details

### **4. Officer Command Changes**
**Question**: "Who replaced Lightoller as first officer and when?"
**Expected Answer**: "Murdoch replaced Lightoller as first officer the night before sailing"
**Current Performance**: EXCELLENT - perfect match with 1.0 relevance

### **5. Ship Safety Inspections**
**Question**: "Who was present during the lifeboat inspection on the morning of sailing?"
**Expected Answer**: "Captain, all officers, marine superintendent, Board of Trade surveyors, Board of Trade doctor"
**Current Performance**: EXCELLENT - finds complete details

### **6. Weather Conditions**
**Question**: "What weather conditions did Boxhall encounter traveling from Belfast to Southampton?"
**Expected Answer**: "Fine until 2 o'clock morning, foggy at 4 o'clock, cleared at 6 o'clock, smooth seas"
**Current Performance**: UNKNOWN - needs testing

### **7. Ship Speed Details**
**Question**: "What speed was the Titanic traveling at during the collision according to Franklin?"
**Expected Answer**: "About 21 knots" (Franklin had heard but no direct information)
**Current Performance**: UNKNOWN - needs testing

### **8. Navigation Training**
**Question**: "What navigation training did Boxhall receive before joining the merchant marine?"
**Expected Answer**: "12 months training in navigation school in Hull, England - navigation and nautical astronomy"
**Current Performance**: UNKNOWN - needs testing

### **9. Ship Testing Details**
**Question**: "How long did the Titanic's sea trials last from Belfast to Southampton?"
**Expected Answer**: "Left Belfast Tuesday noon, steamed until 7-8 PM, finally left about 8 PM, reached Southampton Thursday midnight"
**Current Performance**: UNKNOWN - needs testing

### **10. Company Communication**
**Question**: "Did Franklin send cable messages asking for information about ship positions during the disaster?"
**Expected Answer**: "No, did not send any cable message asking for information about position of any ship or anybody"
**Current Performance**: UNKNOWN - needs testing

## 🔍 **Performance Analysis Needed**

### **Strong Areas:**
- Witness testimony and personal details
- Officer roles and responsibilities  
- Procedural descriptions (inspections, drills)

### **Weak Areas:**
- Specific numerical facts (distances, measurements)
- Corporate/business structure details
- Technical specifications and procedures

## 🛠️ **Recommended Improvements**

1. **Enhance fact extraction**: Improve detection of specific numbers, measurements, and technical details
2. **Expand context windows**: Include more surrounding context for factual information
3. **Improve metadata matching**: Better recognition of corporate names, technical terms
4. **Add semantic similarity**: Better matching of conceptually related terms (e.g., "managing firm" = "agents and managers")
5. **Implement fact verification**: Cross-reference factual claims across multiple chunks

## 📊 **Success Metrics**
- **Current**: 3/5 test cases excellent performance (60%)
- **Target**: 8/10 test cases excellent performance (80%)
- **Focus**: Improve factual detail retrieval while maintaining narrative search quality

---
*These test cases will help identify specific areas where our search system needs improvement to handle both narrative testimony and precise factual queries effectively.*