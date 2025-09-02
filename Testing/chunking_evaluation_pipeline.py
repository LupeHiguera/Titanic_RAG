"""
Chunking Evaluation Pipeline for Titanic RAG System

Creates a comprehensive testing framework to evaluate different chunking strategies
using real Titanic testimony data, with focus on:

1. Biographical information preservation (e.g., "Ismay age")
2. Factual query accuracy (e.g., "ship speed", "lifeboat details") 
3. Contradiction detection capabilities
4. Source citation accuracy

This pipeline helps optimize chunking before implementing the full semantic search system.
"""

import pytest
from pathlib import Path
import sys
import json
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import re

# Add the root directory to path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from Services.chunking import IntelligentChunker, WitnessChunk
from Services.document_ingestion import DocumentIngestion


class QueryType(Enum):
    BIOGRAPHICAL = "biographical"
    FACTUAL = "factual" 
    PROCEDURAL = "procedural"
    CONTRADICTION = "contradiction"
    TEMPORAL = "temporal"


@dataclass
class GoldenQuery:
    """A test query with expected information that should be retrievable"""
    query: str
    type: QueryType
    expected_info: List[str]  # Key facts that should be found
    should_not_contain: List[str] = None  # Info that should NOT be returned
    required_witnesses: List[str] = None  # Witnesses that must be represented
    expected_contradictions: bool = False
    description: str = ""


@dataclass 
class ChunkEvaluation:
    """Evaluation results for a single chunk against a query"""
    chunk: WitnessChunk
    relevance_score: float  # 0-2 scale
    contains_expected: List[str]  # Which expected info was found
    completeness_score: float  # How complete is the information
    citation_quality: float  # How traceable is this back to source


@dataclass
class QueryEvaluation:
    """Complete evaluation results for a query across all chunks"""
    query: GoldenQuery
    chunk_evaluations: List[ChunkEvaluation]
    overall_score: float
    missing_information: List[str]
    false_positives: List[str]
    contradiction_detected: bool
    issues: List[str]


class ChunkingEvaluationPipeline:
    """Main evaluation pipeline for testing different chunking strategies"""
    
    def __init__(self):
        self.golden_queries = self._create_golden_queries()
        self.ingestion = DocumentIngestion()
        self.ismay_testimony = self._load_ismay_testimony()
    
    def _load_ismay_testimony(self) -> Dict[str, Any]:
        """Load the actual Ismay testimony data for testing"""
        data_pdf_path = root_dir / "Text" / "Data.pdf"
        if not data_pdf_path.exists():
            # Fallback to one page.pdf
            pdf_path = root_dir / "Text" / "one page.pdf"
            if pdf_path.exists():
                result = self.ingestion.extract_text_from_pdf(pdf_path)
                witnesses = self.ingestion.identify_witness_names(result["text"])
                return {
                    'text': result["text"],
                    'witnesses': witnesses,
                    'document_name': result["metadata"].document_name
                }
        else:
            result = self.ingestion.extract_text_from_pdf(data_pdf_path)
            witnesses = self.ingestion.identify_witness_names(result["text"])
            return {
                'text': result["text"], 
                'witnesses': witnesses,
                'document_name': result["metadata"].document_name
            }
        
        return {'text': '', 'witnesses': [], 'document_name': 'Unknown'}
    
    def _create_golden_queries(self) -> List[GoldenQuery]:
        """Create golden test queries based on actual Ismay testimony"""
        return [
            # BIOGRAPHICAL QUERIES - Test preservation of biographical info
            GoldenQuery(
                query="Ismay age",
                type=QueryType.BIOGRAPHICAL,
                expected_info=["50", "December 12th", "12th of December"],
                description="Should find Ismay's age and birthday information"
            ),
            
            GoldenQuery(
                query="Ismay position", 
                type=QueryType.BIOGRAPHICAL,
                expected_info=["Managing Director", "White Star Line", "Ship owner"],
                description="Should find Ismay's role and company"
            ),
            
            GoldenQuery(
                query="Ismay residence",
                type=QueryType.BIOGRAPHICAL, 
                expected_info=["Liverpool"],
                description="Should find Ismay's place of residence"
            ),
            
            # FACTUAL QUERIES - Test retrieval of specific facts
            GoldenQuery(
                query="ship speed revolutions",
                type=QueryType.FACTUAL,
                expected_info=["75 revolutions", "78 revolutions", "never exceeded 75"],
                should_not_contain=["full speed"],
                description="Should find revolution details but clarify ship never at full speed"
            ),
            
            GoldenQuery(
                query="boarding Southampton time",
                type=QueryType.FACTUAL,
                expected_info=["9.30 in the morning", "April 10th", "10th of April"],
                description="Should find specific boarding time and date"
            ),
            
            GoldenQuery(
                query="lifeboat capacity passengers",
                type=QueryType.FACTUAL,
                expected_info=["45", "practically", "full capacity"],
                description="Should find lifeboat occupancy in Ismay's boat"
            ),
            
            # PROCEDURAL QUERIES - Test understanding of procedures
            GoldenQuery(
                query="lifeboat loading procedure",
                type=QueryType.PROCEDURAL,
                expected_info=["women and children first", "natural order", "officers", "ship's people"],
                description="Should find lifeboat loading procedures and priorities"
            ),
            
            GoldenQuery(
                query="collision response actions",
                type=QueryType.PROCEDURAL,
                expected_info=["went to bridge", "found captain", "struck ice", "get boats out"],
                description="Should find sequence of actions after collision"
            ),
            
            # TEMPORAL QUERIES - Test time-based information
            GoldenQuery(
                query="collision time sinking time", 
                type=QueryType.TEMPORAL,
                expected_info=["Sunday night", "2:20", "sank"],
                description="Should find timing of collision and sinking"
            ),
            
            GoldenQuery(
                query="departure Southampton arrival Cherbourg",
                type=QueryType.TEMPORAL,
                expected_info=["12 o'clock", "evening", "68 revolutions"],
                description="Should find departure and arrival timing"
            ),
            
            # CONTRADICTION QUERIES - Test ability to preserve conflicting info
            GoldenQuery(
                query="ice warnings knowledge",
                type=QueryType.CONTRADICTION,
                expected_info=["ice had been reported", "did not know", "ice region"],
                expected_contradictions=True,
                description="Should capture both awareness and lack of specific knowledge about ice"
            ),
            
            GoldenQuery(
                query="captain consultation ship movement",
                type=QueryType.CONTRADICTION, 
                expected_info=["Never", "did not consult", "arranged", "5 o'clock Wednesday"],
                expected_contradictions=True,
                description="Should capture both no consultation but some pre-arrangement"
            ),
            
            # WITNESS IDENTIFICATION - Test witness name preservation
            GoldenQuery(
                query="Thomas Andrews builder representative",
                type=QueryType.BIOGRAPHICAL,
                expected_info=["Thomas Andrews", "representative of builders", "42 or 43 years", "Unfortunately, no"],
                required_witnesses=["Ismay"],
                description="Should find info about Andrews while preserving Ismay as source"
            ),
            
            # COMPLEX PROCEDURAL - Test complex multi-step procedures
            GoldenQuery(
                query="lifeboat departure circumstances last boat",
                type=QueryType.PROCEDURAL,
                expected_info=["no response", "no passengers left", "collapsible boat", "last boat", "being lowered away"],
                description="Should capture the specific circumstances of Ismay's departure"
            ),
            
            # NEGATION QUERIES - Test handling of negative statements
            GoldenQuery(
                query="wireless operator messages sent",
                type=QueryType.FACTUAL,
                expected_info=["did not", "I did not see", "no messages"],
                description="Should preserve negative statements about wireless interaction"
            )
        ]
    
    def evaluate_chunking_strategy(self, chunker: IntelligentChunker, strategy_name: str = "default") -> Dict[str, Any]:
        """Evaluate a chunking strategy against all golden queries"""
        
        # Create chunks from Ismay testimony
        witness_context = {
            'witness': self.ismay_testimony['witnesses'][0] if self.ismay_testimony['witnesses'] else 'Joseph Bruce Ismay',
            'testimony': self.ismay_testimony['text'],
            'page_number': 1,
            'document_name': self.ismay_testimony['document_name']
        }
        
        chunks = chunker.chunk_witness_contexts([witness_context])
        
        # Evaluate each query
        query_evaluations = []
        for query in self.golden_queries:
            evaluation = self._evaluate_query(query, chunks)
            query_evaluations.append(evaluation)
        
        # Calculate overall metrics
        overall_metrics = self._calculate_overall_metrics(query_evaluations)
        
        return {
            'strategy_name': strategy_name,
            'query_evaluations': query_evaluations,
            'overall_metrics': overall_metrics,
            'total_chunks': len(chunks),
            'avg_chunk_size': sum(len(c.content) for c in chunks) / len(chunks) if chunks else 0
        }
    
    def _evaluate_query(self, query: GoldenQuery, chunks: List[WitnessChunk]) -> QueryEvaluation:
        """Evaluate a single query against all chunks"""
        
        chunk_evaluations = []
        for chunk in chunks:
            evaluation = self._evaluate_chunk_for_query(query, chunk)
            chunk_evaluations.append(evaluation)
        
        # Sort by relevance score
        chunk_evaluations.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Calculate overall query results
        best_chunks = chunk_evaluations[:3]  # Top 3 chunks
        
        # Check what expected info was found across all chunks
        all_found_info = set()
        for eval in best_chunks:
            all_found_info.update(eval.contains_expected)
        
        missing_info = [info for info in query.expected_info if not self._info_found_in_chunks(info, best_chunks)]
        
        # Calculate overall score
        expected_found = len(all_found_info)
        total_expected = len(query.expected_info)
        overall_score = expected_found / total_expected if total_expected > 0 else 0.0
        
        # Check for contradictions if expected
        contradiction_detected = False
        if query.expected_contradictions:
            # Simple check: do we have both positive and negative statements
            all_content = " ".join([eval.chunk.content for eval in best_chunks]).lower()
            has_negation = any(neg in all_content for neg in ["not", "never", "no", "did not"])
            has_positive = any(pos in all_content for pos in query.expected_info)
            contradiction_detected = has_negation and has_positive
        
        issues = []
        if overall_score < 0.5:
            issues.append(f"Low coverage: only {expected_found}/{total_expected} expected items found")
        
        if query.should_not_contain:
            false_positives = []
            for eval in best_chunks:
                for should_not in query.should_not_contain:
                    if should_not.lower() in eval.chunk.content.lower():
                        false_positives.append(should_not)
            if false_positives:
                issues.append(f"Contains prohibited info: {false_positives}")
        else:
            false_positives = []
        
        return QueryEvaluation(
            query=query,
            chunk_evaluations=chunk_evaluations,
            overall_score=overall_score,
            missing_information=missing_info,
            false_positives=false_positives,
            contradiction_detected=contradiction_detected,
            issues=issues
        )
    
    def _evaluate_chunk_for_query(self, query: GoldenQuery, chunk: WitnessChunk) -> ChunkEvaluation:
        """Evaluate how well a single chunk answers a query"""
        
        content_lower = chunk.content.lower()
        query_lower = query.query.lower()
        
        # Calculate relevance score (0-2 scale)
        relevance_score = 0.0
        
        # Check if query terms appear in chunk
        query_words = set(query_lower.split())
        chunk_words = set(content_lower.split())
        query_overlap = len(query_words.intersection(chunk_words)) / len(query_words)
        relevance_score += query_overlap * 0.5
        
        # Check for expected information
        found_expected = []
        for expected in query.expected_info:
            if self._fuzzy_match(expected.lower(), content_lower):
                found_expected.append(expected)
                relevance_score += 0.5
        
        # Cap relevance at 2.0
        relevance_score = min(2.0, relevance_score)
        
        # Calculate completeness (how much of expected info is in this chunk)
        completeness = len(found_expected) / len(query.expected_info) if query.expected_info else 0.0
        
        # Citation quality - can we trace this back to source?
        citation_quality = 1.0 if chunk.metadata.page_number > 0 else 0.5
        if chunk.witness_name and len(chunk.witness_name) > 1:
            citation_quality += 0.5
        citation_quality = min(1.0, citation_quality)
        
        return ChunkEvaluation(
            chunk=chunk,
            relevance_score=relevance_score,
            contains_expected=found_expected,
            completeness_score=completeness,
            citation_quality=citation_quality
        )
    
    def _fuzzy_match(self, needle: str, haystack: str) -> bool:
        """Check if needle appears in haystack with some flexibility"""
        # Direct substring match
        if needle in haystack:
            return True
        
        # Word-based matching for numbers and key terms
        needle_words = needle.split()
        for word in needle_words:
            if len(word) > 2 and word in haystack:
                return True
        
        return False
    
    def _info_found_in_chunks(self, info: str, chunk_evaluations: List[ChunkEvaluation]) -> bool:
        """Check if specific info was found across chunk evaluations"""
        for eval in chunk_evaluations:
            if info in eval.contains_expected:
                return True
        return False
    
    def _calculate_overall_metrics(self, query_evaluations: List[QueryEvaluation]) -> Dict[str, float]:
        """Calculate overall performance metrics"""
        
        if not query_evaluations:
            return {}
        
        # Coverage: percentage of queries that found at least some expected info
        covered_queries = sum(1 for qe in query_evaluations if qe.overall_score > 0)
        coverage = covered_queries / len(query_evaluations)
        
        # Average query score
        avg_score = sum(qe.overall_score for qe in query_evaluations) / len(query_evaluations)
        
        # Precision: how often we avoid false positives
        queries_with_false_positives = sum(1 for qe in query_evaluations if qe.false_positives)
        precision = 1.0 - (queries_with_false_positives / len(query_evaluations))
        
        # Citation quality across all chunks
        all_chunks = []
        for qe in query_evaluations:
            all_chunks.extend(qe.chunk_evaluations[:3])  # Top 3 per query
        
        if all_chunks:
            avg_citation_quality = sum(ce.citation_quality for ce in all_chunks) / len(all_chunks)
        else:
            avg_citation_quality = 0.0
        
        # Issue rate
        queries_with_issues = sum(1 for qe in query_evaluations if qe.issues)
        issue_rate = queries_with_issues / len(query_evaluations)
        
        return {
            'coverage': coverage,
            'average_score': avg_score,
            'precision': precision,
            'citation_quality': avg_citation_quality,
            'issue_rate': issue_rate,
            'total_queries': len(query_evaluations)
        }
    
    def compare_chunking_strategies(self, strategies: List[Tuple[IntelligentChunker, str]]) -> Dict[str, Any]:
        """Compare multiple chunking strategies side by side"""
        
        results = []
        for chunker, strategy_name in strategies:
            result = self.evaluate_chunking_strategy(chunker, strategy_name)
            results.append(result)
        
        # Create comparison report
        comparison = {
            'strategies': results,
            'winner_by_metric': {},
            'detailed_query_comparison': {}
        }
        
        # Find winner by each metric
        metrics = ['coverage', 'average_score', 'precision', 'citation_quality']
        for metric in metrics:
            best_strategy = max(results, key=lambda x: x['overall_metrics'].get(metric, 0))
            comparison['winner_by_metric'][metric] = {
                'strategy': best_strategy['strategy_name'],
                'score': best_strategy['overall_metrics'].get(metric, 0)
            }
        
        # Detailed query-by-query comparison for top issues
        problem_queries = []
        for i, query in enumerate(self.golden_queries):
            query_results = []
            for result in results:
                if i < len(result['query_evaluations']):
                    query_eval = result['query_evaluations'][i]
                    query_results.append({
                        'strategy': result['strategy_name'],
                        'score': query_eval.overall_score,
                        'issues': query_eval.issues,
                        'missing': query_eval.missing_information
                    })
            
            # If any strategy struggles with this query
            if any(qr['score'] < 0.5 for qr in query_results):
                problem_queries.append({
                    'query': query.query,
                    'type': query.type.value,
                    'results': query_results
                })
        
        comparison['problem_queries'] = problem_queries
        
        return comparison
    
    def generate_report(self, evaluation_result: Dict[str, Any], output_file: Optional[str] = None) -> str:
        """Generate a human-readable evaluation report"""
        
        report_lines = []
        
        report_lines.append(f"# Chunking Strategy Evaluation Report")
        report_lines.append(f"Strategy: {evaluation_result['strategy_name']}")
        report_lines.append(f"Total Chunks: {evaluation_result['total_chunks']}")
        report_lines.append(f"Average Chunk Size: {evaluation_result['avg_chunk_size']:.1f} characters")
        report_lines.append("")
        
        # Overall metrics
        metrics = evaluation_result['overall_metrics']
        report_lines.append("## Overall Performance")
        report_lines.append(f"- **Coverage**: {metrics['coverage']:.1%} ({metrics['coverage']*metrics['total_queries']:.0f}/{metrics['total_queries']} queries)")
        report_lines.append(f"- **Average Score**: {metrics['average_score']:.2f}/1.0")
        report_lines.append(f"- **Precision**: {metrics['precision']:.1%} (avoiding false positives)")
        report_lines.append(f"- **Citation Quality**: {metrics['citation_quality']:.2f}/1.0")
        report_lines.append(f"- **Issue Rate**: {metrics['issue_rate']:.1%}")
        report_lines.append("")
        
        # Query-by-query results
        report_lines.append("## Query Performance")
        
        for qe in evaluation_result['query_evaluations']:
            score_emoji = "✅" if qe.overall_score >= 0.8 else "⚠️" if qe.overall_score >= 0.5 else "❌"
            report_lines.append(f"### {score_emoji} {qe.query.query} ({qe.query.type.value})")
            report_lines.append(f"**Score**: {qe.overall_score:.2f}/1.0")
            
            if qe.missing_information:
                report_lines.append(f"**Missing**: {', '.join(qe.missing_information)}")
            
            if qe.false_positives:
                report_lines.append(f"**False Positives**: {', '.join(qe.false_positives)}")
            
            if qe.issues:
                report_lines.append("**Issues**:")
                for issue in qe.issues:
                    report_lines.append(f"  - {issue}")
            
            # Show best chunk for this query
            if qe.chunk_evaluations:
                best_chunk = qe.chunk_evaluations[0]
                preview = best_chunk.chunk.content[:200] + "..." if len(best_chunk.chunk.content) > 200 else best_chunk.chunk.content
                report_lines.append(f"**Best Chunk** (score: {best_chunk.relevance_score:.1f}): {preview}")
            
            report_lines.append("")
        
        report = "\n".join(report_lines)
        
        if output_file:
            Path(output_file).write_text(report)
        
        return report


def create_chunking_strategies() -> List[Tuple[IntelligentChunker, str]]:
    """Create different chunking strategies to test"""
    
    strategies = [
        # Current default strategy
        (IntelligentChunker(chunk_size=500, overlap_size=50), "current_default"),
        
        # Smaller chunks for better precision
        (IntelligentChunker(chunk_size=300, overlap_size=30), "small_precise"),
        
        # Larger chunks for better context
        (IntelligentChunker(chunk_size=800, overlap_size=80), "large_context"),
        
        # High overlap for continuity
        (IntelligentChunker(chunk_size=500, overlap_size=100), "high_overlap"),
        
        # Minimal overlap for distinct chunks
        (IntelligentChunker(chunk_size=500, overlap_size=20), "minimal_overlap"),
    ]
    
    return strategies


if __name__ == "__main__":
    # Run the evaluation pipeline
    pipeline = ChunkingEvaluationPipeline()
    strategies = create_chunking_strategies()
    
    print("Running chunking strategy comparison...")
    comparison = pipeline.compare_chunking_strategies(strategies)
    
    print("\n=== CHUNKING STRATEGY COMPARISON ===")
    
    # Print winner by metric
    print("\n## Winners by Metric:")
    for metric, winner in comparison['winner_by_metric'].items():
        print(f"- **{metric}**: {winner['strategy']} ({winner['score']:.3f})")
    
    # Show problematic queries
    if comparison['problem_queries']:
        print(f"\n## Queries needing improvement ({len(comparison['problem_queries'])}):")
        for pq in comparison['problem_queries'][:3]:  # Top 3 problems
            print(f"- **{pq['query']}** ({pq['type']})")
            for result in pq['results']:
                print(f"  - {result['strategy']}: {result['score']:.2f} - {', '.join(result['issues']) if result['issues'] else 'OK'}")
    
    # Generate detailed report for best strategy
    best_overall = max(comparison['strategies'], key=lambda x: x['overall_metrics']['average_score'])
    print(f"\n## Detailed Report for Best Strategy: {best_overall['strategy_name']}")
    
    report = pipeline.generate_report(best_overall)
    
    # Save to file
    report_file = root_dir / "chunking_evaluation_report.md"
    Path(report_file).write_text(report)
    print(f"\nDetailed report saved to: {report_file}")