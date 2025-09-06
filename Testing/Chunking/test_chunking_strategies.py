#!/usr/bin/env python3
"""
Quick script to test and compare different chunking strategies for Titanic RAG

Usage:
  python test_chunking_strategies.py                    # Run all strategies
  python test_chunking_strategies.py --strategy large   # Test just one strategy
  python test_chunking_strategies.py --detailed         # Generate detailed reports
"""

import argparse
from pathlib import Path
import sys

# Add the root directory to path
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

from Testing.chunking_evaluation_pipeline import ChunkingEvaluationPipeline, create_chunking_strategies
from Services.chunking import IntelligentChunker


def create_extended_strategies():
    """Create an extended set of chunking strategies based on initial findings"""
    
    strategies = [
        # Original strategies
        (IntelligentChunker(chunk_size=500, overlap_size=50), "current_default"),
        (IntelligentChunker(chunk_size=300, overlap_size=30), "small_precise"), 
        (IntelligentChunker(chunk_size=800, overlap_size=80), "large_context"),
        (IntelligentChunker(chunk_size=500, overlap_size=100), "high_overlap"),
        
        # New optimized strategies based on results
        (IntelligentChunker(chunk_size=700, overlap_size=70), "medium_large"),
        (IntelligentChunker(chunk_size=600, overlap_size=60), "balanced_optimal"),
        
        # Specific optimizations for biographical info preservation
        (IntelligentChunker(chunk_size=400, overlap_size=80), "biographical_focused"),
        
        # Q&A preservation focused
        (IntelligentChunker(chunk_size=450, overlap_size=45), "qa_optimized"),
    ]
    
    return strategies


def main():
    parser = argparse.ArgumentParser(description="Test chunking strategies for Titanic RAG")
    parser.add_argument("--strategy", choices=["current", "small", "large", "high_overlap", "all"],
                      default="all", help="Which strategy to test")
    parser.add_argument("--detailed", action="store_true", help="Generate detailed reports")
    parser.add_argument("--output-dir", default="reports", help="Output directory for reports")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    pipeline = ChunkingEvaluationPipeline()
    
    if args.strategy == "all":
        print("🔍 Testing all chunking strategies...")
        strategies = create_extended_strategies()
        
        # Run comparison
        comparison = pipeline.compare_chunking_strategies(strategies)
        
        print("\n" + "="*60)
        print("📊 CHUNKING STRATEGY COMPARISON RESULTS")
        print("="*60)
        
        # Print summary table
        print(f"\n{'Strategy':<20} {'Coverage':<10} {'Avg Score':<10} {'Precision':<10} {'Issues':<8}")
        print("-" * 70)
        
        for result in comparison['strategies']:
            metrics = result['overall_metrics']
            print(f"{result['strategy_name']:<20} "
                  f"{metrics['coverage']:.1%}      "
                  f"{metrics['average_score']:.3f}     "
                  f"{metrics['precision']:.1%}      "
                  f"{metrics['issue_rate']:.0%}")
        
        # Show winners
        print(f"\n🏆 WINNERS BY METRIC:")
        for metric, winner in comparison['winner_by_metric'].items():
            print(f"  • {metric.title()}: {winner['strategy']} ({winner['score']:.3f})")
        
        # Show problem areas
        if comparison['problem_queries']:
            print(f"\n⚠️  QUERIES NEEDING IMPROVEMENT ({len(comparison['problem_queries'])}):")
            for i, pq in enumerate(comparison['problem_queries'][:5]):
                print(f"  {i+1}. {pq['query']} ({pq['type']})")
                worst = min(pq['results'], key=lambda x: x['score'])
                best = max(pq['results'], key=lambda x: x['score'])
                print(f"     Worst: {worst['strategy']} ({worst['score']:.2f}) | Best: {best['strategy']} ({best['score']:.2f})")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        best_overall = max(comparison['strategies'], key=lambda x: x['overall_metrics']['average_score'])
        best_balanced = max(comparison['strategies'], 
                          key=lambda x: x['overall_metrics']['average_score'] * x['overall_metrics']['precision'])
        
        print(f"  • Best Overall: {best_overall['strategy_name']} (avg score: {best_overall['overall_metrics']['average_score']:.3f})")
        print(f"  • Best Balanced: {best_balanced['strategy_name']} (score×precision: {best_balanced['overall_metrics']['average_score'] * best_balanced['overall_metrics']['precision']:.3f})")
        
        if args.detailed:
            print(f"\n📝 Generating detailed reports...")
            for result in comparison['strategies']:
                report = pipeline.generate_report(result)
                report_file = output_dir / f"chunking_report_{result['strategy_name']}.md"
                report_file.write_text(report)
                print(f"  • {result['strategy_name']}: {report_file}")
    
    else:
        # Test single strategy
        strategy_map = {
            "current": (IntelligentChunker(chunk_size=500, overlap_size=50), "current_default"),
            "small": (IntelligentChunker(chunk_size=300, overlap_size=30), "small_precise"),
            "large": (IntelligentChunker(chunk_size=800, overlap_size=80), "large_context"),
            "high_overlap": (IntelligentChunker(chunk_size=500, overlap_size=100), "high_overlap"),
        }
        
        if args.strategy in strategy_map:
            chunker, name = strategy_map[args.strategy]
            print(f"🔍 Testing {name} strategy...")
            
            result = pipeline.evaluate_chunking_strategy(chunker, name)
            
            # Print quick summary
            metrics = result['overall_metrics']
            print(f"\n📊 Results for {name}:")
            print(f"  Coverage: {metrics['coverage']:.1%}")
            print(f"  Average Score: {metrics['average_score']:.3f}")
            print(f"  Precision: {metrics['precision']:.1%}")
            print(f"  Citation Quality: {metrics['citation_quality']:.3f}")
            print(f"  Issue Rate: {metrics['issue_rate']:.1%}")
            
            if args.detailed:
                report = pipeline.generate_report(result)
                report_file = output_dir / f"chunking_report_{name}.md"
                report_file.write_text(report)
                print(f"\nDetailed report: {report_file}")


if __name__ == "__main__":
    main()