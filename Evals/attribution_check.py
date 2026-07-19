"""Offline attribution coverage check — no API keys required.

Runs the real ingestion attribution path (page map → witness index) over both
inquiry PDFs and reports which witnesses receive testimony, how many pages
and characters each gets, and whether any indexed witness ends up with
nothing. Run before/after attribution changes to see exactly who moved.

    python Evals/attribution_check.py [--verbose]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from Services.british_witness_index import BritishWitnessIndex
from Services.document_ingestion import DocumentIngestion
from Services.page_map import build_page_map
from Services.pinecone_upload import build_witness_contexts
from Services.witness_index import WitnessIndex

# Witnesses that page-level attribution used to swallow entirely (same-page
# tie-breaking on either side) — the regressions this script exists to guard.
RECOVERED_US = ["James Widgery", "Frederick M. Sammis"]
RECOVERED_BRITISH = [
    "Cyril Evans", "Robert Pusey", "William Chantler", "Eben Sharpe",
    "Frederick Sheath", "Alstander Boyle", "Ernest Archer",
    "Hugh Young", "John Fairfull",
]


def check(pdf_path: str, index, label: str, must_have: list, verbose: bool) -> bool:
    print(f"\n=== {label}: {pdf_path} ===")
    ingestion = DocumentIngestion()
    page_texts = ingestion.extract_pages_from_pdf(Path(pdf_path))
    page_map = build_page_map(pdf_path)

    # Run the REAL ingestion session builder (heading-based sub-page splits,
    # page tags, cleaning) — not a simplified per-page approximation.
    contexts = build_witness_contexts(page_texts, index, page_map, label,
                                      doc_ingestion=ingestion)

    coverage: dict = {}  # name -> {sessions: int, start_pages: [], chars: int}
    for ctx in contexts:
        slot = coverage.setdefault(ctx['witness'], {"sessions": 0, "pages": [], "chars": 0})
        slot["sessions"] += 1
        slot["pages"].append(ctx['page_number'])
        slot["chars"] += len(ctx['testimony'])

    indexed = {w.name for w in index.get_unique_witnesses()}
    covered = set(coverage)
    missing = sorted(indexed - covered)

    print(f"pages: {len(page_texts)} pdf → {len(contexts)} witness sessions")
    print(f"witnesses: {len(covered)}/{len(indexed)} indexed witnesses received testimony")

    ok = True
    for name in must_have:
        if name in coverage:
            c = coverage[name]
            print(f"  ✅ recovered: {name}  (sessions @ pp. "
                  f"{sorted(c['pages'])}, {c['chars']} chars)")
        else:
            print(f"  ❌ STILL MISSING: {name}")
            ok = False

    if missing:
        print(f"  ⚠ indexed but no testimony attributed: {', '.join(missing)}")
    if verbose:
        for name, c in sorted(coverage.items(), key=lambda kv: -kv[1]["chars"]):
            print(f"    {name}: {c['sessions']} session(s) @ pp.{sorted(c['pages'])}, {c['chars']} chars")

    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    ok = check("Text/USInq.pdf", WitnessIndex(), "US Senate Inquiry",
               RECOVERED_US, args.verbose)
    ok &= check("Text/BritishInquiry.pdf", BritishWitnessIndex(),
                "British Wreck Commissioner's Inquiry", RECOVERED_BRITISH, args.verbose)

    print("\n" + ("✅ attribution check PASSED" if ok else "❌ attribution check FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
