"""Attribution correctness: same-page tie-breaks, testimony bounds, and
page-map marker filtering."""

from Services.witness_index import WitnessIndex
from Services.british_witness_index import BritishWitnessIndex, BRITISH_TO_US_CANONICAL
from Services.page_map import _plausible


class TestUsTieBreaks:
    def setup_method(self):
        self.index = WitnessIndex()

    def test_widgery_not_swallowed_by_hardy(self):
        # Hardy (recalled) and Widgery both start on printed p.601; Widgery's
        # testimony continues to p.602. The later TOC entry must win the tie.
        assert self.index.get_witness_by_page_range(601).name == "James Widgery"
        assert self.index.get_witness_by_page_range(602).name == "James Widgery"

    def test_sammis_not_swallowed_by_marconi(self):
        # Marconi (recalled) and Sammis both start on p.845.
        assert self.index.get_witness_by_page_range(845).name == "Frederick M. Sammis"
        assert self.index.get_witness_by_page_range(850).name == "Frederick M. Sammis"

    def test_normal_range_lookup_still_works(self):
        assert self.index.get_witness_by_page_range(46).name == "Charles Herbert Lightoller"
        assert self.index.get_witness_by_page_range(94).name == "Charles Herbert Lightoller"
        assert self.index.get_witness_by_page_range(714).name == "Stanley Lord"

    def test_bounds_exclude_front_matter_and_appendices(self):
        assert self.index.get_witness_by_page_range(1) is None
        assert self.index.get_witness_by_page_range(1143) is None  # affidavits
        assert self.index.get_witness_by_page_range(1160) is None  # digest
        # Barrett still owns the last real testimony pages
        assert self.index.get_witness_by_page_range(1141).name == "Frederick Barrett"


class TestBritishTieBreaks:
    def setup_method(self):
        self.index = BritishWitnessIndex()

    def test_cyril_evans_not_swallowed_by_stone(self):
        # Charles V. Stone (recalled) and Cyril Evans both start on
        # transcript p.201; pages up to Moore (p.207) are Evans's.
        assert self.index.get_witness_by_page_range(201).name == "Cyril Evans"
        assert self.index.get_witness_by_page_range(205).name == "Cyril Evans"

    def test_pusey_chantler_sharpe_recovered(self):
        assert self.index.get_witness_by_page_range(294).name == "Robert Pusey"
        assert self.index.get_witness_by_page_range(676).name == "William Chantler"
        assert self.index.get_witness_by_page_range(698).name == "Eben Sharpe"

    def test_bounds_unchanged(self):
        assert self.index.get_witness_by_page_range(16) is None
        assert self.index.get_witness_by_page_range(749) is None


class TestCanonicalMap:
    def test_new_cross_inquiry_aliases_present(self):
        for british, us in [
            ("Robert Hitchins", "Robert Hichens"),
            ("George Rowe", "George Thomas Rowe"),
            ("Edward Buley", "Edward John Buley"),
            ("Samuel Hemmings", "Samuel S. Hemming"),
            ("James Moore", "James Henry Moore"),
            ("Cyril Evans", "Cyril Furmstone Evans"),
        ]:
            assert BRITISH_TO_US_CANONICAL[british] == us


class TestPageMapPlausibility:
    def test_printed_page_never_ahead_of_pdf_page(self):
        assert not _plausible(603, 3, None, None)  # TOC reference line
        assert _plausible(2, 6, None, None)

    def test_monotonic_non_decreasing(self):
        assert not _plausible(100, 500, 150, 490)
        assert _plausible(150, 500, 150, 499)  # same page repeats fine

    def test_forward_jump_bounded_by_pdf_gap(self):
        # 11-page printed jump over a 1-page pdf gap = noise marker
        assert not _plausible(146, 325, 135, 324)
        # gradual advance is fine
        assert _plausible(137, 329, 135, 324)
