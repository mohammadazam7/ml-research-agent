import numpy as np
import pytest

from ml_research_agent.baselines import BaselineError, mae, roc_auc
from ml_research_agent.report import calibration, format_report, slices


def test_perfectly_calibrated_scores_have_no_gap() -> None:
    # Within each bin, the share of positives equals the score.
    score = np.array([0.25] * 4 + [0.75] * 4)
    positive = np.array([True, False, False, False, True, True, True, False])
    cal = calibration(positive, score, bins=2)
    assert cal.expected_calibration_error == pytest.approx(0.0)
    assert [(b.rows, b.positive_rate) for b in cal.bins] == [(4, 0.25), (4, 0.75)]


def test_overconfident_scores_show_their_gap() -> None:
    score = np.array([0.9] * 10)
    positive = np.array([True] * 6 + [False] * 4)
    cal = calibration(positive, score, bins=10)
    assert cal.expected_calibration_error == pytest.approx(0.3)
    assert cal.brier == pytest.approx((6 * 0.01 + 4 * 0.81) / 10)
    [only] = cal.bins
    assert (only.low, only.high) == pytest.approx((0.9, 1.0))


def test_a_score_of_one_lands_in_the_last_bin() -> None:
    cal = calibration(np.array([True]), np.array([1.0]), bins=4)
    assert cal.bins[0].low == 0.75


@pytest.mark.parametrize("score", [np.array([]), np.array([1.2]), np.array([-0.1])])
def test_calibration_refuses_scores_it_cannot_read(score: np.ndarray) -> None:
    with pytest.raises(BaselineError):
        calibration(np.zeros(len(score), dtype=bool), score)


def _data() -> tuple[np.ndarray, np.ndarray, list[object]]:
    rng = np.random.default_rng(0)
    n = 600
    region = [("north", "south", "east")[i % 3] for i in range(n)]
    positive = rng.random(n) < 0.4
    # Good scores everywhere except "east", where the score is noise.
    score = np.where(positive, rng.uniform(0.4, 1.0, n), rng.uniform(0.0, 0.6, n))
    east = np.array([r == "east" for r in region])
    score[east] = rng.random(east.sum())
    return positive, score, region


def test_a_slice_where_the_model_fails_is_flagged_first() -> None:
    positive, score, region = _data()
    overall, found = slices({"region": region}, positive, score, roc_auc)
    assert found[0].value == "east"
    assert found[0].flagged
    assert found[0].score is not None and found[0].score < overall - 0.05
    assert [s.flagged for s in found[1:]] == [False, False]


def test_small_slices_are_listed_but_never_flagged() -> None:
    positive, score, region = _data()
    region = list(region)
    for i in range(10):
        region[i * 3 + 2] = "tiny"  # ten "east" rows renamed
    _, found = slices({"region": region}, positive, score, roc_auc)
    tiny = next(s for s in found if s.value == "tiny")
    assert tiny.rows == 10
    assert not tiny.flagged


def test_a_slice_with_one_class_has_no_score() -> None:
    positive = np.array([True, False, True, True])
    score = np.array([0.9, 0.1, 0.8, 0.7])
    _, found = slices({"g": ["a", "a", "b", "b"]}, positive, score, roc_auc)
    b = next(s for s in found if s.value == "b")
    assert b.score is None
    assert not b.flagged
    assert found[-1] is b


def test_lower_is_better_metrics_flag_higher_errors() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0] * 20)
    predicted = y.copy()
    groups = ["ok", "bad"] * 40
    predicted[1::2] += 1.0
    _, found = slices({"g": groups}, y, predicted, mae, higher_is_better=False)
    assert (found[0].value, found[0].flagged) == ("bad", True)


def test_missing_values_form_their_own_slice() -> None:
    positive, score, _ = _data()
    column: list[object] = [None if i % 2 else "x" for i in range(len(score))]
    _, found = slices({"c": column}, positive, score, roc_auc)
    assert {s.value for s in found} == {"x", "(missing)"}


def test_columns_must_match_the_rows() -> None:
    with pytest.raises(BaselineError, match="2 values for 3 rows"):
        slices({"c": ["a", "b"]}, np.array([True, False, True]), np.zeros(3), roc_auc)


def test_the_report_lists_bins_and_marks_worse_slices() -> None:
    positive, score, region = _data()
    cal = calibration(positive, score)
    overall, found = slices({"region": region}, positive, score, roc_auc)
    text = format_report(cal, overall, found, "roc_auc")
    assert text.startswith("Calibration: ECE ")
    assert "region=east" in text.splitlines()[-3]
    assert text.count("worse") == 1
