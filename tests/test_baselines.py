import numpy as np
import pyarrow as pa
import pytest

from ml_research_agent import baselines
from ml_research_agent.baselines import (
    BaselineError,
    average_precision,
    macro_f1,
    roc_auc,
)
from ml_research_agent.plan import plan, split
from ml_research_agent.profile import profile

N = 600


def _run(table: pa.Table, target: str) -> dict[str, baselines.Score]:
    p = profile(table, target)
    chosen = plan(p)
    return {s.baseline: s for s in baselines.run(table, p, chosen, split(table, p, chosen))}


def _features() -> dict[str, list[object]]:
    rng = np.random.default_rng(0)
    return {
        "amount": rng.normal(size=N).round(4).tolist(),
        "noise": rng.normal(size=N).round(4).tolist(),
        "region": [("north", "south", "east")[i % 3] for i in range(N)],
    }


def test_roc_auc_counts_ties_as_half() -> None:
    positive = np.array([True, False, True, False])
    assert roc_auc(positive, np.array([0.9, 0.1, 0.8, 0.2])) == 1.0
    assert roc_auc(positive, np.array([0.5, 0.5, 0.5, 0.5])) == 0.5
    # One positive below one negative: 3 of 4 pairs ordered correctly.
    assert roc_auc(positive, np.array([0.9, 0.7, 0.6, 0.1])) == 0.75


def test_average_precision_treats_tied_scores_as_one_block() -> None:
    positive = np.array([True, False, False, True])
    assert average_precision(positive, np.array([0.9, 0.2, 0.1, 0.8])) == 1.0
    # Every row tied: precision is the positive rate.
    assert average_precision(positive, np.full(4, 0.3)) == 0.5
    # Ranks 1 and 3: (1/1 + 2/3) / 2.
    assert average_precision(positive, np.array([0.9, 0.5, 0.1, 0.4])) == pytest.approx(5 / 6)


def test_macro_f1_weighs_each_class_equally() -> None:
    y = np.array(["a", "a", "a", "b"], dtype=object)
    all_a = np.array(["a"] * 4, dtype=object)
    # Class a: F1 = 6/7; class b: never predicted, F1 = 0.
    assert macro_f1(y, all_a, ["a", "b"]) == pytest.approx(3 / 7)


def test_metrics_refuse_a_test_set_with_one_class() -> None:
    with pytest.raises(BaselineError):
        roc_auc(np.array([True, True]), np.array([0.1, 0.2]))
    with pytest.raises(BaselineError):
        average_precision(np.array([False, False]), np.array([0.1, 0.2]))


def test_binary_linear_baseline_finds_a_real_signal() -> None:
    columns = _features()
    columns["bought"] = [int(a > 0.3) for a in columns["amount"]]  # type: ignore[operator]
    scores = _run(pa.table(columns), "bought")
    assert scores["prior"].metric == "roc_auc"
    assert scores["prior"].value == 0.5
    assert scores["linear"].value > 0.9
    assert scores["linear"].higher_is_better
    assert scores["linear"].features == 2 + 4  # two numbers, three regions and "other"


def test_rare_positive_class_is_scored_with_average_precision() -> None:
    columns = _features()
    columns["fraud"] = [int(a > 1.2) for a in columns["amount"]]  # type: ignore[operator]
    scores = _run(pa.table(columns), "fraud")
    assert scores["prior"].metric == "average_precision"
    test_rate = sum(columns["fraud"]) / N  # type: ignore[arg-type]
    assert scores["prior"].value == pytest.approx(test_rate, abs=0.06)
    assert scores["linear"].value > 2 * scores["prior"].value


def test_regression_baselines_use_mean_absolute_error() -> None:
    columns = _features()
    columns["spend"] = [3.0 * a + 10 for a in columns["amount"]]  # type: ignore[operator]
    scores = _run(pa.table(columns), "spend")
    assert scores["prior"].metric == "mae"
    assert not scores["prior"].higher_is_better
    assert scores["linear"].value < 0.1 * scores["prior"].value


def test_multiclass_prior_predicts_the_majority_class() -> None:
    columns = _features()
    columns["segment"] = [
        "big" if a > 0.5 else "small" if a < -0.5 else "mid"  # type: ignore[operator]
        for a in columns["amount"]
    ]
    scores = _run(pa.table(columns), "segment")
    assert scores["prior"].metric == "macro_f1"
    # Predicting one class of three: at best a third of the classes score above zero.
    assert scores["prior"].value < 0.34
    assert scores["linear"].value > scores["prior"].value


def test_rows_with_a_missing_target_are_left_out() -> None:
    columns = _features()
    bought: list[object] = [int(a > 0.3) for a in columns["amount"]]  # type: ignore[operator]
    for i in range(0, N, 10):
        bought[i] = None
    columns["bought"] = bought
    scores = _run(pa.table(columns), "bought")
    assert scores["linear"].train_rows + scores["linear"].test_rows == N - N // 10


def test_a_table_without_usable_features_is_refused() -> None:
    table = pa.table({"id": [f"r{i}" for i in range(N)], "bought": [i % 2 for i in range(N)]})
    with pytest.raises(BaselineError, match="no usable"):
        _run(table, "bought")
