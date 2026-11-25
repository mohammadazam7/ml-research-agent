import datetime as dt

import pyarrow as pa
import pytest

from ml_research_agent.plan import PlanError, Strategy, plan, split
from ml_research_agent.profile import profile

N = 400


def _table(*, time: bool = False, group: bool = False, positive_every: int = 3) -> pa.Table:
    columns: dict[str, list[object]] = {
        "amount": [float((i * 37) % 101) + i / 1000 for i in range(N)],
        "plan": [("free", "pro")[i % 2] for i in range(N)],
        "tier": [("a", "b", "c")[i % 3] for i in range(N)],
        "churned": [int(i % positive_every == 0) for i in range(N)],
    }
    if time:
        # Not in row order, so a split that just took the last rows would be wrong.
        columns["seen_at"] = [
            dt.datetime(2026, 1, 1) + dt.timedelta(hours=(i * 7919) % N) for i in range(N)
        ]
    if group:
        columns["customer_id"] = [f"c{i % 80}" for i in range(N)]
    return pa.table(columns)


def _planned(table: pa.Table, target: str = "churned", **kwargs: float):  # type: ignore[no-untyped-def]
    p = profile(table, target)
    chosen = plan(p, **kwargs)  # type: ignore[arg-type]
    return p, chosen, split(table, p, chosen)


def test_time_beats_group_beats_random() -> None:
    assert _planned(_table(time=True, group=True))[1].strategy is Strategy.TIME
    assert _planned(_table(group=True))[1].strategy is Strategy.GROUP
    assert _planned(_table())[1].strategy is Strategy.RANDOM


def test_every_plan_explains_itself() -> None:
    _, chosen, _ = _planned(_table(time=True, group=True))
    text = " ".join(chosen.reasons)
    assert "seen_at" in text and "customer_id" in text and chosen.column == "seen_at"


@pytest.mark.parametrize(
    ("table", "target", "metric"),
    [
        (_table(positive_every=3), "churned", "roc_auc"),
        (_table(positive_every=20), "churned", "average_precision"),
        (_table(), "amount", "mae"),
        (
            pa.table({"x": list(range(N)), "y": [("a", "b", "c")[i % 3] for i in range(N)]}),
            "y",
            "macro_f1",
        ),
    ],
)
def test_the_metric_follows_the_target(table: pa.Table, target: str, metric: str) -> None:
    assert plan(profile(table, target)).metric == metric


def test_a_split_covers_every_row_exactly_once() -> None:
    for table in (_table(time=True), _table(group=True), _table()):
        _, _, result = _planned(table)
        assert sorted(result.train + result.test) == list(range(N))
        assert not set(result.train) & set(result.test)


def test_time_split_trains_strictly_before_the_cutoff() -> None:
    table = _table(time=True)
    _, _, result = _planned(table)
    seen = table["seen_at"].to_pylist()
    assert max(seen[i] for i in result.train) < result.cutoff
    assert min(seen[i] for i in result.test) == result.cutoff
    assert len(result.test) == pytest.approx(N * 0.2, abs=2)


def test_rows_sharing_the_cutoff_time_all_go_to_the_test_side() -> None:
    days = [dt.datetime(2026, 1, 1) + dt.timedelta(days=i // 50) for i in range(N)]
    table = _table().append_column("seen_at", pa.array(days))
    _, _, result = _planned(table)
    assert {days[i] for i in result.train}.isdisjoint({days[i] for i in result.test})


def test_group_split_keeps_each_entity_on_one_side() -> None:
    table = _table(group=True)
    _, _, result = _planned(table)
    customers = table["customer_id"].to_pylist()
    assert {customers[i] for i in result.train}.isdisjoint({customers[i] for i in result.test})
    assert 0.1 < len(result.test) / N < 0.35


def test_random_split_keeps_the_class_balance() -> None:
    table = _table(positive_every=10)
    _, _, result = _planned(table)
    labels = table["churned"].to_pylist()
    share = sum(labels[i] for i in result.test) / len(result.test)
    assert share == pytest.approx(0.1, abs=0.01)
    assert len(result.test) == pytest.approx(N * 0.2, abs=2)


def test_the_same_inputs_give_the_same_split_and_a_new_seed_a_new_one() -> None:
    table = _table()
    first = _planned(table)[2]
    assert _planned(table)[2] == first
    assert _planned(table, seed=1)[2] != first


def test_duplicates_are_called_out_in_the_plan() -> None:
    table = _table()
    doubled = pa.concat_tables([table, table.slice(0, 5)])
    assert any("duplicate" in r for r in plan(profile(doubled, "churned")).reasons)
    assert not any("duplicate" in r for r in plan(profile(table, "churned")).reasons)


def test_plans_that_cannot_work_are_refused() -> None:
    small = _table().slice(0, 60)
    with pytest.raises(PlanError, match="fewer than 20"):
        plan(profile(small, "churned"))
    with pytest.raises(PlanError, match="between 0 and 1"):
        plan(profile(_table(), "churned"), test_share=1.5)
    table = _table(time=True)
    p = profile(table, "churned")
    with pytest.raises(PlanError, match="not the one that was profiled"):
        split(table.slice(0, 100), p, plan(p))


def test_a_time_column_with_gaps_or_one_value_cannot_order_rows() -> None:
    stamps: list[object] = [dt.datetime(2026, 1, 1) + dt.timedelta(hours=i) for i in range(N)]
    stamps[3] = None
    table = _table().append_column("seen_at", pa.array(stamps, pa.timestamp("us")))
    p = profile(table, "churned")
    with pytest.raises(PlanError, match="missing values"):
        split(table, p, plan(p))
