import datetime as dt

import pyarrow as pa
import pytest

from ml_research_agent.profile import Kind, ProfileError, Task, profile

N = 120


def _table(**overrides: list[object]) -> pa.Table:
    base: dict[str, list[object]] = {
        "row_id": list(range(N)),
        "customer_id": [f"c{i % 30}" for i in range(N)],
        "signup": [dt.datetime(2026, 1, 1) + dt.timedelta(hours=i) for i in range(N)],
        "amount": [float(i % 17) * 1.5 for i in range(N)],
        "visits": [i % 9 for i in range(N)],
        "plan": [("free", "pro", "team")[i % 3] for i in range(N)],
        "is_trial": [i % 2 == 0 for i in range(N)],
        "note": [f"customer wrote a long message number {i} about the product" for i in range(N)],
        "region": ["emea"] * N,
        "legacy": [None] * N,
        "churned": [int(i % 5 == 0) for i in range(N)],
    }
    return pa.table(base | overrides)


def test_each_column_gets_the_kind_a_person_would_give_it() -> None:
    p = profile(_table(), "churned")
    kinds = {c.name: c.kind for c in p.columns}
    assert kinds == {
        "row_id": Kind.IDENTIFIER,
        "customer_id": Kind.CATEGORICAL,
        "signup": Kind.DATETIME,
        "amount": Kind.NUMERIC,
        "visits": Kind.NUMERIC,
        "plan": Kind.CATEGORICAL,
        "is_trial": Kind.BOOLEAN,
        "note": Kind.TEXT,
        "region": Kind.CONSTANT,
        "legacy": Kind.EMPTY,
    }
    assert "churned" not in kinds


def test_strings_are_split_into_text_keys_and_categories() -> None:
    repeated = [f"the customer described problem number {i % 60} in some detail" for i in range(N)]
    assert profile(_table(note=repeated), "churned").column("note").kind is Kind.TEXT
    keys = [f"ord-{i:05d}" for i in range(N)]
    assert profile(_table(note=keys), "churned").column("note").kind is Kind.IDENTIFIER
    cities = [f"city {i % 80}" for i in range(N)]
    assert profile(_table(note=cities), "churned").column("note").kind is Kind.TEXT


def test_identifiers_constants_and_empty_columns_are_not_features() -> None:
    p = profile(_table(), "churned")
    unusable = {c.name for c in p.columns if not c.usable_as_feature}
    assert unusable == {"row_id", "region", "legacy"}


def test_missing_share_and_distinct_counts() -> None:
    amounts: list[object] = [None if i % 4 == 0 else float(i % 3) for i in range(N)]
    col = profile(_table(amount=amounts), "churned").column("amount")
    assert col.missing_share == pytest.approx(0.25)
    assert col.distinct == 3


def test_binary_target_reports_class_balance_largest_first() -> None:
    target = profile(_table(), "churned").target
    assert target.task is Task.BINARY
    assert list(target.class_shares.items()) == [("0", 0.8), ("1", 0.2)]
    assert target.minority_share == pytest.approx(0.2)


def test_target_task_follows_the_values() -> None:
    plans = profile(_table(), "plan").target
    assert plans.task is Task.MULTICLASS and len(plans.class_shares) == 3
    amount = profile(_table(), "amount").target
    assert amount.task is Task.REGRESSION and amount.minority_share is None
    counts = profile(_table(visits=list(range(N))), "visits").target
    assert counts.task is Task.REGRESSION


def test_time_and_group_candidates_are_found() -> None:
    p = profile(_table(), "churned")
    assert p.time_columns == ("signup",)
    # customer_id repeats (four rows each); row_id never does, so it cannot group rows.
    assert p.group_columns == ("customer_id",)


def test_duplicate_rows_are_counted() -> None:
    table = _table()
    doubled = pa.concat_tables([table, table.slice(0, 7)])
    assert profile(table, "churned").duplicate_rows == 0
    assert profile(doubled, "churned").duplicate_rows == 7


@pytest.mark.parametrize(
    ("table", "target", "message"),
    [
        (pa.table({"x": pa.array([], pa.int64()), "y": pa.array([], pa.int64())}), "y", "no rows"),
        (_table(), "missing", "not in the table"),
        (_table(churned=[1] * N), "churned", "single value"),
        (_table(churned=[None] * N), "churned", "no values"),
    ],
)
def test_unusable_inputs_are_refused_with_a_reason(
    table: pa.Table, target: str, message: str
) -> None:
    with pytest.raises(ProfileError, match=message):
        profile(table, target)
