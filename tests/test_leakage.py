import datetime as dt

import pyarrow as pa

from ml_research_agent.leakage import Check, check
from ml_research_agent.plan import Plan, Strategy, plan, split
from ml_research_agent.profile import profile

N = 400


def _base(**extra: list[object]) -> pa.Table:
    """A churn table with no leakage: the target depends weakly and noisily on usage."""
    columns: dict[str, list[object]] = {
        "visits": [(i * 7919) % 23 for i in range(N)],
        # A measurement with many distinct values, as real tables have, so rows do not
        # repeat by chance.
        "spend": [round(10 + ((i * 2654435761) % 10007) / 37, 2) for i in range(N)],
        "plan": [("free", "pro", "team")[(i * 31) % 3] for i in range(N)],
        "region": [("emea", "na", "apac")[(i * 17) % 3] for i in range(N)],
        "churned": [int((i * 7919) % 23 < 8 and (i * 13) % 5 != 0) for i in range(N)],
    }
    return pa.table(columns | extra)


def _run(table: pa.Table, target: str = "churned"):  # type: ignore[no-untyped-def]
    p = profile(table, target)
    chosen = plan(p)
    return check(table, p, chosen, split(table, p, chosen))


def test_a_table_without_leakage_is_clean() -> None:
    report = _run(_base())
    assert report.clean, report.findings


def test_a_category_that_is_the_target_in_disguise_is_found() -> None:
    table = _base()
    status = ["cancelled" if c else "active" for c in table["churned"].to_pylist()]
    report = _run(table.append_column("account_status", pa.array(status)))
    [finding] = report.by_check(Check.SINGLE_FEATURE)
    assert finding.column == "account_status" and finding.value == 1.0


def test_a_number_that_separates_the_classes_is_found() -> None:
    table = _base()
    days_since_cancel = [
        float(3 + i % 40) if c else 0.0 for i, c in enumerate(table["churned"].to_pylist())
    ]
    report = _run(table.append_column("days_since_cancel", pa.array(days_since_cancel)))
    assert [f.column for f in report.by_check(Check.SINGLE_FEATURE)] == ["days_since_cancel"]


def test_a_merely_useful_feature_is_not_reported() -> None:
    # visits drives the target in the generator, but noise keeps it well short of perfect.
    report = _run(_base())
    assert not report.by_check(Check.SINGLE_FEATURE)


def test_a_regression_feature_that_copies_the_target_is_found() -> None:
    revenue = [float((i * 37) % 101) + i / 7 for i in range(N)]
    table = pa.table(
        {
            "revenue": revenue,
            "revenue_with_tax": [r * 1.13 for r in revenue],
            "visits": [float((i * 7919) % 23) for i in range(N)],
        }
    )
    report = _run(table, target="revenue")
    assert [f.column for f in report.by_check(Check.SINGLE_FEATURE)] == ["revenue_with_tax"]


def test_duplicated_rows_across_the_split_are_counted() -> None:
    table = _base()
    doubled = pa.concat_tables([table, table])
    report = _run(doubled)
    [finding] = report.by_check(Check.ROW_OVERLAP)
    # Every row has one copy, which lands in training about 80 percent of the time.
    assert 0.6 < finding.value < 0.95


def test_entities_on_both_sides_of_a_random_split_are_reported() -> None:
    table = _base()
    p = profile(
        table.append_column("customer_id", pa.array([f"c{i % 40}" for i in range(N)])), "churned"
    )
    t = table.append_column("customer_id", pa.array([f"c{i % 40}" for i in range(N)]))
    # Force a random split even though a grouped one would be chosen.
    forced = Plan(Strategy.RANDOM, None, "roc_auc", 0.2, 0, ())
    report = check(t, p, forced, split(t, p, forced))
    [finding] = report.by_check(Check.ENTITY_OVERLAP)
    assert finding.column == "customer_id" and finding.value == 1.0


def test_a_grouped_split_has_no_entity_overlap() -> None:
    t = _base(customer_id=[f"c{i % 40}" for i in range(N)])
    assert not _run(t).by_check(Check.ENTITY_OVERLAP)


def test_a_date_recorded_after_the_cutoff_is_reported() -> None:
    start = dt.datetime(2026, 1, 1)
    seen = [start + dt.timedelta(hours=(i * 7919) % N) for i in range(N)]
    # closed_at is filled in later than the row itself, for some rows past the cutoff.
    closed = [s + dt.timedelta(days=2) for s in seen]
    t = _base(seen_at=seen, closed_at=closed)
    p = profile(t, "churned")
    chosen = plan(p)
    assert chosen.strategy is Strategy.TIME and chosen.column == "seen_at"
    report = check(t, p, chosen, split(t, p, chosen))
    [finding] = report.by_check(Check.AFTER_CUTOFF)
    assert finding.column == "closed_at" and 0 < finding.value < 1


def test_a_date_known_before_the_cutoff_is_fine() -> None:
    start = dt.datetime(2026, 1, 1)
    seen = [start + dt.timedelta(hours=(i * 7919) % N) for i in range(N)]
    signed_up = [s - dt.timedelta(days=90) for s in seen]
    report = _run(_base(seen_at=seen, signed_up_at=signed_up))
    assert not report.by_check(Check.AFTER_CUTOFF)


def test_every_finding_explains_what_was_measured() -> None:
    table = _base()
    status = ["cancelled" if c else "active" for c in table["churned"].to_pylist()]
    report = _run(table.append_column("account_status", pa.array(status)))
    [finding] = report.findings
    assert "account_status" in finding.detail and "1.000" in finding.detail
