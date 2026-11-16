"""Describe a tabular dataset before anything is trained on it.

The profile is what the evaluation planner and the leakage checks read: which columns are
numbers, categories, times or identifiers, how much is missing, how balanced the target
is, and which columns could order the rows in time or tie rows to the same entity. It is
computed by plain code so the agent starts from facts it did not have to guess.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

import pyarrow as pa
import pyarrow.compute as pc

# A string column with at most this many distinct values is treated as a category.
MAX_CATEGORIES = 50
# Long free text is not a category even when values repeat.
TEXT_MEAN_LENGTH = 40
# A column must repeat this much on average to tie rows to one entity.
MIN_ROWS_PER_GROUP = 2.0


class Kind(Enum):
    NUMERIC = "numeric"
    BOOLEAN = "boolean"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    TEXT = "text"
    IDENTIFIER = "identifier"
    CONSTANT = "constant"
    EMPTY = "empty"


class Task(Enum):
    BINARY = "binary classification"
    MULTICLASS = "multiclass classification"
    REGRESSION = "regression"


class ProfileError(ValueError):
    pass


@dataclass(frozen=True)
class Column:
    name: str
    kind: Kind
    missing_share: float
    distinct: int

    @property
    def usable_as_feature(self) -> bool:
        """Identifiers, constants and empty columns carry nothing a model should learn."""
        return self.kind not in {Kind.IDENTIFIER, Kind.CONSTANT, Kind.EMPTY}


@dataclass(frozen=True)
class Target:
    name: str
    task: Task
    missing_share: float
    # Class shares for classification, largest first; empty for regression.
    class_shares: Mapping[str, float]

    @property
    def minority_share(self) -> float | None:
        return min(self.class_shares.values()) if self.class_shares else None


@dataclass(frozen=True)
class Profile:
    rows: int
    columns: tuple[Column, ...]
    target: Target
    time_columns: tuple[str, ...]
    group_columns: tuple[str, ...]
    duplicate_rows: int

    def column(self, name: str) -> Column:
        for col in self.columns:
            if col.name == name:
                return col
        raise KeyError(name)


def _distinct(values: pa.ChunkedArray) -> int:
    """Distinct non-null values. A column of only nulls has the null type and none."""
    if pa.types.is_null(values.type):
        return 0
    return int(pc.count_distinct(values).as_py())


def _is_time(dtype: pa.DataType) -> bool:
    return bool(pa.types.is_timestamp(dtype) or pa.types.is_date(dtype))


def _kind(values: pa.ChunkedArray, rows: int, distinct: int) -> Kind:
    present = rows - values.null_count
    dtype = values.type
    if present == 0:
        return Kind.EMPTY
    if distinct == 1:
        return Kind.CONSTANT
    if pa.types.is_boolean(dtype):
        return Kind.BOOLEAN
    if _is_time(dtype):
        return Kind.DATETIME
    if pa.types.is_floating(dtype) or pa.types.is_decimal(dtype):
        return Kind.NUMERIC
    if pa.types.is_integer(dtype):
        # Whole numbers that never repeat are row keys, not measurements.
        return Kind.IDENTIFIER if distinct == present and present > MAX_CATEGORIES else Kind.NUMERIC
    if pa.types.is_string(dtype) or pa.types.is_large_string(dtype):
        # Length is checked first: free text is usually unique per row too, and it is
        # something a model can read, unlike a key.
        mean_length = pc.mean(pc.utf8_length(values)).as_py() or 0.0
        if mean_length > TEXT_MEAN_LENGTH:
            return Kind.TEXT
        if distinct == present and present > MAX_CATEGORIES:
            return Kind.IDENTIFIER
        return Kind.CATEGORICAL if distinct <= MAX_CATEGORIES else Kind.TEXT
    raise ProfileError(f"unsupported column type {dtype}")


def _target(table: pa.Table, name: str) -> Target:
    if name not in table.column_names:
        raise ProfileError(f"target column {name!r} is not in the table")
    values = table[name]
    present = pc.drop_null(values)
    if len(present) == 0:
        raise ProfileError(f"target column {name!r} has no values")
    distinct = _distinct(present)
    if distinct < 2:
        raise ProfileError(f"target column {name!r} has a single value")
    dtype = values.type
    numeric = pa.types.is_floating(dtype) or pa.types.is_decimal(dtype)
    many_integers = pa.types.is_integer(dtype) and distinct > MAX_CATEGORIES
    missing = values.null_count / len(values)
    if numeric or many_integers:
        return Target(name, Task.REGRESSION, missing, {})
    counts = pc.value_counts(present).to_pylist()
    counts.sort(key=lambda c: (-c["counts"], str(c["values"])))
    shares = {str(c["values"]): c["counts"] / len(present) for c in counts}
    return Target(name, Task.BINARY if distinct == 2 else Task.MULTICLASS, missing, shares)


def profile(table: pa.Table, target: str) -> Profile:
    """Profile ``table``; ``target`` is the column a model will be asked to predict."""
    rows = table.num_rows
    if rows == 0:
        raise ProfileError("the table has no rows")
    columns = []
    time_columns = []
    group_columns = []
    for name in table.column_names:
        if name == target:
            continue
        values = table[name]
        distinct = _distinct(values)
        kind = _kind(values, rows, distinct)
        columns.append(Column(name, kind, values.null_count / rows, distinct))
        present = rows - values.null_count
        if kind is Kind.DATETIME:
            time_columns.append(name)
        repeats = present / distinct >= MIN_ROWS_PER_GROUP if distinct else False
        key_like = pa.types.is_integer(values.type) or pa.types.is_string(values.type)
        if key_like and name.lower().endswith(("_id", "id")) and repeats and distinct > 1:
            group_columns.append(name)
    duplicates = rows - table.group_by(table.column_names).aggregate([]).num_rows
    return Profile(
        rows=rows,
        columns=tuple(columns),
        target=_target(table, target),
        time_columns=tuple(time_columns),
        group_columns=tuple(group_columns),
        duplicate_rows=duplicates,
    )
