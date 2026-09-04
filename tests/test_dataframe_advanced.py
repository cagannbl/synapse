import math
import os
import tempfile
import numpy as np
import pytest

from synapse.core.dataframe import DataFrame, dataframe, col, read_arrow_ipc, write_arrow_ipc
from synapse.core.arrow_ipc import (
    ARROW_MAGIC,
    TYPE_INT64,
    TYPE_FLOAT64,
    TYPE_UTF8,
)


# =============================================================================
# 1. GroupBy & Aggregations
# =============================================================================
def test_groupby_single_key_all_aggregations():
    """Test groupby with all supported aggregations: sum, mean, min, max, count, std, var, first, last."""
    df = DataFrame({
        "dept": ["IT", "Sales", "IT", "Sales", "IT", "HR"],
        "salary": [100000, 60000, 120000, 80000, 80000, 50000],
        "bonus": [10000.0, 5000.0, 20000.0, 7000.0, 6000.0, 3000.0],
        "age": [30, 25, 45, 35, 28, 50],
        "emp_id": ["E1", "E2", "E3", "E4", "E5", "E6"],
    })

    # Test individual aggregations in one dict
    aggs = {
        "salary": "mean",
        "bonus": "sum",
        "age": "max",
        "emp_id": "count",
    }
    res = df.groupby("dept").agg(aggs)

    assert res.shape == (3, 5)
    assert set(res.columns) == {"dept", "salary", "bonus", "age", "emp_id"}

    it_row = res.filter(col("dept") == "IT")
    assert it_row["salary"][0] == pytest.approx((100000 + 120000 + 80000) / 3.0)
    assert it_row["bonus"][0] == pytest.approx(10000.0 + 20000.0 + 6000.0)
    assert it_row["age"][0] == 45
    assert it_row["emp_id"][0] == 3

    hr_row = res.filter(col("dept") == "HR")
    assert hr_row["salary"][0] == 50000
    assert hr_row["bonus"][0] == pytest.approx(3000.0)
    assert hr_row["age"][0] == 50
    assert hr_row["emp_id"][0] == 1


def test_groupby_statistical_and_order_aggregations():
    """Test std, var, first, last, min aggregations."""
    df = DataFrame({
        "dept": ["A", "A", "A", "B", "B"],
        "val": [10.0, 20.0, 30.0, 100.0, 200.0],
        "name": ["first_a", "mid_a", "last_a", "first_b", "last_b"],
    })

    res = df.groupby("dept").agg({
        "val": "std",
        "name": "first",
    })
    a_row = res.filter(col("dept") == "A")
    # Sample std of [10, 20, 30] is 10.0
    assert a_row["val"][0] == pytest.approx(10.0)
    assert a_row["name"][0] == "first_a"

    # Test var, min, last
    res2 = df.groupby("dept").agg({
        "val": "var",
        "name": "last",
    })
    a_row2 = res2.filter(col("dept") == "A")
    # Sample var of [10, 20, 30] is 100.0
    assert a_row2["val"][0] == pytest.approx(100.0)
    assert a_row2["name"][0] == "last_a"


def test_groupby_multiple_keys():
    """Test groupby with composite keys (list of column names)."""
    df = DataFrame({
        "dept": ["IT", "IT", "IT", "Sales", "Sales"],
        "level": ["Senior", "Junior", "Senior", "Senior", "Junior"],
        "salary": [120000, 70000, 130000, 90000, 55000],
    })

    grouped = df.groupby(["dept", "level"]).agg({"salary": "mean"})
    assert grouped.shape == (4, 3)
    assert grouped.columns == ["dept", "level", "salary"]

    it_senior = grouped.filter((col("dept") == "IT") & (col("level") == "Senior"))
    assert len(it_senior) == 1
    assert it_senior["salary"][0] == pytest.approx(125000.0)

    sales_junior = grouped.filter((col("dept") == "Sales") & (col("level") == "Junior"))
    assert len(sales_junior) == 1
    assert sales_junior["salary"][0] == pytest.approx(55000.0)


def test_groupby_varargs_syntax():
    """Test groupby with varargs df.groupby('dept', 'level')."""
    df = DataFrame({
        "dept": ["Eng", "Eng", "Sales"],
        "loc": ["NYC", "NYC", "SF"],
        "rating": [4.5, 4.8, 4.2],
    })
    grouped = df.groupby("dept", "loc").agg({"rating": "max"})
    assert grouped.shape == (2, 3)
    eng_nyc = grouped.filter(col("dept") == "Eng")
    assert eng_nyc["rating"][0] == pytest.approx(4.8)


def test_groupby_multi_agg_per_column():
    """Test groupby where a column has a list of aggregations e.g. {'salary': ['mean', 'max']}."""
    df = DataFrame({
        "team": ["Alpha", "Alpha", "Beta"],
        "pts": [10, 20, 30],
    })
    grouped = df.groupby("team").agg({"pts": ["mean", "max"]})
    assert "pts_mean" in grouped.columns
    assert "pts_max" in grouped.columns
    alpha = grouped.filter(col("team") == "Alpha")
    assert alpha["pts_mean"][0] == pytest.approx(15.0)
    assert alpha["pts_max"][0] == 20


def test_groupby_convenience_methods():
    """Test GroupBy convenience methods .mean(), .sum(), .min(), .max(), .std(), .var()."""
    df = DataFrame({
        "group": ["G1", "G1", "G2", "G2"],
        "score1": [10.0, 20.0, 30.0, 40.0],
        "score2": [100.0, 200.0, 300.0, 400.0],
    })
    means = df.groupby("group").mean()
    assert means.shape == (2, 3)
    g1_mean = means.filter(col("group") == "G1")
    assert g1_mean["score1"][0] == pytest.approx(15.0)
    assert g1_mean["score2"][0] == pytest.approx(150.0)

    sums = df.groupby("group").sum()
    g2_sum = sums.filter(col("group") == "G2")
    assert g2_sum["score1"][0] == pytest.approx(70.0)
    assert g2_sum["score2"][0] == pytest.approx(700.0)


# =============================================================================
# 2. Hash Join (inner, left, outer)
# =============================================================================
def test_join_inner():
    """Test Hash Join inner join behavior."""
    users = DataFrame({
        "id": [1, 2, 3, 4],
        "name": ["Alice", "Bob", "Charlie", "David"],
    })
    orders = DataFrame({
        "id": [2, 3, 5],
        "amount": [250.0, 180.0, 400.0],
    })

    joined = users.join(orders, on="id", how="inner")
    assert len(joined) == 2
    assert set(joined.columns) == {"id", "name", "amount"}
    assert list(joined["id"]) == [2, 3]
    assert list(joined["name"]) == ["Bob", "Charlie"]
    assert list(joined["amount"]) == [250.0, 180.0]


def test_join_left():
    """Test Hash Join left outer join behavior."""
    users = DataFrame({
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
    })
    salaries = DataFrame({
        "id": [2, 3, 4],
        "salary": [80000, 90000, 100000],
    })

    joined = users.join(salaries, on="id", how="left")
    assert len(joined) == 3
    assert list(joined["id"]) == [1, 2, 3]
    assert list(joined["name"]) == ["Alice", "Bob", "Charlie"]
    # Row 1 has no match in right -> None
    assert joined["salary"][0] is None
    assert joined["salary"][1] == 80000
    assert joined["salary"][2] == 90000


def test_join_outer():
    """Test Hash Join full outer join behavior."""
    df_left = DataFrame({
        "id": [1, 2, 3],
        "val_left": ["L1", "L2", "L3"],
    })
    df_right = DataFrame({
        "id": [2, 3, 4],
        "val_right": ["R2", "R3", "R4"],
    })

    joined = df_left.join(df_right, on="id", how="outer")
    assert len(joined) == 4
    assert set(joined["id"]) == {1, 2, 3, 4}

    # Row with id=1 has left value, right value is None
    row1 = joined.filter(col("id") == 1)
    assert row1["val_left"][0] == "L1"
    assert row1["val_right"][0] is None

    # Row with id=4 has right value, left value is None
    row4 = joined.filter(col("id") == 4)
    assert row4["val_left"][0] is None
    assert row4["val_right"][0] == "R4"

    # Rows with id 2 and 3 have both
    row2 = joined.filter(col("id") == 2)
    assert row2["val_left"][0] == "L2"
    assert row2["val_right"][0] == "R2"


def test_join_column_collision_resolution():
    """Test that colliding non-key column names are prefixed/suffixed properly."""
    df1 = DataFrame({
        "id": [1, 2],
        "score": [85, 90],
        "status": ["active", "active"],
    })
    df2 = DataFrame({
        "id": [1, 2],
        "score": [95, 100],
        "status": ["pending", "verified"],
    })

    joined = df1.join(df2, on="id", how="inner")
    assert "score" in joined.columns
    assert "score_right" in joined.columns
    assert "status" in joined.columns
    assert "status_right" in joined.columns

    assert list(joined["score"]) == [85, 90]
    assert list(joined["score_right"]) == [95, 100]
    assert list(joined["status"]) == ["active", "active"]
    assert list(joined["status_right"]) == ["pending", "verified"]


def test_join_different_key_names():
    """Test join with left_on and right_on having different column names."""
    employees = DataFrame({
        "emp_code": ["E01", "E02", "E03"],
        "name": ["Ann", "Ben", "Carl"],
    })
    payroll = DataFrame({
        "badge_no": ["E02", "E03", "E04"],
        "rate": [50.0, 60.0, 70.0],
    })

    joined = employees.join(payroll, left_on="emp_code", right_on="badge_no", how="inner")
    assert len(joined) == 2
    assert "emp_code" in joined.columns
    assert "badge_no" in joined.columns
    assert list(joined["emp_code"]) == ["E02", "E03"]
    assert list(joined["badge_no"]) == ["E02", "E03"]
    assert list(joined["rate"]) == [50.0, 60.0]


def test_join_composite_keys():
    """Test join with multiple join keys (composite key)."""
    df1 = DataFrame({
        "dept": ["IT", "IT", "HR"],
        "year": [2024, 2025, 2024],
        "budget": [1000, 1200, 800],
    })
    df2 = DataFrame({
        "dept": ["IT", "IT", "HR"],
        "year": [2024, 2026, 2024],
        "actual": [950, 1100, 780],
    })

    joined = df1.join(df2, on=["dept", "year"], how="inner")
    assert len(joined) == 2
    it_2024 = joined.filter((col("dept") == "IT") & (col("year") == 2024))
    assert it_2024["budget"][0] == 1000
    assert it_2024["actual"][0] == 950


# =============================================================================
# 3. Multi-Column Sorting
# =============================================================================
def test_sort_single_column():
    """Test single column sorting ascending and descending."""
    df = DataFrame({"x": [30, 10, 50, 20, 40]})
    asc = df.sort("x", ascending=True)
    assert list(asc["x"]) == [10, 20, 30, 40, 50]

    desc = df.sort("x", ascending=False)
    assert list(desc["x"]) == [50, 40, 30, 20, 10]


def test_sort_multi_column_mixed_order():
    """Test multi-column sorting with mixed ascending/descending order."""
    df = DataFrame({
        "dept": ["IT", "Sales", "IT", "Sales", "IT", "HR"],
        "salary": [80000, 60000, 95000, 75000, 70000, 50000],
        "name": ["Bob", "Dan", "Alice", "Charlie", "Eve", "Frank"],
    })

    # Sort by dept ASC, then within dept salary DESC
    sorted_df = df.sort(columns=["dept", "salary"], ascending=[True, False])

    # Expect:
    # HR: [50000]
    # IT: [95000 (Alice), 80000 (Bob), 70000 (Eve)]
    # Sales: [75000 (Charlie), 60000 (Dan)]
    dept_res = list(sorted_df["dept"])
    salary_res = list(sorted_df["salary"])
    name_res = list(sorted_df["name"])

    assert dept_res == ["HR", "IT", "IT", "IT", "Sales", "Sales"]
    assert salary_res == [50000, 95000, 80000, 70000, 75000, 60000]
    assert name_res == ["Frank", "Alice", "Bob", "Eve", "Charlie", "Dan"]


def test_sort_stability_with_ties():
    """Test that equal values preserve original relative ordering (stable sort)."""
    df = DataFrame({
        "category": ["A", "B", "A", "B", "A"],
        "order_tag": [1, 2, 3, 4, 5],
    })
    # Sort by category ascending; tags within A should be [1, 3, 5], within B should be [2, 4]
    sorted_df = df.sort("category", ascending=True)
    assert list(sorted_df["category"]) == ["A", "A", "A", "B", "B"]
    assert list(sorted_df["order_tag"]) == [1, 3, 5, 2, 4]


# =============================================================================
# 4. Arrow IPC Binary Serialization & Deserialization
# =============================================================================
def test_arrow_ipc_binary_roundtrip_in_memory():
    """Test in-memory roundtrip serialization with int64, float64, and utf8."""
    df = DataFrame({
        "id": [101, 102, 103],
        "rating": [4.9, 3.8, 5.0],
        "title": ["Intro to Synapse", "Autograd Internals", "Distributed Swarm"],
    })

    # Serialize to bytes
    ipc_bytes = df.to_arrow_ipc()
    assert isinstance(ipc_bytes, bytes)
    assert ipc_bytes.startswith(ARROW_MAGIC)
    assert ipc_bytes.endswith(ARROW_MAGIC + b"\x00\x00")

    # Deserialize from bytes
    df_recovered = DataFrame.from_arrow_ipc(ipc_bytes)
    assert df_recovered.shape == df.shape
    assert df_recovered.columns == df.columns
    assert np.all(df_recovered["id"] == [101, 102, 103])
    assert np.allclose(df_recovered["rating"], [4.9, 3.8, 5.0])
    assert list(df_recovered["title"]) == ["Intro to Synapse", "Autograd Internals", "Distributed Swarm"]


def test_arrow_ipc_validity_bitmap_with_nulls():
    """Test that Arrow IPC accurately encodes and decodes validity bitmaps for null/None values."""
    df = DataFrame({
        "int_nulls": [10, None, 30, None, 50],
        "float_nulls": [1.1, 2.2, None, 4.4, None],
        "str_nulls": ["Alpha", None, "Gamma", None, "Epsilon"],
    })

    ipc_bytes = write_arrow_ipc(df)
    recovered = read_arrow_ipc(ipc_bytes)

    assert len(recovered) == 5
    # Check int nulls
    assert recovered["int_nulls"][0] == 10
    assert recovered["int_nulls"][1] is None
    assert recovered["int_nulls"][2] == 30
    assert recovered["int_nulls"][3] is None
    assert recovered["int_nulls"][4] == 50

    # Check float nulls
    assert recovered["float_nulls"][0] == pytest.approx(1.1)
    assert recovered["float_nulls"][2] is None

    # Check str nulls
    assert recovered["str_nulls"][0] == "Alpha"
    assert recovered["str_nulls"][1] is None
    assert recovered["str_nulls"][2] == "Gamma"
    assert recovered["str_nulls"][3] is None
    assert recovered["str_nulls"][4] == "Epsilon"


def test_arrow_ipc_file_roundtrip():
    """Test saving to file and reading back from file."""
    df = DataFrame({
        "code": [1, 2, 3],
        "weight": [72.5, 68.0, 85.2],
        "city": ["Berlin", "Tokyo", "London"],
    })

    with tempfile.NamedTemporaryFile(suffix=".arrow", delete=False) as tmp:
        path = tmp.name

    try:
        df.to_arrow_ipc(path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

        df_loaded = DataFrame.from_arrow_ipc(path)
        assert df_loaded.shape == (3, 3)
        assert df_loaded.columns == ["code", "weight", "city"]
        assert list(df_loaded["city"]) == ["Berlin", "Tokyo", "London"]
        assert np.allclose(df_loaded["weight"], [72.5, 68.0, 85.2])
    finally:
        if os.path.exists(path):
            os.remove(path)


# =============================================================================
# 5. Memory-Mapped IPC Reading (mmap.mmap)
# =============================================================================
def test_arrow_ipc_memory_mapped_reading():
    """Test memory-mapped zero-copy reading using mmap.mmap."""
    large_n = 1000
    df = DataFrame({
        "idx": list(range(large_n)),
        "val": [float(i * 1.5) for i in range(large_n)],
        "label": [f"item_{i}" for i in range(large_n)],
    })

    with tempfile.NamedTemporaryFile(suffix=".arrow", delete=False) as tmp:
        path = tmp.name

    try:
        df.to_arrow_ipc(path)

        # Read with mmap=True
        mmap_df = DataFrame.from_arrow_ipc(path, mmap=True)
        assert mmap_df.shape == (large_n, 3)
        assert hasattr(mmap_df, "_mmap") and mmap_df._mmap is not None

        # Verify zero-copy numeric column values
        assert mmap_df["idx"][0] == 0
        assert mmap_df["idx"][500] == 500
        assert mmap_df["val"][10] == pytest.approx(15.0)
        assert mmap_df["label"][100] == "item_100"

        # Verify operations like filter work on memory-mapped DataFrame
        high_vals = mmap_df.filter(col("idx") >= 995)
        assert len(high_vals) == 5
        assert list(high_vals["idx"]) == [995, 996, 997, 998, 999]

        mmap_df.close()
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_arrow_ipc_context_manager():
    """Test memory-mapped reading within a context manager (with ...)."""
    df = DataFrame({
        "x": [10, 20, 30, 40],
        "y": [1.5, 2.5, 3.5, 4.5],
    })

    with tempfile.NamedTemporaryFile(suffix=".arrow", delete=False) as tmp:
        path = tmp.name

    try:
        df.to_arrow_ipc(path)
        with DataFrame.from_arrow_ipc(path, mmap=True) as mmap_df:
            assert len(mmap_df) == 4
            assert list(mmap_df["x"]) == [10, 20, 30, 40]
        # Cleanly released upon exit
    finally:
        if os.path.exists(path):
            os.remove(path)


# =============================================================================
# 6. LazyFrame and DataFrame Compatibility
# =============================================================================
def test_lazyframe_multi_sort_and_limit():
    """Test LazyFrame query plan optimization with multi-column sort and limit."""
    df = DataFrame({
        "dept": ["IT", "Sales", "IT", "Sales", "IT"],
        "salary": [80000, 60000, 95000, 75000, 70000],
        "active": [True, True, True, False, True],
    })

    plan = (
        df.lazy()
        .filter(col("active") == True)
        .sort(columns=["dept", "salary"], ascending=[True, False])
        .limit(2)
        .select("dept", "salary")
    )
    res = plan.collect()
    assert len(res) == 2
    assert res.columns == ["dept", "salary"]
    assert list(res["dept"]) == ["IT", "IT"]
    assert list(res["salary"]) == [95000, 80000]


def test_dataframe_limit():
    """Test DataFrame.limit() method."""
    df = DataFrame({"a": [1, 2, 3, 4, 5]})
    assert len(df.limit(3)) == 3
    assert list(df.limit(3)["a"]) == [1, 2, 3]
