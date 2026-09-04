import os
import io
import tempfile
import numpy as np
import pytest
from synapse.core.dataframe import DataFrame, dataframe, col, read_csv
from synapse.core.tensor import Tensor, tensor
from synapse.lexer.lexer import Lexer
from synapse.parser.parser import Parser
from synapse.vm.compiler import Compiler
from synapse.vm.virtual_machine import VirtualMachine


def run_synapse(code: str) -> VirtualMachine:
    tokens = Lexer(code).tokenize()
    ast = Parser(tokens).parse()
    compiled = Compiler().compile(ast)
    vm = VirtualMachine()
    vm.execute(compiled)
    return vm


# =========================================================================
# 1. DataFrame Creation & Basic Properties
# =========================================================================
def test_dataframe_from_dict():
    data = {
        "name": ["Alice", "Bob", "Charlie"],
        "age": [25, 30, 35],
        "salary": [70000.0, 85000.0, 92000.0]
    }
    df = DataFrame(data)
    assert df.columns == ["name", "age", "salary"]
    assert df.shape == (3, 3)
    assert len(df) == 3
    assert np.all(df["age"] == [25, 30, 35])


def test_dataframe_from_records():
    records = [
        {"product": "Laptop", "price": 1200, "qty": 5},
        {"product": "Mouse", "price": 25, "qty": 50},
        {"product": "Monitor", "price": 300, "qty": 15},
    ]
    df = dataframe(records)
    assert df.shape == (3, 3)
    assert "product" in df.columns
    assert df["price"][0] == 1200


def test_dataframe_from_numpy_array():
    mat = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    df = DataFrame(mat, columns=["x1", "x2"])
    assert df.shape == (3, 2)
    assert df.columns == ["x1", "x2"]
    assert np.all(df["x1"] == [1.0, 3.0, 5.0])


# =========================================================================
# 2. Selection, Projections & Slicing
# =========================================================================
def test_column_selection_and_slicing():
    df = DataFrame({
        "a": [1, 2, 3, 4, 5],
        "b": [10, 20, 30, 40, 50],
        "c": [100, 200, 300, 400, 500]
    })
    sub_df = df[["a", "c"]]
    assert sub_df.columns == ["a", "c"]
    assert sub_df.shape == (5, 2)

    head_df = df.head(2)
    assert len(head_df) == 2
    assert head_df["a"][1] == 2

    tail_df = df.tail(2)
    assert len(tail_df) == 2
    assert tail_df["a"][0] == 4


# =========================================================================
# 3. Filtering & Column Expressions (col)
# =========================================================================
def test_filtering_with_col_expression():
    df = DataFrame({
        "employee": ["E1", "E2", "E3", "E4"],
        "score": [45, 88, 92, 60],
        "active": [True, True, False, True]
    })

    # Filter score > 70
    passed = df.filter(col("score") > 70)
    assert len(passed) == 2
    assert list(passed["employee"]) == ["E2", "E3"]

    # Arithmetic in expression
    bonus = df.select(col("employee"), (col("score") * 10).alias("bonus"))
    assert bonus.columns == ["employee", "bonus"]
    assert bonus["bonus"][0] == 450


def test_combined_boolean_filter():
    df = DataFrame({
        "val": [10, 20, 30, 40, 50]
    })
    filtered = df.filter((col("val") >= 20) & (col("val") <= 40))
    assert len(filtered) == 3
    assert list(filtered["val"]) == [20, 30, 40]


# =========================================================================
# 4. Transformations & Sorting
# =========================================================================
def test_with_column_and_sort():
    df = DataFrame({
        "item": ["A", "B", "C"],
        "price": [100.0, 200.0, 150.0]
    })
    with_tax = df.with_column("total", col("price") * 1.2)
    assert "total" in with_tax.columns
    assert with_tax["total"][0] == pytest.approx(120.0)

    # Sort descending
    sorted_df = with_tax.sort("price", ascending=False)
    assert list(sorted_df["item"]) == ["B", "C", "A"]


# =========================================================================
# 5. Group By & Aggregations
# =========================================================================
def test_groupby_aggregations():
    df = DataFrame({
        "department": ["IT", "Sales", "IT", "Sales", "HR"],
        "salary": [80000, 60000, 95000, 75000, 50000],
        "bonus": [5000, 10000, 6000, 12000, 3000]
    })
    grouped = df.groupby("department").agg({"salary": "mean", "bonus": "sum"})
    assert grouped.shape == (3, 3)
    assert "department" in grouped.columns

    it_row = grouped.filter(col("department") == "IT")
    assert it_row["salary"][0] == pytest.approx(87500.0)
    assert it_row["bonus"][0] == 11000


# =========================================================================
# 6. Zero-Copy Tensor Interop (DataFrame <-> Tensor)
# =========================================================================
def test_dataframe_to_tensor_and_back():
    df = DataFrame({
        "id": ["p1", "p2", "p3"],
        "feature1": [1.0, 2.0, 3.0],
        "feature2": [4.0, 5.0, 6.0]
    })

    # To Tensor
    t = df.to_tensor(["feature1", "feature2"], requires_grad=True)
    assert isinstance(t, Tensor)
    assert t.shape == (3, 2)
    assert t.requires_grad is True

    # Mathematical operation on model tensor
    weights = tensor([[0.5], [1.5]], requires_grad=True)
    pred = t @ weights
    loss = pred.sum()
    loss.backward()

    assert t.grad is not None
    assert weights.grad is not None

    # Tensor back to DataFrame
    df_recovered = t.to_dataframe(columns=["f1", "f2"])
    assert isinstance(df_recovered, DataFrame)
    assert df_recovered.columns == ["f1", "f2"]
    assert df_recovered.shape == (3, 2)
    assert np.allclose(df_recovered["f1"], [1.0, 2.0, 3.0])


# =========================================================================
# 7. Lazy Evaluation & Optimization (LazyFrame)
# =========================================================================
def test_lazyframe_pipeline():
    df = DataFrame({
        "name": ["A", "B", "C", "D", "E"],
        "val": [10, 50, 30, 90, 70],
        "tag": ["x", "y", "x", "y", "x"]
    })

    # Build query plan
    lazy_plan = (
        df.lazy()
        .filter(col("val") > 20)
        .sort("val", ascending=False)
        .limit(3)
        .select("name", "val")
    )

    # Collect result
    result = lazy_plan.collect()
    assert len(result) == 3
    assert list(result["name"]) == ["D", "E", "B"]
    assert list(result["val"]) == [90, 70, 50]
    assert result.columns == ["name", "val"]


# =========================================================================
# 8. CSV I/O with Type Inference
# =========================================================================
def test_csv_read_and_write():
    csv_text = """title,views,rating,published
Intro to Synapse,1500,4.9,true
Autograd Deep Dive,2300,4.8,true
Draft Video,0,0.0,false
"""
    df = read_csv(io.StringIO(csv_text))
    assert df.shape == (3, 4)
    assert df["views"][0] == 1500
    assert df["rating"][1] == 4.8
    assert bool(df["published"][0]) is True
    assert bool(df["published"][2]) is False

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        df.to_csv(tmp_path)
        assert os.path.exists(tmp_path)
        df_reloaded = read_csv(tmp_path)
        assert df_reloaded.shape == (3, 4)
        assert df_reloaded["views"][1] == 2300
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# =========================================================================
# 9. Synapse VM Integration (Script Level Native DataFrame)
# =========================================================================
def test_synapse_vm_dataframe_integration():
    code = """
let data = {
    "x": [1.0, 2.0, 3.0],
    "y": [4.0, 5.0, 6.0]
}

let df = dataframe(data)
let filtered = df.filter(col("x") > 1.5)
let t = filtered.to_tensor(["x", "y"])
let tensor_shape = t.shape
"""
    vm = run_synapse(code)
    df = vm.globals["df"]
    assert isinstance(df, DataFrame)
    assert df.shape == (3, 2)

    filtered = vm.globals["filtered"]
    assert len(filtered) == 2

    t = vm.globals["t"]
    assert isinstance(t, Tensor)
    assert t.shape == (2, 2)
    assert np.allclose(t.data, [[2.0, 5.0], [3.0, 6.0]])
