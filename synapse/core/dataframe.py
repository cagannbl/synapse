from __future__ import annotations
import csv
import io
import json
import math
import os
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union
import numpy as np


class ColumnExpr:
    """
    Sütun seviyesinde tembel sorgu ifadesi (Lazy Column Expression).
    Örn: col("revenue") > 1000, col("price") * 1.18, col("age") + 5
    """
    def __init__(self, fn: Callable[[DataFrame], np.ndarray], name: Optional[str] = None):
        self._fn = fn
        self._name = name

    def evaluate(self, df: DataFrame) -> np.ndarray:
        return self._fn(df)

    def __gt__(self, other: Union[ColumnExpr, Any]) -> ColumnExpr:
        return self._binary_op(other, lambda a, b: a > b)

    def __ge__(self, other: Union[ColumnExpr, Any]) -> ColumnExpr:
        return self._binary_op(other, lambda a, b: a >= b)

    def __lt__(self, other: Union[ColumnExpr, Any]) -> ColumnExpr:
        return self._binary_op(other, lambda a, b: a < b)

    def __le__(self, other: Union[ColumnExpr, Any]) -> ColumnExpr:
        return self._binary_op(other, lambda a, b: a <= b)

    def __eq__(self, other: Union[ColumnExpr, Any]) -> ColumnExpr:  # type: ignore[override]
        return self._binary_op(other, lambda a, b: a == b)

    def __ne__(self, other: Union[ColumnExpr, Any]) -> ColumnExpr:  # type: ignore[override]
        return self._binary_op(other, lambda a, b: a != b)

    def __add__(self, other: Union[ColumnExpr, Any]) -> ColumnExpr:
        return self._binary_op(other, lambda a, b: a + b)

    def __sub__(self, other: Union[ColumnExpr, Any]) -> ColumnExpr:
        return self._binary_op(other, lambda a, b: a - b)

    def __mul__(self, other: Union[ColumnExpr, Any]) -> ColumnExpr:
        return self._binary_op(other, lambda a, b: a * b)

    def __truediv__(self, other: Union[ColumnExpr, Any]) -> ColumnExpr:
        return self._binary_op(other, lambda a, b: a / b)

    def __and__(self, other: Union[ColumnExpr, Any]) -> ColumnExpr:
        return self._binary_op(other, lambda a, b: np.logical_and(a, b))

    def __or__(self, other: Union[ColumnExpr, Any]) -> ColumnExpr:
        return self._binary_op(other, lambda a, b: np.logical_or(a, b))

    def __invert__(self) -> ColumnExpr:
        return ColumnExpr(lambda df: np.logical_not(self.evaluate(df)))

    def alias(self, new_name: str) -> ColumnExpr:
        expr = ColumnExpr(self._fn, name=new_name)
        return expr

    def _binary_op(self, other: Any, op: Callable[[Any, Any], Any]) -> ColumnExpr:
        if isinstance(other, ColumnExpr):
            return ColumnExpr(lambda df: op(self.evaluate(df), other.evaluate(df)))
        return ColumnExpr(lambda df: op(self.evaluate(df), other))


def col(name: str) -> ColumnExpr:
    """Belirtilen sütun adına başvuran ifade oluşturur."""
    def _eval(df: DataFrame) -> np.ndarray:
        if name not in df.columns:
            raise KeyError(f"Column '{name}' not found in DataFrame. Available: {df.columns}")
        return df._columns[name]
    return ColumnExpr(_eval, name=name)


class DataFrame:
    """
    Synapse Yerleşik Sütun Bazlı Veri Çerçevesi (Columnar DataFrame).
    Apache Arrow felsefesiyle tasarlanmış; her sütun homojen, ardışık bellek dizisi olarak saklanır.
    NumPy, Pandas veya PyArrow kurulu olmasa dahi sıfır bağımlılıkla yüksek hızda çalışır.
    """
    def __init__(self, data: Optional[Union[Dict[str, Any], Sequence[Dict[str, Any]], np.ndarray]] = None, columns: Optional[Sequence[str]] = None):
        self._columns: Dict[str, np.ndarray] = {}
        self._length = 0

        if data is not None:
            self._ingest_data(data, columns)

    def _ingest_data(self, data: Any, columns: Optional[Sequence[str]] = None):
        if isinstance(data, dict):
            # Sözlük sütunları: {"name": ["Alice", "Bob"], "age": [25, 30]}
            first_key = next(iter(data), None)
            if first_key is not None:
                self._length = len(data[first_key])
            for k, v in data.items():
                arr = np.asarray(v)
                if len(arr) != self._length:
                    raise ValueError(f"Column '{k}' has length {len(arr)}, expected {self._length}")
                self._columns[k] = arr

        elif isinstance(data, (list, tuple)):
            if len(data) == 0:
                self._length = 0
                return
            if isinstance(data[0], dict):
                # Satır bazlı kayıt listesi: [{"name": "A", "age": 20}, {"name": "B", "age": 30}]
                keys = columns or list(data[0].keys())
                self._length = len(data)
                for k in keys:
                    col_vals = [row.get(k, None) for row in data]
                    self._columns[k] = np.asarray(col_vals)
            else:
                # 2D liste/matris
                arr = np.asarray(data)
                self._ingest_numpy_array(arr, columns)

        elif isinstance(data, np.ndarray):
            self._ingest_numpy_array(data, columns)
        else:
            raise TypeError(f"Unsupported data type for DataFrame: {type(data)}")

    def _ingest_numpy_array(self, arr: np.ndarray, columns: Optional[Sequence[str]] = None):
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.ndim != 2:
            raise ValueError(f"Array must be 1D or 2D, got shape {arr.shape}")

        rows, cols = arr.shape
        self._length = rows
        col_names = list(columns) if columns else [f"col_{i}" for i in range(cols)]
        if len(col_names) != cols:
            raise ValueError(f"Columns count ({len(col_names)}) does not match array width ({cols})")

        for i, cname in enumerate(col_names):
            self._columns[cname] = arr[:, i]

    @property
    def columns(self) -> List[str]:
        return list(self._columns.keys())

    @property
    def shape(self) -> Tuple[int, int]:
        return (self._length, len(self._columns))

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, item: Any) -> Any:
        if isinstance(item, str):
            if item not in self._columns:
                raise KeyError(f"Column '{item}' not found.")
            return self._columns[item]
        elif isinstance(item, (list, tuple)):
            # Çoklu sütun seçimi: df[["a", "b"]]
            sub_cols = {c: self._columns[c] for c in item if c in self._columns}
            df = DataFrame()
            df._columns = sub_cols
            df._length = self._length
            return df
        elif isinstance(item, ColumnExpr):
            # Filtreleme: df[col("age") > 20]
            mask = item.evaluate(self)
            return self.filter(mask)
        elif isinstance(item, (slice, int)):
            if isinstance(item, int):
                item = slice(item, item + 1)
            sub_cols = {c: arr[item] for c, arr in self._columns.items()}
            df = DataFrame()
            df._columns = sub_cols
            df._length = len(next(iter(sub_cols.values()))) if sub_cols else 0
            return df
        raise TypeError(f"Invalid index type for DataFrame: {type(item)}")

    def filter(self, condition: Union[ColumnExpr, np.ndarray, Sequence[bool]]) -> DataFrame:
        """Belirtilen koşula uyan satırları filtreler."""
        if isinstance(condition, ColumnExpr):
            mask = condition.evaluate(self)
        else:
            mask = np.asarray(condition, dtype=bool)

        new_cols = {c: arr[mask] for c, arr in self._columns.items()}
        new_len = int(np.sum(mask))
        df = DataFrame()
        df._columns = new_cols
        df._length = new_len
        return df

    def select(self, *col_names: Union[str, ColumnExpr, Sequence[str]]) -> DataFrame:
        """Yalnızca belirtilen sütunları veya türetilmiş ifadeleri seçer."""
        flat_cols: List[Union[str, ColumnExpr]] = []
        for c in col_names:
            if isinstance(c, (list, tuple)):
                flat_cols.extend(c)
            else:
                flat_cols.append(c)

        new_cols: Dict[str, np.ndarray] = {}
        for c in flat_cols:
            if isinstance(c, str):
                if c not in self._columns:
                    raise KeyError(f"Column '{c}' not found in DataFrame.")
                new_cols[c] = self._columns[c]
            elif isinstance(c, ColumnExpr):
                c_name = c._name or f"expr_{len(new_cols)}"
                new_cols[c_name] = c.evaluate(self)

        df = DataFrame()
        df._columns = new_cols
        df._length = self._length
        return df

    def with_column(self, name: str, expr_or_val: Union[ColumnExpr, np.ndarray, Sequence[Any], Any]) -> DataFrame:
        """Yeni bir sütun ekler veya mevcut sütunu günceller."""
        if isinstance(expr_or_val, ColumnExpr):
            val_arr = expr_or_val.evaluate(self)
        elif isinstance(expr_or_val, (list, tuple, np.ndarray)):
            val_arr = np.asarray(expr_or_val)
        else:
            # Skaler broadcasting
            val_arr = np.full(self._length, expr_or_val)

        if len(val_arr) != self._length:
            raise ValueError(f"New column length {len(val_arr)} does not match DataFrame length {self._length}")

        new_cols = dict(self._columns)
        new_cols[name] = val_arr
        df = DataFrame()
        df._columns = new_cols
        df._length = self._length
        return df

    def head(self, n: int = 5) -> DataFrame:
        """İlk n satırı döner."""
        return self[:n]

    def tail(self, n: int = 5) -> DataFrame:
        """Son n satırı döner."""
        start = max(0, self._length - n)
        return self[start:]

    def limit(self, n: int = 5) -> DataFrame:
        """İlk n satırı döner (head ile eşdeğer)."""
        return self.head(n)

    def sort(
        self,
        columns: Optional[Union[str, Sequence[str]]] = None,
        ascending: Union[bool, Sequence[bool]] = True,
        by: Optional[Union[str, Sequence[str]]] = None,
    ) -> DataFrame:
        """
        DataFrame'i tekli veya çoklu sütunlara göre sıralar.

        Parametreler:
        - columns / by: Sıralanacak sütun adı veya sütun adları listesi
        - ascending: Sıralama yönü (tekil bool veya her sütun için bool listesi)
        """
        cols = columns if columns is not None else by
        if cols is None:
            raise ValueError("sort() requires column name(s) to sort by.")

        cols_list = [cols] if isinstance(cols, str) else list(cols)
        if not cols_list:
            return self

        for c in cols_list:
            if c not in self._columns:
                raise KeyError(f"Sort column '{c}' not found in DataFrame. Available: {self.columns}")

        if isinstance(ascending, bool):
            asc_list = [ascending] * len(cols_list)
        else:
            asc_list = list(ascending)
            if len(asc_list) != len(cols_list):
                raise ValueError(
                    f"Length of 'ascending' ({len(asc_list)}) must match length of 'columns' ({len(cols_list)})"
                )

        if self._length <= 1:
            res = DataFrame()
            res._columns = {c: arr.copy() for c, arr in self._columns.items()}
            res._length = self._length
            return res

        # Kararlı (stable) çoklu sütun Timsort algoritması:
        # Sütunlar ters sırada kararlı şekilde sıralanır.
        def _sort_key(v: Any) -> Tuple[int, int, Any]:
            if v is None:
                return (0, 0, 0)
            if isinstance(v, (int, float, np.number)) and not isinstance(v, bool):
                try:
                    if math.isnan(float(v)):
                        return (0, 0, 0)
                except Exception:
                    pass
                return (1, 0, float(v))
            if isinstance(v, bool):
                return (1, 1, int(v))
            return (1, 2, str(v))

        indices = list(range(self._length))
        for col_name, asc in reversed(list(zip(cols_list, asc_list))):
            col_data = self._columns[col_name]
            indices.sort(key=lambda idx: _sort_key(col_data[idx]), reverse=not asc)

        np_indices = np.array(indices, dtype=np.int64)
        new_cols = {c: arr[np_indices] for c, arr in self._columns.items()}

        df = DataFrame()
        df._columns = new_cols
        df._length = self._length
        return df

    def groupby(self, by: Union[str, Sequence[str]], *args: str) -> GroupBy:
        """Gruplama nesnesi oluşturur."""
        if isinstance(by, str):
            group_keys = [by, *args] if args else [by]
        else:
            group_keys = list(by)
            if args:
                group_keys.extend(args)
        return GroupBy(self, group_keys)

    def join(
        self,
        other: DataFrame,
        on: Optional[Union[str, Sequence[str]]] = None,
        left_on: Optional[Union[str, Sequence[str]]] = None,
        right_on: Optional[Union[str, Sequence[str]]] = None,
        how: str = "inner",
        suffixes: Tuple[str, str] = ("", "_right"),
    ) -> DataFrame:
        """
        Hash Join destekli iki DataFrame'i birleştirir.

        Parametreler:
        - other: Sağdaki DataFrame
        - on: Her iki DataFrame'de de ortak olan birleşme sütun(lar)ı
        - left_on: Soldaki DataFrame'in birleşme sütun(lar)ı
        - right_on: Sağdaki DataFrame'in birleşme sütun(lar)ı
        - how: Birleşme tipi ('inner', 'left', 'outer' veya 'right')
        - suffixes: Sütun adı çakışmalarında eklenecek son ek çifti (varsayılan: ("", "_right"))
        """
        if not isinstance(other, DataFrame):
            raise TypeError(f"Expected other to be DataFrame, got {type(other)}")

        how = how.lower()
        if how not in ("inner", "left", "outer", "right"):
            raise ValueError(f"Unsupported join type: '{how}'. Supported: 'inner', 'left', 'outer', 'right'")

        if on is not None:
            left_keys = [on] if isinstance(on, str) else list(on)
            right_keys = [on] if isinstance(on, str) else list(on)
        elif left_on is not None and right_on is not None:
            left_keys = [left_on] if isinstance(left_on, str) else list(left_on)
            right_keys = [right_on] if isinstance(right_on, str) else list(right_on)
        else:
            common = [c for c in self.columns if c in other.columns]
            if not common:
                raise ValueError("No common columns found for join. Specify 'on' or ('left_on', 'right_on').")
            left_keys = common
            right_keys = common

        if len(left_keys) != len(right_keys):
            raise ValueError(f"left_on length ({len(left_keys)}) does not match right_on length ({len(right_keys)})")

        for k in left_keys:
            if k not in self._columns:
                raise KeyError(f"Join key '{k}' not found in left DataFrame. Available: {self.columns}")
        for k in right_keys:
            if k not in other._columns:
                raise KeyError(f"Join key '{k}' not found in right DataFrame. Available: {other.columns}")

        # Sağ DataFrame üzerinde Hash Table inşa et
        right_hash: Dict[Tuple[Any, ...], List[int]] = {}
        for r_idx in range(len(other)):
            key_items = []
            has_none = False
            for k in right_keys:
                v = other._columns[k][r_idx]
                if v is None or (isinstance(v, (float, np.floating)) and math.isnan(v)):
                    has_none = True
                    break
                if hasattr(v, "item") and not isinstance(v, (str, bytes)):
                    v = v.item()
                key_items.append(v)
            if not has_none:
                key_tuple = tuple(key_items)
                if key_tuple not in right_hash:
                    right_hash[key_tuple] = []
                right_hash[key_tuple].append(r_idx)

        left_indices: List[Optional[int]] = []
        right_indices: List[Optional[int]] = []
        matched_keys: List[Tuple[Any, ...]] = []
        right_matched: set[int] = set()

        # Sol DataFrame'i tara
        for l_idx in range(len(self)):
            key_items = []
            has_none = False
            for k in left_keys:
                v = self._columns[k][l_idx]
                if v is None or (isinstance(v, (float, np.floating)) and math.isnan(v)):
                    has_none = True
                    break
                if hasattr(v, "item") and not isinstance(v, (str, bytes)):
                    v = v.item()
                key_items.append(v)

            key_tuple = tuple(key_items) if not has_none else None

            if key_tuple is not None and key_tuple in right_hash:
                for r_idx in right_hash[key_tuple]:
                    left_indices.append(l_idx)
                    right_indices.append(r_idx)
                    matched_keys.append(key_tuple)
                    right_matched.add(r_idx)
            else:
                if how in ("left", "outer"):
                    left_indices.append(l_idx)
                    right_indices.append(None)
                    matched_keys.append(
                        key_tuple if key_tuple is not None else tuple(self._columns[k][l_idx] for k in left_keys)
                    )

        # Sağda eşleşmeyen satırlar (outer veya right join için)
        if how in ("outer", "right"):
            for r_idx in range(len(other)):
                if r_idx not in right_matched:
                    r_key = tuple(
                        other._columns[k][r_idx].item()
                        if hasattr(other._columns[k][r_idx], "item") and not isinstance(other._columns[k][r_idx], (str, bytes))
                        else other._columns[k][r_idx]
                        for k in right_keys
                    )
                    left_indices.append(None)
                    right_indices.append(r_idx)
                    matched_keys.append(r_key)

        num_out_rows = len(left_indices)
        out_cols: Dict[str, np.ndarray] = {}

        is_same_keys = (left_keys == right_keys)
        lsuffix, rsuffix = suffixes if suffixes else ("", "_right")
        if not rsuffix:
            rsuffix = "_right"

        # Soldaki sütunları yerleştir
        for c in self.columns:
            if is_same_keys and c in left_keys:
                target_name = c
            elif c in other.columns and lsuffix:
                target_name = f"{c}{lsuffix}"
            else:
                target_name = c

            raw_vals = []
            for i, l_idx in enumerate(left_indices):
                if l_idx is not None:
                    raw_vals.append(self._columns[c][l_idx])
                else:
                    if is_same_keys and c in left_keys:
                        k_idx = left_keys.index(c)
                        raw_vals.append(matched_keys[i][k_idx])
                    else:
                        raw_vals.append(None)
            out_cols[target_name] = _to_numpy_column(raw_vals)

        # Sağdaki sütunları yerleştir
        for c in other.columns:
            if is_same_keys and c in right_keys:
                continue

            target_name = c
            if target_name in out_cols or target_name in self.columns:
                target_name = f"{c}{rsuffix}"
                idx_suffix = 2
                while target_name in out_cols:
                    target_name = f"{c}{rsuffix}{idx_suffix}"
                    idx_suffix += 1

            raw_vals = []
            for i, r_idx in enumerate(right_indices):
                if r_idx is not None:
                    raw_vals.append(other._columns[c][r_idx])
                else:
                    raw_vals.append(None)
            out_cols[target_name] = _to_numpy_column(raw_vals)

        res_df = DataFrame()
        res_df._columns = out_cols
        res_df._length = num_out_rows
        return res_df

    def to_arrow_ipc(self, path: Optional[Union[str, os.PathLike]] = None) -> Optional[bytes]:
        """Zero-dependency Apache Arrow / Feather IPC binary biçimine serileştirir."""
        from synapse.core.arrow_ipc import write_arrow_ipc
        return write_arrow_ipc(self, path)

    @classmethod
    def from_arrow_ipc(
        cls,
        file_or_path: Union[str, bytes, bytearray, memoryview, os.PathLike, io.BytesIO],
        mmap: bool = False,
        use_mmap: Optional[bool] = None,
    ) -> DataFrame:
        """Apache Arrow / Feather IPC binary biçiminden DataFrame üretir (mmap destekli)."""
        from synapse.core.arrow_ipc import read_arrow_ipc
        return read_arrow_ipc(file_or_path, mmap=mmap, use_mmap=use_mmap)

    def close(self):
        """Açık olan bellek eşlemesi (mmap) veya dosya tanıtıcılarını kapatır."""
        if hasattr(self, "_mmap") and self._mmap is not None:
            self._columns.clear()
            try:
                self._mmap.close()
            except Exception:
                pass
            self._mmap = None
        if hasattr(self, "_file") and self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None

    def __enter__(self) -> DataFrame:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def mean(self) -> Dict[str, float]:
        """Tüm sayısal sütunların ortalamasını döner."""
        res = {}
        for c, arr in self._columns.items():
            if np.issubdtype(arr.dtype, np.number):
                res[c] = float(np.mean(arr))
        return res

    def sum(self) -> Dict[str, float]:
        """Tüm sayısal sütunların toplamını döner."""
        res = {}
        for c, arr in self._columns.items():
            if np.issubdtype(arr.dtype, np.number):
                res[c] = float(np.sum(arr))
        return res

    def to_tensor(self, columns: Optional[Sequence[str]] = None, requires_grad: bool = False, device: str = "cpu") -> Any:
        """
        Sıfır-Kopyalı / Doğrudan Bellek Eşlemeli Synapse Tensörü Üretir.
        Yalnızca sayısal sütunlar bir araya getirilerek modele hazır 2D matrise dönüştürülür.
        """
        from synapse.core.tensor import Tensor
        target_cols = list(columns) if columns else [c for c, arr in self._columns.items() if np.issubdtype(arr.dtype, np.number)]
        if not target_cols:
            raise ValueError("No numeric columns available in DataFrame to convert to Tensor.")

        # Sütunları 2D matris olarak istifle
        cols_data = [self._columns[c].astype(np.float64) for c in target_cols]
        matrix = np.column_stack(cols_data)
        return Tensor(matrix, requires_grad=requires_grad, device=device)

    def to_dict(self) -> Dict[str, list]:
        """Sözlük formatına dönüştürür."""
        return {c: arr.tolist() for c, arr in self._columns.items()}

    def to_csv(self, filepath: str, delimiter: str = ","):
        """Veriyi CSV dosyasına kaydeder."""
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=delimiter)
            writer.writerow(self.columns)
            for i in range(self._length):
                row = [self._columns[c][i] for c in self.columns]
                writer.writerow(row)

    def lazy(self) -> LazyFrame:
        """Tembel değerlendirme (Lazy Execution Plan) motoruna aktarır."""
        return LazyFrame(self)

    def __repr__(self) -> str:
        rows, cols = self.shape
        header = f"DataFrame(shape=({rows}, {cols}))\n"
        if rows == 0 or cols == 0:
            return header + "Empty"

        col_names = self.columns
        display_rows = min(rows, 8)
        table_lines = [" | ".join(col_names)]
        table_lines.append("-" * len(table_lines[0]))

        for i in range(display_rows):
            row_vals = [str(self._columns[c][i]) for c in col_names]
            table_lines.append(" | ".join(row_vals))

        if rows > display_rows:
            table_lines.append(f"... ({rows - display_rows} more rows)")

        return header + "\n".join(table_lines)


class GroupBy:
    """DataFrame Gruplama ve Kümeleme (Aggregation) Motoru."""
    def __init__(self, df: DataFrame, by: List[str]):
        self.df = df
        self.by = by
        for k in self.by:
            if k not in self.df._columns:
                raise KeyError(f"Group key '{k}' not found in DataFrame. Available: {self.df.columns}")

    def _compute_agg(self, col_data: np.ndarray, fn: Union[str, Callable[[Any], Any]]) -> Any:
        if callable(fn):
            return fn(col_data)
        if not isinstance(fn, str):
            raise TypeError(f"Invalid aggregation: {fn}")

        fn_name = fn.lower()
        if fn_name == "count":
            return sum(
                1 for x in col_data
                if x is not None and not (isinstance(x, (float, np.floating)) and math.isnan(x))
            )

        valid_vals = [
            x for x in col_data
            if x is not None and not (isinstance(x, (float, np.floating)) and math.isnan(x))
        ]
        if not valid_vals:
            return None

        if fn_name == "sum":
            arr = np.asarray(valid_vals)
            if np.issubdtype(arr.dtype, np.integer):
                return int(np.sum(arr))
            return float(np.sum(arr))
        elif fn_name == "mean":
            arr = np.asarray(valid_vals, dtype=float)
            return float(np.mean(arr))
        elif fn_name == "min":
            val = min(valid_vals)
            return val.item() if hasattr(val, "item") and not isinstance(val, (str, bytes)) else val
        elif fn_name == "max":
            val = max(valid_vals)
            return val.item() if hasattr(val, "item") and not isinstance(val, (str, bytes)) else val
        elif fn_name == "std":
            arr = np.asarray(valid_vals, dtype=float)
            return float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        elif fn_name == "var":
            arr = np.asarray(valid_vals, dtype=float)
            return float(np.var(arr, ddof=1)) if len(arr) > 1 else 0.0
        elif fn_name == "first":
            val = col_data[0]
            return val.item() if hasattr(val, "item") and not isinstance(val, (str, bytes)) else val
        elif fn_name == "last":
            val = col_data[-1]
            return val.item() if hasattr(val, "item") and not isinstance(val, (str, bytes)) else val
        else:
            raise ValueError(
                f"Unknown aggregation function: '{fn}'. "
                "Supported: sum, mean, min, max, count, std, var, first, last"
            )

    def agg(self, aggregations: Dict[str, Union[str, Sequence[str], Callable[[Any], Any]]]) -> DataFrame:
        """
        Gruplara göre kümeleme yapar:
        df.groupby("dept").agg({"salary": "mean", "bonus": "sum", "age": "max"})
        df.groupby(["dept", "role"]).agg({"salary": "mean", "id": "count"})
        df.groupby("dept").agg({"salary": ["mean", "max"]})
        """
        if self.df._length == 0:
            out_cols = {k: np.array([], dtype=object) for k in self.by}
            for target_col, agg_fn in aggregations.items():
                if isinstance(agg_fn, (list, tuple)):
                    for fn in agg_fn:
                        fn_name = fn if isinstance(fn, str) else getattr(fn, "__name__", "agg")
                        out_cols[f"{target_col}_{fn_name}"] = np.array([], dtype=object)
                else:
                    out_cols[target_col] = np.array([], dtype=object)
            return DataFrame(out_cols)

        # Benzersiz grupları ve satır indislerini topla
        groups: Dict[Any, List[int]] = {}
        for i in range(self.df._length):
            if len(self.by) == 1:
                val = self.df._columns[self.by[0]][i]
                if hasattr(val, "item") and not isinstance(val, (str, bytes)):
                    val = val.item()
                key = val
            else:
                key_list = []
                for k in self.by:
                    val = self.df._columns[k][i]
                    if hasattr(val, "item") and not isinstance(val, (str, bytes)):
                        val = val.item()
                    key_list.append(val)
                key = tuple(key_list)

            if key not in groups:
                groups[key] = []
            groups[key].append(i)

        try:
            sorted_keys = sorted(groups.keys())
            groups = {k: groups[k] for k in sorted_keys}
        except TypeError:
            pass

        out_data: Dict[str, list] = {k: [] for k in self.by}
        for target_col, agg_spec in aggregations.items():
            if target_col not in self.df._columns:
                raise KeyError(f"Aggregation target column '{target_col}' not found in DataFrame.")
            if isinstance(agg_spec, (list, tuple)):
                for fn in agg_spec:
                    fn_name = fn if isinstance(fn, str) else getattr(fn, "__name__", "agg")
                    out_data[f"{target_col}_{fn_name}"] = []
            else:
                out_data[target_col] = []

        for key, indices in groups.items():
            if len(self.by) == 1:
                out_data[self.by[0]].append(key)
            else:
                for k_idx, col_name in enumerate(self.by):
                    out_data[col_name].append(key[k_idx])

            for target_col, agg_spec in aggregations.items():
                col_data = self.df._columns[target_col][indices]
                if isinstance(agg_spec, (list, tuple)):
                    for fn in agg_spec:
                        val = self._compute_agg(col_data, fn)
                        fn_name = fn if isinstance(fn, str) else getattr(fn, "__name__", "agg")
                        out_data[f"{target_col}_{fn_name}"].append(val)
                else:
                    val = self._compute_agg(col_data, agg_spec)
                    out_data[target_col].append(val)

        return DataFrame(out_data)

    def mean(self) -> DataFrame:
        numeric_cols = [c for c, arr in self.df._columns.items() if c not in self.by and np.issubdtype(arr.dtype, np.number)]
        return self.agg({c: "mean" for c in numeric_cols})

    def sum(self) -> DataFrame:
        numeric_cols = [c for c, arr in self.df._columns.items() if c not in self.by and np.issubdtype(arr.dtype, np.number)]
        return self.agg({c: "sum" for c in numeric_cols})

    def min(self) -> DataFrame:
        target_cols = [c for c in self.df.columns if c not in self.by]
        return self.agg({c: "min" for c in target_cols})

    def max(self) -> DataFrame:
        target_cols = [c for c in self.df.columns if c not in self.by]
        return self.agg({c: "max" for c in target_cols})

    def count(self) -> DataFrame:
        target = [c for c in self.df.columns if c not in self.by][0]
        return self.agg({target: "count"})

    def std(self) -> DataFrame:
        numeric_cols = [c for c, arr in self.df._columns.items() if c not in self.by and np.issubdtype(arr.dtype, np.number)]
        return self.agg({c: "std" for c in numeric_cols})

    def var(self) -> DataFrame:
        numeric_cols = [c for c, arr in self.df._columns.items() if c not in self.by and np.issubdtype(arr.dtype, np.number)]
        return self.agg({c: "var" for c in numeric_cols})

    def first(self) -> DataFrame:
        target_cols = [c for c in self.df.columns if c not in self.by]
        return self.agg({c: "first" for c in target_cols})

    def last(self) -> DataFrame:
        target_cols = [c for c in self.df.columns if c not in self.by]
        return self.agg({c: "last" for c in target_cols})


class LazyFrame:
    """
    Tembel Değerlendirme (Lazy Evaluation) & Sorgu Optimizasyon Planlayıcısı.
    Filtreleri erkenden uygulayarak (Filter Pushdown) ve gereksiz sütunları budayarak (Projection Pruning)
    büyük veri kümelerinde minimum bellek ve maksimum işlem hızı sunar.
    """
    def __init__(self, source_df: DataFrame):
        self._source = source_df
        self._filters: List[ColumnExpr] = []
        self._projections: Optional[List[str]] = None
        self._sort_op: Optional[Tuple[Any, Any]] = None
        self._limit: Optional[int] = None

    def filter(self, condition: ColumnExpr) -> LazyFrame:
        self._filters.append(condition)
        return self

    def select(self, *columns: Union[str, Sequence[str]]) -> LazyFrame:
        flat = []
        for c in columns:
            if isinstance(c, (list, tuple)):
                flat.extend(c)
            else:
                flat.append(c)
        self._projections = flat
        return self

    def sort(
        self,
        columns: Optional[Union[str, Sequence[str]]] = None,
        ascending: Union[bool, Sequence[bool]] = True,
        by: Optional[Union[str, Sequence[str]]] = None,
    ) -> LazyFrame:
        cols = columns if columns is not None else by
        self._sort_op = (cols, ascending)
        return self

    def limit(self, n: int) -> LazyFrame:
        self._limit = n
        return self

    def collect(self) -> DataFrame:
        """Optimize edilmiş sorgu planını çalıştırıp somut DataFrame üretir."""
        df = self._source

        # 1. Optimize: Filter Pushdown (Satırları en başta filtrele)
        for f in self._filters:
            df = df.filter(f)

        # 2. Sort (varsa)
        if self._sort_op and self._sort_op[0] is not None:
            df = df.sort(columns=self._sort_op[0], ascending=self._sort_op[1])

        # 3. Limit (varsa)
        if self._limit is not None:
            df = df.head(self._limit)

        # 4. Projection Pruning (Gereksiz sütunları en son seç)
        if self._projections is not None:
            df = df.select(*self._projections)

        return df


def dataframe(data: Optional[Any] = None, columns: Optional[Sequence[str]] = None) -> DataFrame:
    """Yerleşik DataFrame üretici fonksiyonu."""
    return DataFrame(data=data, columns=columns)


def read_csv(filepath_or_buffer: Union[str, io.StringIO], delimiter: str = ",", has_header: bool = True) -> DataFrame:
    """
    Sıfır bağımlılıklı, otomatik tip çıkarımlı ultra hızlı CSV okuyucu.
    """
    if isinstance(filepath_or_buffer, str) and os.path.exists(filepath_or_buffer):
        with open(filepath_or_buffer, "r", encoding="utf-8") as f:
            content = f.read()
    elif isinstance(filepath_or_buffer, io.StringIO):
        content = filepath_or_buffer.getvalue()
    else:
        content = str(filepath_or_buffer)

    reader = csv.reader(io.StringIO(content.strip()), delimiter=delimiter)
    rows = list(reader)
    if not rows:
        return DataFrame()

    if has_header:
        headers = [h.strip() for h in rows[0]]
        data_rows = rows[1:]
    else:
        headers = [f"col_{i}" for i in range(len(rows[0]))]
        data_rows = rows

    cols_data: Dict[str, list] = {h: [] for h in headers}
    for r in data_rows:
        for i, h in enumerate(headers):
            val_str = r[i].strip() if i < len(r) else ""
            cols_data[h].append(_infer_and_cast_scalar(val_str))

    return DataFrame(cols_data)


def read_parquet(filepath: str) -> DataFrame:
    """
    Parquet formatı okuyucu.
    Sistemde PyArrow veya FastParquet kurulu ise doğrudan Arrow columnar buffer'larını okur;
    yoksa uyarı ile bilgi verir.
    """
    try:
        import pyarrow.parquet as pq
        table = pq.read_table(filepath)
        cols = {name: table[name].to_numpy() for name in table.column_names}
        return DataFrame(cols)
    except ImportError:
        raise NotImplementedError(
            "Parquet support requires 'pyarrow'. Install it via 'pip install pyarrow' "
            "or use standard native CSV/JSON formats with read_csv()."
        )


def _infer_and_cast_scalar(val: str) -> Any:
    """Otomatik tip tespiti (int, float, bool, str)."""
    if val == "":
        return None
    lower = val.lower()
    if lower in ("true", "yes", "1"):
        if lower in ("true", "yes"):
            return True
    elif lower in ("false", "no", "0"):
        if lower in ("false", "no"):
            return False

    # Int denemesi
    try:
        return int(val)
    except ValueError:
        pass

    # Float denemesi
    try:
        return float(val)
    except ValueError:
        pass

    return val


def _to_numpy_column(vals: list) -> np.ndarray:
    """Veri listesini uygun NumPy dizisine dönüştürür, None değerlerini korur."""
    has_none = any(v is None for v in vals)
    if not has_none:
        try:
            return np.asarray(vals)
        except Exception:
            return np.asarray(vals, dtype=object)
    return np.asarray(vals, dtype=object)


def read_arrow_ipc(
    file_or_path: Union[str, bytes, bytearray, memoryview, os.PathLike, io.BytesIO],
    mmap: bool = False,
    use_mmap: Optional[bool] = None,
) -> DataFrame:
    """Apache Arrow IPC / Feather ikili dosyasını veya baytlarını okur."""
    from synapse.core.arrow_ipc import read_arrow_ipc as _read_arrow
    return _read_arrow(file_or_path, mmap=mmap, use_mmap=use_mmap)


def write_arrow_ipc(
    df: DataFrame,
    file_or_path: Optional[Union[str, os.PathLike, io.BytesIO]] = None,
) -> Optional[bytes]:
    """DataFrame'i Apache Arrow IPC / Feather ikili biçimine serileştirir."""
    from synapse.core.arrow_ipc import write_arrow_ipc as _write_arrow
    return _write_arrow(df, file_or_path)
