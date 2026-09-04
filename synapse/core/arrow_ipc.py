from __future__ import annotations
import io
import math
import mmap as _mmap_module
import os
import struct
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from synapse.core.dataframe import DataFrame

# =============================================================================
# Arrow IPC Constants & Type Signatures
# =============================================================================
ARROW_MAGIC = b"ARROW1"
ARROW_VERSION = 1

TYPE_INT64 = 1
TYPE_FLOAT64 = 2
TYPE_UTF8 = 3
TYPE_INT32 = 4
TYPE_FLOAT32 = 5
TYPE_BOOL = 6

TYPE_NAME_TO_ID = {
    "int64": TYPE_INT64,
    "float64": TYPE_FLOAT64,
    "utf8": TYPE_UTF8,
    "string": TYPE_UTF8,
    "str": TYPE_UTF8,
    "int32": TYPE_INT32,
    "float32": TYPE_FLOAT32,
    "bool": TYPE_BOOL,
}

TYPE_ID_TO_NAME = {
    TYPE_INT64: "int64",
    TYPE_FLOAT64: "float64",
    TYPE_UTF8: "utf8",
    TYPE_INT32: "int32",
    TYPE_FLOAT32: "float32",
    TYPE_BOOL: "bool",
}


def _pad_to_8(buf: bytearray) -> int:
    """Buffer sonunu 8-byte hizalama için sıfırlarla doldurur; eklenen bayt sayısını döner."""
    remainder = len(buf) % 8
    if remainder != 0:
        pad_len = 8 - remainder
        buf.extend(b"\x00" * pad_len)
        return pad_len
    return 0


def _pad_len_8(n: int) -> int:
    """Verilen uzunluğu 8-byte katına tamamlamak için gereken ek bayt sayısı."""
    remainder = n % 8
    return (8 - remainder) if remainder != 0 else 0


def _infer_column_type(arr: np.ndarray) -> Tuple[int, str]:
    """Sütunun NumPy veri tipinden Arrow tür kodunu çıkarır."""
    if np.issubdtype(arr.dtype, np.bool_):
        return TYPE_BOOL, "bool"
    elif np.issubdtype(arr.dtype, np.integer):
        return TYPE_INT64, "int64"
    elif np.issubdtype(arr.dtype, np.floating):
        return TYPE_FLOAT64, "float64"
    elif np.issubdtype(arr.dtype, (np.str_, np.bytes_)):
        return TYPE_UTF8, "utf8"
    else:
        # Object array: Null olmayan elemanların türünü incele
        non_nulls = [
            x for x in arr
            if x is not None and not (isinstance(x, (float, np.floating)) and math.isnan(x))
        ]
        if not non_nulls:
            return TYPE_UTF8, "utf8"
        if all(isinstance(x, bool) for x in non_nulls):
            return TYPE_BOOL, "bool"
        if all(isinstance(x, (int, np.integer)) and not isinstance(x, bool) for x in non_nulls):
            return TYPE_INT64, "int64"
        if all(isinstance(x, (int, float, np.number)) and not isinstance(x, bool) for x in non_nulls):
            return TYPE_FLOAT64, "float64"
        return TYPE_UTF8, "utf8"


def _build_validity_bitmap(vals: Sequence[Any], num_rows: int) -> Tuple[bytes, int]:
    """
    Arrow standart geçerlilik (null) bitmaskesi üretir.
    LSB-first: i. satırın biti = (byte[i >> 3] >> (i & 7)) & 1.
    1 = geçerli (non-null), 0 = null / None.
    """
    null_count = 0
    num_bytes = (num_rows + 7) // 8
    bitmap = bytearray(b"\x00" * num_bytes)

    for i in range(num_rows):
        v = vals[i]
        is_null = (v is None) or (isinstance(v, (float, np.floating)) and math.isnan(v))
        if is_null:
            null_count += 1
        else:
            bitmap[i >> 3] |= (1 << (i & 7))

    return bytes(bitmap), null_count


def _read_validity_bitmap(bitmap_bytes: Union[bytes, memoryview], num_rows: int) -> List[bool]:
    """Geçerlilik bitmaskesini boolean listesine açar."""
    validity = [True] * num_rows
    if not bitmap_bytes:
        return validity
    for i in range(num_rows):
        byte_val = bitmap_bytes[i >> 3]
        bit_val = (byte_val >> (i & 7)) & 1
        validity[i] = (bit_val == 1)
    return validity


def _pack_column_descriptor(
    name: str,
    type_id: int,
    null_count: int,
    bitmap_offset: int,
    bitmap_length: int,
    offsets_offset: int,
    offsets_length: int,
    data_offset: int,
    data_length: int,
) -> bytes:
    name_bytes = name.encode("utf-8")
    header = struct.pack("<H", len(name_bytes)) + name_bytes
    fields = struct.pack(
        "<Hqqqqqqq",
        type_id,
        null_count,
        bitmap_offset,
        bitmap_length,
        offsets_offset,
        offsets_length,
        data_offset,
        data_length,
    )
    return header + fields


def write_arrow_ipc(
    df: DataFrame,
    file_or_path: Optional[Union[str, os.PathLike, io.BytesIO]] = None,
) -> Optional[bytes]:
    """
    DataFrame'i sıfır bağımlılıklı Apache Arrow / Feather IPC RecordBatch ikili (binary) biçiminde serileştirir.

    Biçim Yapısı:
    - 6 bayt: b"ARROW1"
    - 2 bayt: uint16 versiyon (1)
    - 8 bayt: int64 satır sayısı (num_rows)
    - 4 bayt: int32 sütun sayısı (num_cols)
    - 4 bayt: int32 sütun tanımlayıcıları boyutu (descriptors_length)
    - Sütun Tanımlayıcıları (Ad, Tip, Null sayısı, Offset ve Boyutlar)
    - 8-byte hizalama dolgusu
    - Veri Tamponları (Geçerlilik bitmaskesi, String Offsetleri, Veri dizileri - her biri 8-byte hizalı)
    - Dipnot: b"ARROW1\\x00\\x00"
    """
    num_rows = len(df)
    num_cols = len(df.columns)

    # 1. Aşama: Her sütunun tamponlarını hazırla
    col_prepared: List[Tuple[str, int, int, bytes, bytes, bytes]] = []

    for cname in df.columns:
        arr = df._columns[cname]
        type_id, type_str = _infer_column_type(arr)
        bitmap_bytes, null_count = _build_validity_bitmap(arr, num_rows)

        # Null yoksa boş bırakılabilir
        active_bitmap_bytes = bitmap_bytes if null_count > 0 else b""

        offsets_bytes = b""
        data_bytes = b""

        if type_id == TYPE_INT64:
            if null_count > 0:
                clean_vals = [0 if (v is None or (isinstance(v, (float, np.floating)) and math.isnan(v))) else int(v) for v in arr]
                data_bytes = np.array(clean_vals, dtype=np.int64).tobytes()
            else:
                data_bytes = np.asarray(arr, dtype=np.int64).tobytes()

        elif type_id == TYPE_FLOAT64:
            if null_count > 0:
                clean_vals = [float("nan") if (v is None or (isinstance(v, (float, np.floating)) and math.isnan(v))) else float(v) for v in arr]
                data_bytes = np.array(clean_vals, dtype=np.float64).tobytes()
            else:
                data_bytes = np.asarray(arr, dtype=np.float64).tobytes()

        elif type_id == TYPE_BOOL:
            if null_count > 0:
                clean_vals = [0 if (v is None or (isinstance(v, (float, np.floating)) and math.isnan(v))) else int(bool(v)) for v in arr]
                data_bytes = np.array(clean_vals, dtype=np.uint8).tobytes()
            else:
                data_bytes = np.asarray(arr, dtype=np.uint8).tobytes()

        elif type_id == TYPE_UTF8:
            encoded_list: List[bytes] = []
            offsets: List[int] = [0]
            for v in arr:
                if v is None or (isinstance(v, (float, np.floating)) and math.isnan(v)):
                    encoded_list.append(b"")
                    offsets.append(offsets[-1])
                else:
                    bs = str(v).encode("utf-8")
                    encoded_list.append(bs)
                    offsets.append(offsets[-1] + len(bs))
            offsets_bytes = np.array(offsets, dtype=np.int64).tobytes()
            data_bytes = b"".join(encoded_list)

        else:
            clean_vals = [0 if v is None else int(v) for v in arr]
            data_bytes = np.array(clean_vals, dtype=np.int64).tobytes()
            type_id = TYPE_INT64

        col_prepared.append((cname, type_id, null_count, active_bitmap_bytes, offsets_bytes, data_bytes))

    # 2. Aşama: Tampon offsetlerini hesapla
    # Sabit başlık prefix'i = 6(magic) + 2(ver) + 8(rows) + 4(cols) + 4(desclen) = 24 bayt
    dummy_descriptors = b"".join(
        _pack_column_descriptor(c[0], c[1], c[2], 0, 0, 0, 0, 0, 0)
        for c in col_prepared
    )
    descriptors_length = len(dummy_descriptors)
    desc_pad = _pad_len_8(descriptors_length)
    body_start_offset = 24 + descriptors_length + desc_pad

    current_offset = body_start_offset
    col_offsets: List[Tuple[int, int, int, int, int, int]] = []

    for c in col_prepared:
        _, _, _, b_bytes, o_bytes, d_bytes = c
        bo = bl = oo = ol = do = dl = 0

        if b_bytes:
            bo = current_offset
            bl = len(b_bytes)
            current_offset += bl + _pad_len_8(bl)

        if o_bytes:
            oo = current_offset
            ol = len(o_bytes)
            current_offset += ol + _pad_len_8(ol)

        if d_bytes:
            do = current_offset
            dl = len(d_bytes)
            current_offset += dl + _pad_len_8(dl)

        col_offsets.append((bo, bl, oo, ol, do, dl))

    # 3. Aşama: İkili veri akışını inşa et
    out = bytearray()
    out.extend(ARROW_MAGIC)
    out.extend(struct.pack("<H", ARROW_VERSION))
    out.extend(struct.pack("<q", num_rows))
    out.extend(struct.pack("<i", num_cols))
    out.extend(struct.pack("<i", descriptors_length))

    # Gerçek tanımlayıcıları ekle
    for c, off in zip(col_prepared, col_offsets):
        desc = _pack_column_descriptor(
            name=c[0],
            type_id=c[1],
            null_count=c[2],
            bitmap_offset=off[0],
            bitmap_length=off[1],
            offsets_offset=off[2],
            offsets_length=off[3],
            data_offset=off[4],
            data_length=off[5],
        )
        out.extend(desc)

    _pad_to_8(out)

    # Veri tamponlarını ekle
    for c in col_prepared:
        _, _, _, b_bytes, o_bytes, d_bytes = c
        if b_bytes:
            out.extend(b_bytes)
            _pad_to_8(out)
        if o_bytes:
            out.extend(o_bytes)
            _pad_to_8(out)
        if d_bytes:
            out.extend(d_bytes)
            _pad_to_8(out)

    # Kapanış dipnotu (Magic + 2 padding byte)
    out.extend(ARROW_MAGIC)
    out.extend(b"\x00\x00")

    result_bytes = bytes(out)

    # Hedefe yaz
    if file_or_path is not None:
        if isinstance(file_or_path, (str, os.PathLike)):
            with open(file_or_path, "wb") as f:
                f.write(result_bytes)
        elif hasattr(file_or_path, "write"):
            file_or_path.write(result_bytes)
        else:
            raise TypeError(f"Unsupported file_or_path type: {type(file_or_path)}")

    return result_bytes


def read_arrow_ipc(
    file_or_path: Union[str, bytes, bytearray, memoryview, os.PathLike, io.BytesIO],
    mmap: bool = False,
    use_mmap: Optional[bool] = None,
) -> DataFrame:
    """
    Apache Arrow / Feather IPC RecordBatch ikili biçiminden DataFrame okur.

    Parametreler:
    - file_or_path: Dosya yolu, bytes, BytesIO veya bellek görünümü.
    - mmap / use_mmap: True ise mmap.mmap kullanarak büyük veri kümelerinde sıfır kopyalı (zero-copy) okuma yapar.
    """
    from synapse.core.dataframe import DataFrame

    enable_mmap = mmap or (use_mmap is True)

    underlying_file = None
    mmap_obj = None

    if isinstance(file_or_path, (str, os.PathLike)):
        filepath = str(file_or_path)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Arrow IPC file not found: {filepath}")

        if enable_mmap:
            underlying_file = open(filepath, "rb")
            file_size = os.path.getsize(filepath)
            if file_size == 0:
                underlying_file.close()
                return DataFrame()
            mmap_obj = _mmap_module.mmap(underlying_file.fileno(), 0, access=_mmap_module.ACCESS_READ)
            buf = memoryview(mmap_obj)
        else:
            with open(filepath, "rb") as f:
                raw_bytes = f.read()
            buf = memoryview(raw_bytes)
    elif isinstance(file_or_path, (bytes, bytearray)):
        buf = memoryview(file_or_path)
    elif isinstance(file_or_path, memoryview):
        buf = file_or_path
    elif hasattr(file_or_path, "read"):
        raw_bytes = file_or_path.read()
        buf = memoryview(raw_bytes)
    else:
        raise TypeError(f"Invalid input type for read_arrow_ipc: {type(file_or_path)}")

    if len(buf) < 24:
        if mmap_obj is not None:
            mmap_obj.close()
        if underlying_file is not None:
            underlying_file.close()
        raise ValueError("File is too small to be a valid Arrow IPC stream.")

    # 1. Sihirli bayt kontrolü
    if buf[:6].tobytes() != ARROW_MAGIC:
        if mmap_obj is not None:
            mmap_obj.close()
        if underlying_file is not None:
            underlying_file.close()
        raise ValueError(f"Invalid Arrow IPC format: expected magic {ARROW_MAGIC}, got {buf[:6].tobytes()}")

    version = struct.unpack_from("<H", buf, 6)[0]
    num_rows = struct.unpack_from("<q", buf, 8)[0]
    num_cols = struct.unpack_from("<i", buf, 16)[0]
    descriptors_length = struct.unpack_from("<i", buf, 20)[0]

    # 2. Sütun tanımlayıcılarını ayrıştır
    desc_start = 24
    pos = desc_start

    columns_data: Dict[str, Any] = {}

    for _ in range(num_cols):
        name_len = struct.unpack_from("<H", buf, pos)[0]
        pos += 2
        col_name = buf[pos : pos + name_len].tobytes().decode("utf-8")
        pos += name_len

        type_id, null_count, bo, bl, oo, ol, do, dl = struct.unpack_from("<Hqqqqqqq", buf, pos)
        pos += struct.calcsize("<Hqqqqqqq")

        # Geçerlilik bitmaskesi
        validity = _read_validity_bitmap(buf[bo : bo + bl], num_rows) if bl > 0 else [True] * num_rows

        if type_id == TYPE_INT64:
            if null_count > 0:
                raw_arr = np.frombuffer(buf[do : do + dl], dtype=np.int64)
                col_vals = [raw_arr[i].item() if validity[i] else None for i in range(num_rows)]
                columns_data[col_name] = np.array(col_vals, dtype=object)
            else:
                if enable_mmap and mmap_obj is not None:
                    columns_data[col_name] = np.frombuffer(mmap_obj, dtype=np.int64, count=num_rows, offset=do)
                else:
                    columns_data[col_name] = np.frombuffer(buf[do : do + dl], dtype=np.int64).copy()

        elif type_id == TYPE_FLOAT64:
            if null_count > 0:
                raw_arr = np.frombuffer(buf[do : do + dl], dtype=np.float64)
                col_vals = [raw_arr[i].item() if validity[i] else None for i in range(num_rows)]
                columns_data[col_name] = np.array(col_vals, dtype=object)
            else:
                if enable_mmap and mmap_obj is not None:
                    columns_data[col_name] = np.frombuffer(mmap_obj, dtype=np.float64, count=num_rows, offset=do)
                else:
                    columns_data[col_name] = np.frombuffer(buf[do : do + dl], dtype=np.float64).copy()

        elif type_id == TYPE_BOOL:
            raw_arr = np.frombuffer(buf[do : do + dl], dtype=np.uint8)
            if null_count > 0:
                col_vals = [bool(raw_arr[i]) if validity[i] else None for i in range(num_rows)]
                columns_data[col_name] = np.array(col_vals, dtype=object)
            else:
                columns_data[col_name] = (raw_arr != 0)

        elif type_id == TYPE_UTF8:
            offsets = np.frombuffer(buf[oo : oo + ol], dtype=np.int64)
            data_slice = buf[do : do + dl]
            str_list = []
            for i in range(num_rows):
                if null_count > 0 and not validity[i]:
                    str_list.append(None)
                else:
                    s_bytes = data_slice[offsets[i] : offsets[i + 1]].tobytes()
                    str_list.append(s_bytes.decode("utf-8"))
            columns_data[col_name] = np.array(str_list, dtype=object if null_count > 0 else str)

        else:
            columns_data[col_name] = np.frombuffer(buf[do : do + dl], dtype=np.int64).copy()

    df = DataFrame(columns_data)
    if enable_mmap and mmap_obj is not None:
        df._mmap = mmap_obj
        df._file = underlying_file

    return df
