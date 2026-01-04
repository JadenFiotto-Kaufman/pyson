"""
Custom serializers for third-party types.

This module provides serializers for popular libraries like pandas.
Import this module to register these serializers.

Usage:
    import seri.custom  # Registers pandas DataFrame serializer
    from seri import serialize, deserialize

    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    payload = serialize(df)
    result = deserialize(payload.model_dump())
"""

from __future__ import annotations

from typing import Literal, Any, TYPE_CHECKING
import base64

from seri.types import SerializedType, ReferenceId

if TYPE_CHECKING:
    from seri.serialize import SerializationContext
else:
    SerializationContext = Any


# =============================================================================
# Pandas Serializer
# =============================================================================

import pandas as pd


class DataFrameType(SerializedType):
    """
    Serializer for pandas DataFrame objects.

    Stores the DataFrame as:
    - data: List of rows (each row is a list of values)
    - columns: Column names
    - index: Index values
    - dtypes: Column dtype names (for accurate reconstruction)
    """

    type: Literal["dataframe"] = "dataframe"
    data: ReferenceId  # Nested list of rows
    columns: ReferenceId
    index: ReferenceId
    dtypes: ReferenceId  # dict of column -> dtype string

    @classmethod
    def serialize(cls, obj: pd.DataFrame, context: SerializationContext):
        # Use .tolist() to convert numpy types to Python natives
        # Keep references alive to prevent id() reuse during serialization
        data = obj.values.tolist()
        columns = obj.columns.tolist()
        index = obj.index.tolist()
        dtypes = {col: str(dt) for col, dt in obj.dtypes.items()}

        return cls(
            data=context.serialize(data),
            columns=context.serialize(columns),
            index=context.serialize(index),
            dtypes=context.serialize(dtypes),
        )

    def deserialize(self, referenceID: ReferenceId, context: SerializationContext):
        # Deserialize components
        data = context.deserialize(self.data)
        columns = context.deserialize(self.columns)
        index = context.deserialize(self.index)
        dtypes = context.deserialize(self.dtypes)

        # Reconstruct DataFrame
        df = pd.DataFrame(data, columns=columns, index=index)

        # Restore dtypes where possible
        for col, dtype_str in dtypes.items():
            try:
                df[col] = df[col].astype(dtype_str)
            except (ValueError, TypeError):
                # Some dtypes can't be restored (e.g., custom dtypes)
                pass

        context.memo[referenceID] = df
        return df


class SeriesType(SerializedType):
    """
    Serializer for pandas Series objects.

    Stores the Series as:
    - data: List of values
    - index: Index values
    - name: Series name
    - dtype: Data type string
    """

    type: Literal["series"] = "series"
    data: ReferenceId
    index: ReferenceId
    name: ReferenceId
    dtype: str

    @classmethod
    def serialize(cls, obj: pd.Series, context: SerializationContext):
        # Use .tolist() to convert numpy types to Python natives
        # Keep references alive to prevent id() reuse
        data = obj.tolist()
        index = obj.index.tolist()
        name = obj.name
        return cls(
            data=context.serialize(data),
            index=context.serialize(index),
            name=context.serialize(name),
            dtype=str(obj.dtype),
        )

    def deserialize(self, referenceID: ReferenceId, context: SerializationContext):
        data = context.deserialize(self.data)
        index = context.deserialize(self.index)
        name = context.deserialize(self.name)

        # Reconstruct Series
        series = pd.Series(data, index=index, name=name)

        # Restore dtype if possible
        try:
            series = series.astype(self.dtype)
        except (ValueError, TypeError):
            pass

        context.memo[referenceID] = series
        return series


# =============================================================================
# NumPy Serializer
# =============================================================================

import numpy as np


class NdarrayType(SerializedType):
    """
    Serializer for numpy ndarray objects.

    Stores the array as:
    - data: Nested list of values (via tolist())
    - dtype: Data type string
    - shape: Tuple of dimensions (for verification/empty arrays)
    """

    type: Literal["ndarray"] = "ndarray"
    data: ReferenceId
    dtype: str
    shape: ReferenceId

    @classmethod
    def serialize(cls, obj: np.ndarray, context: SerializationContext):
        # Use .tolist() to convert to Python natives
        # Keep references alive to prevent id() reuse
        data = obj.tolist()
        shape = list(obj.shape)

        return cls(
            data=context.serialize(data),
            dtype=str(obj.dtype),
            shape=context.serialize(shape),
        )

    def deserialize(self, referenceID: ReferenceId, context: SerializationContext):
        data = context.deserialize(self.data)
        shape = context.deserialize(self.shape)

        # Reconstruct array
        arr = np.array(data, dtype=self.dtype)

        # Handle empty arrays that may need reshaping
        if arr.shape != tuple(shape):
            arr = arr.reshape(shape)

        context.memo[referenceID] = arr
        return arr


# =============================================================================
# PyTorch Tensor Serializer
# =============================================================================


import torch


class TensorType(SerializedType):
    """
    Serializer for PyTorch tensor objects.

    Handles special cases:
    - Sparse tensors: preserved in COO format (memory efficient)
    - Quantized tensors: preserved with quantization parameters
    - GPU/MPS tensors: moved to CPU first
    - bfloat16: stored as int16 bit pattern

    Stores as base64-encoded bytes for efficient transport.
    """

    type: Literal["tensor"] = "tensor"
    data: str  # base64-encoded bytes
    dtype: str  # numpy dtype string
    shape: list[int]
    torch_dtype: str | None = None  # Original torch dtype if different (e.g., bfloat16)
    quantization: dict | None = None  # {scale, zero_point, qtype} for quantized
    sparse: dict | None = (
        None  # {indices, indices_dtype, indices_shape, dense_shape} for sparse
    )

    @classmethod
    def serialize(cls, obj: torch.Tensor, context: SerializationContext):
        t = obj.detach()

        quantization_info = None
        sparse_info = None
        original_dtype = None

        # Reject nested tensors
        if hasattr(t, "is_nested") and t.is_nested:
            raise ValueError(
                "Nested tensors cannot be serialized. "
                "Convert to a list of regular tensors first."
            )

        # Handle sparse tensors - preserve sparsity
        if hasattr(t, "is_sparse") and t.is_sparse:
            t = t.coalesce()
            indices = t._indices().cpu().numpy()
            values = t._values().cpu().numpy()

            indices_bytes = indices.tobytes()
            indices_data = base64.b64encode(indices_bytes).decode("ascii")

            sparse_info = {
                "indices": indices_data,
                "indices_dtype": str(indices.dtype),
                "indices_shape": list(indices.shape),
                "dense_shape": list(t.shape),
            }
            np_array = values

        # Handle quantized tensors - preserve quantization
        elif hasattr(t, "is_quantized") and t.is_quantized:
            quantization_info = {
                "scale": float(t.q_scale()),
                "zero_point": int(t.q_zero_point()),
                "qtype": str(t.dtype).replace("torch.", ""),
            }
            np_array = t.int_repr().cpu().numpy()

        else:
            # Handle bfloat16: view as int16 (same bit pattern)
            original_dtype = str(t.dtype).replace("torch.", "")
            if t.dtype == torch.bfloat16:
                np_array = t.view(torch.int16).cpu().numpy()
            else:
                np_array = t.cpu().numpy()
                original_dtype = None

        # Base64 encode the raw bytes
        raw_bytes = np_array.tobytes()
        b64_data = base64.b64encode(raw_bytes).decode("ascii")

        return cls(
            data=b64_data,
            dtype=str(np_array.dtype),
            shape=list(np_array.shape),
            torch_dtype=original_dtype,
            quantization=quantization_info,
            sparse=sparse_info,
        )

    def deserialize(self, referenceID: ReferenceId, context: SerializationContext):
        import torch

        # Decode base64
        raw_bytes = base64.b64decode(self.data)

        # Reconstruct numpy array
        dtype = np.dtype(self.dtype)
        shape = tuple(self.shape)
        np_array = np.frombuffer(raw_bytes, dtype=dtype).reshape(shape)

        # Convert to torch tensor
        values_tensor = torch.from_numpy(np_array.copy())

        # Restore original torch dtype if different (e.g., bfloat16)
        if self.torch_dtype is not None:
            dtype_map = {"bfloat16": torch.bfloat16}
            target_dtype = dtype_map.get(self.torch_dtype)
            if target_dtype is not None:
                values_tensor = values_tensor.view(target_dtype)

        # Reconstruct sparse tensor
        if self.sparse is not None:
            s = self.sparse
            indices_bytes = base64.b64decode(s["indices"])
            indices_dtype = np.dtype(s["indices_dtype"])
            indices_shape = tuple(s["indices_shape"])
            indices_np = np.frombuffer(indices_bytes, dtype=indices_dtype).reshape(
                indices_shape
            )
            indices_tensor = torch.from_numpy(indices_np.copy())

            dense_shape = tuple(s["dense_shape"])
            tensor = torch.sparse_coo_tensor(indices_tensor, values_tensor, dense_shape)
            result = tensor.coalesce()
            context.memo[referenceID] = result
            return result

        # Re-quantize if original was quantized
        if self.quantization is not None:
            q = self.quantization
            qtype_map = {
                "qint8": torch.qint8,
                "quint8": torch.quint8,
                "qint32": torch.qint32,
            }
            qtype = qtype_map.get(q["qtype"], torch.qint8)
            values_tensor = torch._make_per_tensor_quantized_tensor(
                values_tensor, scale=q["scale"], zero_point=q["zero_point"]
            )

        context.memo[referenceID] = values_tensor
        return values_tensor


# Register the serializers
from seri.serialize import register_serializer

register_serializer(pd.Series, SeriesType)
register_serializer(pd.DataFrame, DataFrameType)
register_serializer(np.ndarray, NdarrayType)
register_serializer(torch.Tensor, TensorType)
