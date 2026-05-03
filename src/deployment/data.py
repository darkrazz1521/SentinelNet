"""Streaming data iterators for SentinelNet Phase 11."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.models.deep_learning.data import load_feature_names, load_label_metadata

from .config import StreamingConfig


@dataclass(slots=True)
class StreamBatch:
    """Single ordered batch of stream-ready rows."""

    feature_frame: pd.DataFrame
    original_indices: np.ndarray
    source_files: np.ndarray
    true_binary_labels: np.ndarray
    true_multiclass_labels: np.ndarray


@dataclass(slots=True)
class StreamingMetadata:
    """Shared metadata for the Phase 11 simulator."""

    feature_names: list[str]
    inverse_multiclass_mapping: dict[int, str]
    selected_indices: np.ndarray | None
    total_rows: int


def load_streaming_metadata(config: StreamingConfig) -> StreamingMetadata:
    """Load feature and label metadata plus the requested split indices."""
    feature_names = load_feature_names(config.feature_manifest_path)
    label_metadata = load_label_metadata(config.label_mapping_path)

    if config.stream_split == "test":
        selected_indices = np.load(config.test_indices_path).astype(np.int64, copy=False)
    elif config.stream_split == "train":
        selected_indices = np.load(config.train_indices_path).astype(np.int64, copy=False)
    elif config.stream_split == "full":
        selected_indices = None
    else:
        raise ValueError("stream_split must be one of: 'train', 'test', or 'full'.")

    if selected_indices is not None:
        selected_indices = np.sort(selected_indices).astype(np.int64, copy=False)
        if config.max_rows is not None:
            selected_indices = selected_indices[: config.max_rows]
        total_rows = int(len(selected_indices))
    else:
        total_rows = int(config.max_rows) if config.max_rows is not None else -1

    return StreamingMetadata(
        feature_names=feature_names,
        inverse_multiclass_mapping=label_metadata["inverse_multiclass_mapping"],
        selected_indices=selected_indices,
        total_rows=total_rows,
    )


def _yield_selected_chunks(
    config: StreamingConfig,
    metadata: StreamingMetadata,
) -> tuple[pd.DataFrame, int]:
    """Yield selected chunks plus the number of retained rows in each chunk."""
    usecols = [
        *metadata.feature_names,
        config.source_file_column,
        config.binary_target_column,
        config.multiclass_target_column,
    ]
    selected_indices = metadata.selected_indices
    chunk_iterator = pd.read_csv(
        config.input_data_path,
        usecols=usecols,
        chunksize=config.chunk_size,
        low_memory=False,
    )

    global_start = 0
    selected_pointer = 0
    yielded_rows = 0
    for chunk in chunk_iterator:
        chunk_length = len(chunk)
        global_end = global_start + chunk_length

        if selected_indices is None:
            selected_chunk = chunk.copy()
            selected_chunk["_original_index"] = np.arange(global_start, global_end, dtype=np.int64)
            if config.max_rows is not None:
                remaining = config.max_rows - yielded_rows
                if remaining <= 0:
                    break
                selected_chunk = selected_chunk.iloc[:remaining].copy()
        else:
            left = selected_pointer
            right = int(np.searchsorted(selected_indices, global_end, side="left"))
            if right <= left:
                global_start = global_end
                continue

            retained_indices = selected_indices[left:right]
            local_positions = retained_indices - global_start
            selected_chunk = chunk.iloc[local_positions].copy()
            selected_chunk["_original_index"] = retained_indices
            selected_pointer = right

        yielded_count = len(selected_chunk)
        if yielded_count > 0:
            yielded_rows += yielded_count
            yield selected_chunk, yielded_count
            if config.max_rows is not None and yielded_rows >= config.max_rows:
                break

        global_start = global_end
        if selected_indices is not None and selected_pointer >= len(selected_indices):
            break


def iter_stream_batches(
    config: StreamingConfig,
    metadata: StreamingMetadata,
) -> tuple[StreamBatch, int]:
    """Yield ordered stream batches sized for incremental inference."""
    buffer: list[pd.DataFrame] = []
    buffered_rows = 0
    yielded_rows = 0

    for selected_chunk, _ in _yield_selected_chunks(config, metadata):
        buffer.append(selected_chunk)
        buffered_rows += len(selected_chunk)

        while buffered_rows >= config.inference_batch_size:
            combined = pd.concat(buffer, ignore_index=True)
            batch_frame = combined.iloc[: config.inference_batch_size].copy()
            remainder = combined.iloc[config.inference_batch_size :].copy()
            buffer = [remainder] if not remainder.empty else []
            buffered_rows = len(remainder)

            yielded_rows += len(batch_frame)
            yield _frame_to_batch(batch_frame, config, metadata), yielded_rows

    if buffer:
        combined = pd.concat(buffer, ignore_index=True)
        if not combined.empty:
            yielded_rows += len(combined)
            yield _frame_to_batch(combined, config, metadata), yielded_rows


def _frame_to_batch(
    frame: pd.DataFrame,
    config: StreamingConfig,
    metadata: StreamingMetadata,
) -> StreamBatch:
    """Convert a retained frame slice into a typed stream batch."""
    feature_frame = frame.loc[:, metadata.feature_names].astype(np.float32)
    return StreamBatch(
        feature_frame=feature_frame,
        original_indices=frame["_original_index"].to_numpy(dtype=np.int64, copy=False),
        source_files=frame[config.source_file_column].astype(str).to_numpy(copy=False),
        true_binary_labels=frame[config.binary_target_column].to_numpy(dtype=np.int32, copy=False),
        true_multiclass_labels=frame[config.multiclass_target_column].to_numpy(dtype=np.int32, copy=False),
    )
