"""Feature creation utilities for SentinelNet Phase 5."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import FeatureEngineeringConfig

EPSILON = 1e-9

REQUIRED_COLUMNS = {
    "source_file",
    "label",
    "label_binary",
    "label_multiclass",
    "destination_port",
    "flow_duration",
    "flow_bytes_per_s",
    "flow_packets_per_s",
    "total_fwd_packets",
    "total_backward_packets",
    "total_length_of_fwd_packets",
    "total_length_of_bwd_packets",
    "max_packet_length",
    "min_packet_length",
    "packet_length_mean",
    "packet_length_std",
    "flow_iat_mean",
    "flow_iat_std",
    "fwd_header_length",
    "bwd_header_length",
    "act_data_pkt_fwd",
    "syn_flag_count",
    "rst_flag_count",
    "fin_flag_count",
    "ack_flag_count",
    "psh_flag_count",
    "idle_mean",
    "active_mean",
    "avg_fwd_segment_size",
    "avg_bwd_segment_size",
}


@dataclass(slots=True)
class RollingWindowState:
    """State carried across chunks for source-file rolling calculations."""

    numeric_history: dict[str, dict[str, list[float]]] = field(default_factory=lambda: defaultdict(dict))
    port_history: dict[str, list[int]] = field(default_factory=dict)


def safe_divide(numerator: pd.Series | np.ndarray, denominator: pd.Series | np.ndarray) -> np.ndarray:
    """Safely divide two aligned arrays and replace invalid results with zero."""
    numerator_array = np.asarray(numerator, dtype=np.float64)
    denominator_array = np.asarray(denominator, dtype=np.float64)
    result = np.divide(
        numerator_array,
        denominator_array,
        out=np.zeros_like(numerator_array, dtype=np.float64),
        where=np.abs(denominator_array) > EPSILON,
    )
    result[~np.isfinite(result)] = 0.0
    return result


def validate_required_columns(columns: list[str]) -> None:
    """Ensure the input dataset contains all columns required for feature engineering."""
    missing = sorted(REQUIRED_COLUMNS.difference(columns))
    if missing:
        raise ValueError(f"Phase 5 requires the following missing columns: {missing}")


def add_statistical_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create flow-based statistical features from the cleaned CICIDS2017 signals."""
    engineered = frame.copy()

    engineered["total_packets"] = engineered["total_fwd_packets"] + engineered["total_backward_packets"]
    engineered["total_bytes"] = engineered["total_length_of_fwd_packets"] + engineered["total_length_of_bwd_packets"]
    engineered["bytes_per_packet"] = safe_divide(engineered["total_bytes"], engineered["total_packets"])
    engineered["fwd_bwd_packet_ratio"] = safe_divide(engineered["total_fwd_packets"], engineered["total_backward_packets"])
    engineered["fwd_bwd_byte_ratio"] = safe_divide(
        engineered["total_length_of_fwd_packets"],
        engineered["total_length_of_bwd_packets"],
    )
    engineered["packet_length_range"] = engineered["max_packet_length"] - engineered["min_packet_length"]
    engineered["flow_iat_cv"] = safe_divide(engineered["flow_iat_std"], engineered["flow_iat_mean"])
    engineered["packet_length_cv"] = safe_divide(engineered["packet_length_std"], engineered["packet_length_mean"])
    engineered["header_payload_ratio"] = safe_divide(
        engineered["fwd_header_length"] + engineered["bwd_header_length"],
        engineered["total_bytes"],
    )
    engineered["forward_payload_efficiency"] = safe_divide(
        engineered["act_data_pkt_fwd"],
        engineered["total_fwd_packets"],
    )
    engineered["idle_active_ratio"] = safe_divide(engineered["idle_mean"], engineered["active_mean"])
    engineered["burstiness_score"] = safe_divide(
        engineered["active_mean"] - engineered["idle_mean"],
        engineered["active_mean"] + engineered["idle_mean"],
    )
    engineered["segment_size_asymmetry"] = safe_divide(
        np.abs(engineered["avg_fwd_segment_size"] - engineered["avg_bwd_segment_size"]),
        engineered["avg_fwd_segment_size"] + engineered["avg_bwd_segment_size"],
    )
    engineered["ack_push_ratio"] = safe_divide(engineered["ack_flag_count"], engineered["psh_flag_count"] + 1.0)
    engineered["payload_density"] = safe_divide(engineered["total_bytes"], engineered["flow_duration"])
    return engineered


def _rolling_unique_counts(values: np.ndarray, window_size: int, history: list[int] | None = None) -> np.ndarray:
    """Compute rolling unique counts with support for history from previous chunks."""
    history_values = list(history or [])
    counter: Counter[int] = Counter(history_values)
    current_window: deque[int] = deque(history_values, maxlen=window_size)
    result = np.empty(len(values), dtype=np.float64)

    for index, raw_value in enumerate(values):
        value = int(raw_value)
        if len(current_window) == window_size:
            dropped = current_window.popleft()
            counter[dropped] -= 1
            if counter[dropped] <= 0:
                del counter[dropped]

        current_window.append(value)
        counter[value] += 1
        result[index] = float(len(counter))

    return result


def _compute_stateful_rolling_mean(values: np.ndarray, window_size: int, history: list[float] | None = None) -> tuple[np.ndarray, list[float]]:
    """Compute a rolling mean while carrying limited history across chunks."""
    buffer_values = list(history or [])
    current_window: deque[float] = deque(buffer_values, maxlen=window_size)
    rolling_sum = float(sum(current_window))
    result = np.empty(len(values), dtype=np.float64)

    for index, raw_value in enumerate(values):
        value = float(raw_value)
        if len(current_window) == window_size:
            rolling_sum -= current_window.popleft()
        current_window.append(value)
        rolling_sum += value
        result[index] = rolling_sum / len(current_window)

    return result, list(current_window)


def _compute_stateful_rolling_sum(values: np.ndarray, window_size: int, history: list[float] | None = None) -> tuple[np.ndarray, list[float]]:
    """Compute a rolling sum while carrying limited history across chunks."""
    buffer_values = list(history or [])
    current_window: deque[float] = deque(buffer_values, maxlen=window_size)
    rolling_sum = float(sum(current_window))
    result = np.empty(len(values), dtype=np.float64)

    for index, raw_value in enumerate(values):
        value = float(raw_value)
        if len(current_window) == window_size:
            rolling_sum -= current_window.popleft()
        current_window.append(value)
        rolling_sum += value
        result[index] = rolling_sum

    return result, list(current_window)


def add_time_and_domain_features(
    frame: pd.DataFrame,
    config: FeatureEngineeringConfig,
    state: RollingWindowState,
) -> pd.DataFrame:
    """Create rolling time-based and domain-specific features in source-file order."""
    engineered = frame.copy()
    max_window = max(config.rolling_windows)

    engineered["syn_ratio"] = safe_divide(engineered["syn_flag_count"], engineered["total_packets"])
    engineered["connection_failure_score"] = safe_divide(
        engineered["rst_flag_count"] + engineered["fin_flag_count"],
        engineered["total_packets"],
    )
    engineered["connection_reset_ratio"] = safe_divide(engineered["rst_flag_count"], engineered["total_packets"])

    for source_file, group in engineered.groupby(config.source_file_column, sort=False):
        group_indices = group.index
        source_history = state.numeric_history[source_file]
        port_history = state.port_history.get(source_file, [])

        flow_bytes = group["flow_bytes_per_s"].to_numpy(dtype=np.float64, copy=False)
        flow_packets = group["flow_packets_per_s"].to_numpy(dtype=np.float64, copy=False)
        short_flow_flags = (group["flow_duration"].to_numpy(dtype=np.float64, copy=False) <= config.short_flow_duration_threshold).astype(np.float64)
        syn_flags = group["syn_flag_count"].to_numpy(dtype=np.float64, copy=False)
        rst_flags = group["rst_flag_count"].to_numpy(dtype=np.float64, copy=False)
        ports = group["destination_port"].to_numpy(dtype=np.int64, copy=False)

        for window_size in config.rolling_windows:
            flow_bytes_mean, source_history[f"flow_bytes_per_s_w{window_size}"] = _compute_stateful_rolling_mean(
                flow_bytes,
                window_size,
                source_history.get(f"flow_bytes_per_s_w{window_size}"),
            )
            flow_packets_mean, source_history[f"flow_packets_per_s_w{window_size}"] = _compute_stateful_rolling_mean(
                flow_packets,
                window_size,
                source_history.get(f"flow_packets_per_s_w{window_size}"),
            )
            short_flow_fraction, source_history[f"short_flow_flags_w{window_size}"] = _compute_stateful_rolling_mean(
                short_flow_flags,
                window_size,
                source_history.get(f"short_flow_flags_w{window_size}"),
            )
            syn_sum, source_history[f"syn_flag_count_w{window_size}"] = _compute_stateful_rolling_sum(
                syn_flags,
                window_size,
                source_history.get(f"syn_flag_count_w{window_size}"),
            )
            rst_sum, source_history[f"rst_flag_count_w{window_size}"] = _compute_stateful_rolling_sum(
                rst_flags,
                window_size,
                source_history.get(f"rst_flag_count_w{window_size}"),
            )

            engineered.loc[group_indices, f"rolling_flow_bytes_mean_w{window_size}"] = flow_bytes_mean
            engineered.loc[group_indices, f"rolling_flow_packets_mean_w{window_size}"] = flow_packets_mean
            engineered.loc[group_indices, f"rolling_short_flow_fraction_w{window_size}"] = short_flow_fraction
            engineered.loc[group_indices, f"rolling_syn_sum_w{window_size}"] = syn_sum
            engineered.loc[group_indices, f"rolling_rst_sum_w{window_size}"] = rst_sum

        unique_port_counts = _rolling_unique_counts(ports, max_window, history=port_history)
        state.port_history[source_file] = list(ports[-(max_window - 1) :]) if max_window > 1 else []

        engineered.loc[group_indices, f"rolling_unique_destination_ports_w{max_window}"] = unique_port_counts
        engineered.loc[group_indices, f"rolling_syn_ratio_w{max_window}"] = safe_divide(
            engineered.loc[group_indices, f"rolling_syn_sum_w{max_window}"],
            max_window,
        )
        engineered.loc[group_indices, f"rolling_connection_failure_ratio_w{max_window}"] = safe_divide(
            engineered.loc[group_indices, f"rolling_rst_sum_w{max_window}"],
            max_window,
        )
        engineered.loc[group_indices, f"port_scan_pattern_score_w{max_window}"] = (
            engineered.loc[group_indices, f"rolling_unique_destination_ports_w{max_window}"]
            * engineered.loc[group_indices, f"rolling_short_flow_fraction_w{max_window}"]
            * (engineered.loc[group_indices, f"rolling_syn_ratio_w{max_window}"] + 1.0)
        )

    return engineered


def engineer_feature_chunk(
    chunk: pd.DataFrame,
    config: FeatureEngineeringConfig,
    state: RollingWindowState,
) -> pd.DataFrame:
    """Apply all Phase 5 feature engineering steps to a chunk."""
    engineered = add_statistical_features(chunk)
    engineered = add_time_and_domain_features(engineered, config, state)
    engineered.replace([np.inf, -np.inf], 0.0, inplace=True)
    return engineered

