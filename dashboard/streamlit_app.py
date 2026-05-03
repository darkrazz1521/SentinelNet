"""Streamlit dashboard for SentinelNet Phase 13."""

from __future__ import annotations

import os
import sys

# Add project root (sent/) to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import streamlit as st
except ImportError as exc:  # pragma: no cover - exercised only when the app is launched without deps
    raise RuntimeError("Install Streamlit dependencies before running the dashboard.") from exc

from dashboard.config import DashboardConfig
from dashboard.dashboard_data import DashboardSnapshot, build_dashboard_snapshot


def _inject_theme() -> None:
    """Apply a custom cyber-operations visual theme to the Streamlit page."""
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(14, 116, 144, 0.18), transparent 28%),
                radial-gradient(circle at top right, rgba(245, 158, 11, 0.14), transparent 24%),
                linear-gradient(180deg, #07131a 0%, #0a1c24 45%, #10242d 100%);
            color: #e5eef3;
        }
        .block-container {
            padding-top: 1.8rem;
            padding-bottom: 2.5rem;
        }
        .sentinel-hero {
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 20px;
            padding: 1.4rem 1.6rem;
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.82), rgba(17, 94, 89, 0.42));
            box-shadow: 0 18px 40px rgba(2, 8, 23, 0.25);
            margin-bottom: 1rem;
        }
        .sentinel-kicker {
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: #f59e0b;
            font-size: 0.78rem;
            font-weight: 700;
        }
        .sentinel-title {
            font-size: 2.2rem;
            font-weight: 800;
            margin-top: 0.2rem;
            margin-bottom: 0.4rem;
            color: #f8fafc;
        }
        .sentinel-subtitle {
            font-size: 1rem;
            line-height: 1.6;
            color: #cbd5e1;
            max-width: 58rem;
        }
        div[data-testid="metric-container"] {
            background: rgba(15, 23, 42, 0.55);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 16px;
            padding: 0.6rem 0.8rem;
        }
        .sentinel-panel-note {
            color: #94a3b8;
            font-size: 0.92rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _resolve_config_path() -> str | None:
    """Resolve the dashboard configuration path from the environment if provided."""
    return os.getenv("SENTINELNET_DASHBOARD_CONFIG")


@st.cache_data(ttl=60, show_spinner=False)
def _load_snapshot(
    config_path: str | None,
    recent_rows: int,
    explanation_top_n: int,
    multiclass_top_k: int,
) -> tuple[DashboardConfig, DashboardSnapshot]:
    """Cache the artifact-backed dashboard snapshot for lightweight refreshes."""
    config = DashboardConfig.from_json(config_path=config_path)
    snapshot = build_dashboard_snapshot(
        config,
        recent_rows=recent_rows,
        explanation_top_n=explanation_top_n,
        multiclass_top_k=multiclass_top_k,
    )
    return config, snapshot


def _render_recent_tables(snapshot: DashboardSnapshot) -> None:
    """Render live event tables for operators."""
    recent_alerts, recent_predictions = st.tabs(["Recent Alerts", "Recent Predictions"])

    with recent_alerts:
        st.dataframe(
            snapshot.recent_alerts,
            use_container_width=True,
            hide_index=True,
            column_config={
                "binary_attack_probability": st.column_config.NumberColumn("Binary Attack Prob.", format="%.4f"),
                "multiclass_confidence": st.column_config.NumberColumn("Class Confidence", format="%.4f"),
                "risk_score": st.column_config.NumberColumn("Risk Score", format="%.2f"),
            },
        )

    with recent_predictions:
        st.dataframe(
            snapshot.recent_predictions,
            use_container_width=True,
            hide_index=True,
            column_config={
                "binary_attack_probability": st.column_config.NumberColumn("Binary Attack Prob.", format="%.4f"),
                "multiclass_confidence": st.column_config.NumberColumn("Class Confidence", format="%.4f"),
                "risk_score": st.column_config.NumberColumn("Risk Score", format="%.2f"),
            },
        )


def _render_confusion_matrix(title: str, matrix: object) -> None:
    """Render a styled confusion matrix table."""
    st.subheader(title)
    st.dataframe(
        matrix.style.background_gradient(cmap="YlOrBr"),
        use_container_width=True,
    )


def _render_explainability_panel(title: str, table: object, *, label_column: str) -> None:
    """Render a top-contribution chart plus the underlying table."""
    st.subheader(title)
    if table.empty:
        st.info("No explainability artifacts were available for this view.")
        return
    chart = table[[label_column, "mean_abs_contribution"]].set_index(label_column)
    st.bar_chart(chart, use_container_width=True)
    st.dataframe(table, use_container_width=True, hide_index=True)


def main() -> None:
    """Render the Phase 13 SentinelNet operations dashboard."""
    st.set_page_config(page_title="SentinelNet v2", page_icon=":shield:", layout="wide", initial_sidebar_state="expanded")
    _inject_theme()

    st.markdown(
        """
        <div class="sentinel-hero">
            <div class="sentinel-kicker">SentinelNet v2</div>
            <div class="sentinel-title">Real-Time Intrusion Operations Dashboard</div>
            <div class="sentinel-subtitle">
                Live attack scoring, ensemble-driven detection, and explainability views backed by the
                persisted Phase 10 to Phase 12 artifacts from the CICIDS2017 pipeline.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Dashboard Controls")
        if st.button("Refresh Artifacts", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        recent_rows = int(st.slider("Recent Rows", min_value=20, max_value=250, value=80, step=10))
        explanation_top_n = int(st.slider("Top Explanations", min_value=5, max_value=20, value=10, step=1))
        multiclass_top_k = int(st.slider("Multiclass Labels", min_value=4, max_value=12, value=8, step=1))
        st.caption("Optional config override")
        config_path = st.text_input("Config Path", value=_resolve_config_path() or "config/dashboard_config.json")

    try:
        active_config, snapshot = _load_snapshot(config_path, recent_rows, explanation_top_n, multiclass_top_k)
    except Exception as exc:  # pragma: no cover - exercised only in the live app
        st.error(f"Unable to load Phase 13 artifacts: {exc}")
        st.stop()

    metrics_row = st.columns(6)
    metrics_row[0].metric("Rows Streamed", f"{snapshot.overview_metrics['rows_streamed']:,}")
    metrics_row[1].metric("Alerts Emitted", f"{snapshot.overview_metrics['alert_rows_written']:,}")
    metrics_row[2].metric("Attack Alerts", f"{snapshot.overview_metrics['attack_alerts']:,}")
    metrics_row[3].metric("Avg Risk Score", f"{snapshot.overview_metrics['average_risk_score']:.2f}")
    metrics_row[4].metric("Throughput", f"{snapshot.overview_metrics['throughput_rows_per_second']:.2f} rows/s")
    metrics_row[5].metric("Avg Batch Latency", f"{snapshot.overview_metrics['average_batch_latency_ms']:.2f} ms")

    st.caption(
        "Selected live variants: "
        f"binary=`{snapshot.overview_metrics['selected_binary_variant']}` | "
        f"multiclass=`{snapshot.overview_metrics['selected_multiclass_variant']}`"
    )

    overview_tab, analytics_tab, performance_tab, explainability_tab = st.tabs(
        ["Live Operations", "Attack Analytics", "Performance", "Explainability"]
    )

    with overview_tab:
        left_col, right_col = st.columns([1.4, 1.0])
        with left_col:
            st.subheader("Alert Timeline")
            if snapshot.alert_timeline.empty:
                st.info("No alert timeline is available yet.")
            else:
                timeline_chart = snapshot.alert_timeline.set_index("timestamp")
                st.line_chart(timeline_chart, use_container_width=True)
            st.markdown(
                '<div class="sentinel-panel-note">Timeline is aggregated from the persisted Phase 12 alert log at one-minute resolution.</div>',
                unsafe_allow_html=True,
            )
        with right_col:
            st.subheader("Alert Level Distribution")
            alert_level_chart = snapshot.alert_level_counts.set_index("alert_level")
            st.bar_chart(alert_level_chart, use_container_width=True)
            st.dataframe(snapshot.alert_level_counts, use_container_width=True, hide_index=True)

        _render_recent_tables(snapshot)

    with analytics_tab:
        attack_col, confusion_col = st.columns([1.0, 1.1])
        with attack_col:
            st.subheader("Attack Distribution")
            if snapshot.attack_distribution.empty:
                st.info("No attack alerts were available for distribution analysis.")
            else:
                attack_chart = snapshot.attack_distribution.set_index("attack_type")
                st.bar_chart(attack_chart, use_container_width=True)
                st.dataframe(snapshot.attack_distribution, use_container_width=True, hide_index=True)

        with confusion_col:
            _render_confusion_matrix("Binary Confusion Matrix", snapshot.binary_confusion_matrix)
            st.caption("Binary confusion matrix is computed from the full Phase 11 replay output.")

        _render_confusion_matrix("Multiclass Confusion Matrix", snapshot.multiclass_confusion_matrix)
        st.caption("The multiclass confusion matrix focuses on BENIGN plus the most frequent true labels, with the remainder collapsed into OTHER.")

    with performance_tab:
        roc_col, metrics_col = st.columns([1.0, 1.1])
        with roc_col:
            st.subheader("Binary ROC Curve")
            st.metric("Binary ROC-AUC", f"{snapshot.binary_roc_auc:.6f}")
            roc_chart = snapshot.binary_roc_curve.set_index("false_positive_rate")[["true_positive_rate"]]
            st.line_chart(roc_chart, use_container_width=True)
            st.caption("The ROC curve is downsampled for dashboard rendering, but the AUC is computed from the full replay.")

        with metrics_col:
            st.subheader("Phase 9 Ensemble Metrics")
            st.dataframe(
                snapshot.phase9_metrics,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "accuracy": st.column_config.NumberColumn(format="%.6f"),
                    "precision": st.column_config.NumberColumn(format="%.6f"),
                    "recall": st.column_config.NumberColumn(format="%.6f"),
                    "f1_score": st.column_config.NumberColumn(format="%.6f"),
                    "roc_auc": st.column_config.NumberColumn(format="%.6f"),
                },
            )

    with explainability_tab:
        raw_feature_tab, ensemble_tab = st.tabs(["Raw Feature SHAP Insights", "Ensemble Component Insights"])

        with raw_feature_tab:
            binary_col, multiclass_col = st.columns(2)
            with binary_col:
                _render_explainability_panel(
                    "Binary Feature Importance",
                    snapshot.binary_shap_summary,
                    label_column="feature_name",
                )
            with multiclass_col:
                _render_explainability_panel(
                    "Multiclass Feature Importance",
                    snapshot.multiclass_shap_summary,
                    label_column="feature_name",
                )

        with ensemble_tab:
            binary_col, multiclass_col = st.columns(2)
            with binary_col:
                _render_explainability_panel(
                    "Binary Ensemble Components",
                    snapshot.binary_ensemble_summary,
                    label_column="feature_name",
                )
            with multiclass_col:
                _render_explainability_panel(
                    "Multiclass Ensemble Components",
                    snapshot.multiclass_ensemble_summary,
                    label_column="feature_name",
                )

    with st.expander("Artifact Paths"):
        st.json(active_config.to_dict())


if __name__ == "__main__":  # pragma: no cover - exercised only when the app is launched manually
    main()
