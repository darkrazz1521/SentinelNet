# SentinelNet v2

SentinelNet v2 is a production-oriented AI-powered Network Intrusion Detection System (NIDS) built around the CICIDS2017 dataset. The repository is being implemented phase-by-phase; the current codebase completes Phase 1 multi-file ingestion, Phase 2 data cleaning, Phase 3 label engineering, Phase 4 preprocessing, Phase 5 feature engineering, Phase 6 classical ML training, Phase 7 deep-learning training, Phase 8 anomaly detection, Phase 9 ensemble learning, Phase 10 explainability, Phase 11 real-time simulation, Phase 12 alert generation, Phase 13 dashboarding, Phase 14 API deployment, Phase 15 performance optimization, and Phase 16 advanced response automation with executable pipelines, reports, logging, tests, a Streamlit operations UI, a FastAPI service layer, tuned deployment recommendations, and continuous-learning artifacts.

## Current Status

Implemented:

- automatic discovery of all CSV files in `data/raw/`
- delimiter and encoding detection with resilient fallbacks
- schema profiling, normalization, and alignment logging
- chunked ingestion for large files
- dtype optimization and source lineage tracking
- export of `data/interim/combined.csv`
- Phase 2 cleaning with duplicate removal, infinity handling, missing-value remediation, label normalization, and validation
- Phase 3 label handling with binary and multiclass targets plus persistent mapping artifacts
- Phase 4 preprocessing with stratified splitting, scaling, categorical encoding, and task-specific imbalance handling
- Phase 5 feature engineering with flow statistics, rolling behavioral features, domain-driven attack signals, and train-only feature selection
- Phase 6 classical ML training and evaluation for Random Forest, Logistic Regression, XGBoost, and LightGBM
- Phase 7 deep-learning training and evaluation for DNN, LSTM sequence modeling, and autoencoder-based anomaly detection
- Phase 8 anomaly detection with Isolation Forest, One-Class SVM, Local Outlier Factor, and autoencoder reconstruction scoring
- Phase 9 ensemble learning with binary and multiclass soft voting, weighted scoring, and stacking across ML, deep-learning, and anomaly subsystems
- Phase 10 explainability with native feature importance, Monte Carlo SHAP-style raw-feature attributions, and component-level ensemble explanations
- Phase 11 real-time simulation with stateful streaming inference, persisted event outputs, and measured batch latency/throughput
- Phase 12 risk scoring and alert generation with severity-aware alert levels, recommended actions, and persisted alert logs
- Phase 13 Streamlit dashboard for live predictions, alert analytics, confusion matrices, ROC visualization, and explainability views
- Phase 14 FastAPI endpoints for prediction, stream replay, alert retrieval, and operational metrics
- Phase 15 performance optimization with predictor benchmarking, API profiling, metrics-cache measurement, and optimized deployment configs
- Phase 16 advanced response automation with attack-family classification, zero-day candidate detection, simulated auto-block actions, feature-drift tracking, and continuous-learning queues
- production React SOC frontend in `sentinelnet-frontend/` with Vite, Tailwind CSS, Axios, React Router, Recharts, collapsible navigation, live stream filters, alert filters, health status, and a `/predict` inference lab
- report generation and structured logging for all completed phases
- unit tests for ingestion, schema alignment, dtype optimization, label normalization, cleaning behavior, feature engineering, classical ML, deep learning, anomaly detection, ensemble learning, explainability, streaming simulation, alert generation, dashboard aggregation, API endpoints, performance optimization, and advanced response automation

## Run Phase 1

```bash
python -m src.data_pipeline --config config/ingestion_config.json
```

## Run Phase 2

```bash
python -m src.data_pipeline.run_phase2 --config config/cleaning_config.json
```

Optional overrides:

```bash
python -m src.data_pipeline.run_phase2 --config config/cleaning_config.json --chunk-size 50000 --log-level DEBUG
```

Artifacts:

- Phase 1 combined dataset: `data/interim/combined.csv`
- Phase 1 report: `data/interim/combined_report.json`
- Phase 2 cleaned dataset: `data/interim/cleaned.csv`
- Phase 2 report: `data/interim/cleaned_report.json`
- Phase 3 labeled dataset: `data/processed/labeled_dataset.csv`
- Phase 3 mappings: `data/processed/label_mappings.json`
- Phase 3 report: `data/processed/label_report.json`
- Phase 4 preprocessing artifacts: `data/processed/preprocessed/`
- Phase 5 engineered artifacts: `data/processed/feature_engineered/`
- Phase 6 model artifacts: `models/saved_models/phase6_ml/`
- Phase 7 deep-learning artifacts: `models/saved_models/phase7_deep_learning/`
- Phase 8 anomaly-detection artifacts: `models/saved_models/phase8_anomaly_detection/`
- Phase 9 ensemble artifacts: `models/saved_models/phase9_ensemble/`
- Phase 10 explainability artifacts: `models/saved_models/phase10_explainability/`
- Phase 11 streaming artifacts: `data/streaming/`
- Phase 12 alerting artifacts: `data/streaming/`
- Phase 13 dashboard app: `dashboard/streamlit_app.py`
- Phase 14 API app: `api/fastapi_app.py`
- Phase 15 performance artifacts: `models/saved_models/phase15_performance/`
- Phase 16 advanced response artifacts: `data/streaming/phase16_advanced/`
- React SOC frontend: `sentinelnet-frontend/`
- logs: `logs/phase1_ingestion.log`, `logs/phase2_cleaning.log`, `logs/phase3_label_handling.log`, `logs/phase4_preprocessing.log`, `logs/phase5_feature_engineering.log`, `logs/phase6_ml_training.log`, `logs/phase7_deep_learning.log`, `logs/phase8_anomaly_detection.log`, `logs/phase9_ensemble.log`, `logs/phase10_explainability.log`, `logs/phase11_streaming.log`, `logs/phase12_alerting.log`, `logs/phase14_api.log`, `logs/phase15_performance.log`, `logs/phase16_advanced_features.log`

## Run Phase 3

```bash
python -m src.data_pipeline.run_phase3 --config config/label_config.json
```

## Run Phase 4

```bash
python -m src.data_pipeline.run_phase4 --config config/preprocessing_config.json
```

## Run Phase 5

```bash
python -m src.feature_engineering.run_phase5 --config config/feature_engineering_config.json
```

## Run Phase 6

```bash
python -m src.models.ml.run_phase6 --config config/ml_training_config.json
```

## Run Phase 7

```bash
python -m src.models.deep_learning.run_phase7 --config config/deep_learning_config.json
```

Phase 7 uses the Phase 5 selected-feature dataset plus the persisted train/test indices from earlier phases. The DNN and autoencoder operate on the scaled tabular feature vectors, while the LSTM builds fixed-length sequences in preserved row order within each `source_file`.

## Run Phase 8

```bash
python -m src.anomaly_detection.run_phase8 --config config/anomaly_detection_config.json
```

Phase 8 reuses the Phase 5 selected features and the persisted split indices, trains the anomaly detectors on benign-only traffic, calibrates thresholds on held-out benign validation traffic, and scores the test split for anomaly detection.

## Run Phase 9

```bash
python -m src.models.ensemble.run_phase9 --config config/ensemble_config.json
```

Phase 9 consumes the persisted outputs from Phases 6, 7, and 8, scores a shared calibration/test subset, and trains three ensemble variants per task:

- binary soft voting, weighted scoring, and stacking using classical ML, DNN/LSTM, and anomaly detectors
- multiclass soft voting and weighted scoring across classical ML plus DNN/LSTM outputs
- multiclass stacking that augments multiclass base probabilities with Phase 8 anomaly scores as extra meta-features

Primary artifacts:

- report: `models/saved_models/phase9_ensemble/ensemble_report.json`
- metrics summary: `models/saved_models/phase9_ensemble/metrics_summary.csv`
- binary stacker: `models/saved_models/phase9_ensemble/binary/stacking.joblib`
- multiclass stacker: `models/saved_models/phase9_ensemble/multiclass/stacking.joblib`

## Run Phase 10

```bash
python -m src.explainability.run_phase10 --config config/explainability_config.json
```

Phase 10 consumes the persisted outputs from Phases 6 through 9 and produces two explanation layers:

- raw-feature explainability for the strongest Phase 6 binary and multiclass classifiers using Monte Carlo SHAP-style attributions over the selected CICIDS2017 features
- component-level explainability for every Phase 9 ensemble variant, including soft voting, weighted scoring, and stacking

Primary artifacts:

- report: `models/saved_models/phase10_explainability/explainability_report.json`
- artifact summary: `models/saved_models/phase10_explainability/artifact_summary.csv`
- Phase 6 native feature importance: `models/saved_models/phase10_explainability/phase6/native_importance/`
- Phase 6 SHAP-style outputs: `models/saved_models/phase10_explainability/phase6/shap_values/`
- Phase 9 ensemble explanations: `models/saved_models/phase10_explainability/phase9/`

The default real-data run selects the best binary Phase 6 model by `roc_auc` and the best multiclass Phase 6 model by `f1_score`. On the current artifacts, that yields `lightgbm` for binary raw-feature SHAP and `random_forest` for multiclass raw-feature SHAP.

## Run Phase 11

```bash
python -m src.deployment.run_phase11 --config config/streaming_config.json
```

Phase 11 replays the selected-feature dataset as a simulated live stream, keeps Phase 7 LSTM state across batches, and performs real-time inference with the best persisted Phase 9 ensemble variants.

Primary artifacts:

- predictions: `data/streaming/stream_predictions.csv`
- report: `data/streaming/streaming_report.json`

The current real-data run streams the held-out `test` split with `504,472` rows, writes one prediction event per row, and selects:

- binary deployment variant: `weighted_scoring`
- multiclass deployment variant: `stacking`

Current performance on the real replay:

- average batch latency: `909.28 ms`
- p95 batch latency: `1146.95 ms`
- throughput: `546.42 rows/sec`

Because the default replay targets the held-out `test` split rather than the full corpus, the stateful LSTM history only spans observed test rows and does not include omitted rows outside that split.

## Run Phase 12

```bash
python -m src.deployment.run_phase12 --config config/alerting_config.json
```

Phase 12 consumes the Phase 11 streaming prediction log, calculates a severity-aware risk score for each streamed event, and emits operational alert records with three levels:

- `Normal`
- `Suspicious`
- `Attack`

Primary artifacts:

- enriched stream with alert metadata: `data/streaming/stream_predictions_with_alerts.csv`
- filtered alert log: `data/streaming/alerts.csv`
- report: `data/streaming/alerting_report.json`

The current real-data run processed the full held-out streaming replay with `504,472` rows and produced:

- total alert rows written: `85,089`
- level distribution: `419,383 Normal`, `21,864 Suspicious`, `63,225 Attack`
- average risk score: `15.84`
- maximum risk score: `84.67`

The most common predicted attack families among emitted alerts in the current real-data run are `DoS Hulk`, `DDoS`, and `PortScan`.

## Run Phase 13

```bash
streamlit run dashboard/streamlit_app.py
```

Phase 13 consumes the persisted outputs from Phases 9 through 12 and renders an operator-facing dashboard with:

- live recent prediction and alert tables from the Phase 11 and Phase 12 event logs
- alert-level and predicted-attack distributions from the Phase 12 report
- binary and compact multiclass confusion matrices computed from the Phase 11 replay output
- binary ROC visualization from the streamed attack probabilities
- Phase 10 SHAP-style feature summaries and Phase 9 ensemble component importance views

Primary dashboard inputs:

- streaming predictions: `data/streaming/stream_predictions.csv`
- enriched alert stream: `data/streaming/stream_predictions_with_alerts.csv`
- alert log: `data/streaming/alerts.csv`
- explainability artifacts: `models/saved_models/phase10_explainability/`
- ensemble metrics: `models/saved_models/phase9_ensemble/metrics_summary.csv`

Configuration defaults live in `config/dashboard_config.json`. You can override the config path at runtime with the `SENTINELNET_DASHBOARD_CONFIG` environment variable.

The current real-artifact snapshot build completed successfully against the persisted CICIDS2017 outputs and loaded:

- `504,472` streamed rows
- `85,089` emitted alert rows
- `20` recent predictions and `20` recent alerts in the validation snapshot
- binary ROC-AUC of `0.999858` from the Phase 11 replay

## Run Phase 14

```bash
uvicorn api.fastapi_app:app --host 0.0.0.0 --port 8000
```

Phase 14 exposes an artifact-backed FastAPI layer over the persisted SentinelNet outputs and deployment models. The API reuses the real Phase 11 predictor path, the Phase 12 alert scorer, and the Phase 13 metrics aggregation layer.

Primary endpoints:

- `POST /predict` for batch inference on selected-feature records with returned risk score and alert level
- `GET /stream` for replaying the enriched Phase 11 and Phase 12 event stream as JSON or NDJSON
- `GET /alerts` for filtered operational alert retrieval
- `GET /metrics` for overview metrics, confusion matrices, ROC data, and explainability summaries
- `GET /health` for readiness and artifact checks

Configuration defaults live in `config/api_config.json`. You can override the config path at runtime with the `SENTINELNET_API_CONFIG` environment variable.

### `/predict` Request and Response Schema

The `/predict` schema is already implemented in `api/schemas.py` using Pydantic:

- request models: `PredictRequest` and `PredictRecord`
- response models: `PredictResponse` and `PredictionRecordResponse`
- alert-level enum: `AlertLevel = Literal["Normal", "Suspicious", "Attack"]`

Request body:

```json
{
  "records": [
    {
      "source_file": "manual-soc-validation",
      "event_time_utc": "2026-04-30T08:45:00Z",
      "features": {
        "packet_length_variance": 0.0,
        "bwd_packet_length_mean": 0.0,
        "total_length_of_fwd_packets": 0.0
      }
    }
  ]
}
```

Important validation rules:

- `records` is required and must contain `1` to `2048` records.
- Each record may include `source_file`; default is `"api"`.
- Each record may include `event_time_utc`; if omitted, the API normalizes the current UTC timestamp.
- `features` is required and must be a numeric object.
- Extra top-level fields are rejected by Pydantic (`extra="forbid"`).
- The service layer requires every record to include all deployment selected features listed in `data/processed/feature_engineered/selected_feature_manifest.json`. Missing selected features return HTTP `422`.

Required `/predict` feature keys, in deployment order:

```text
packet_length_variance
bwd_packet_length_mean
total_length_of_fwd_packets
bwd_packet_length_max
init_win_bytes_backward
max_packet_length
fwd_packet_length_max
flow_iat_max
fwd_header_length
flow_duration
fwd_packet_length_mean
fwd_iat_mean
destination_port
flow_bytes_per_s
flow_iat_mean
fwd_iat_std
flow_iat_std
fwd_packet_length_std
rolling_unique_destination_ports_w20
bwd_iat_max
forward_payload_efficiency
bwd_iat_total
bwd_iat_mean
total_fwd_packets
burstiness_score
active_mean
active_min
bwd_iat_min
min_packet_length
flow_iat_min
```

Successful response body:

```json
{
  "record_count": 1,
  "feature_count": 30,
  "selected_binary_variant": "weighted_scoring",
  "selected_multiclass_variant": "stacking",
  "results": [
    {
      "stream_order": 0,
      "event_time_utc": "2026-04-30T08:45:00+00:00",
      "source_file": "manual-soc-validation",
      "predicted_binary_label": 1,
      "predicted_binary_label_name": "ATTACK",
      "binary_attack_probability": 0.97,
      "predicted_multiclass_label": 4,
      "predicted_multiclass_label_name": "DoS Hulk",
      "multiclass_confidence": 0.91,
      "selected_binary_variant": "weighted_scoring",
      "selected_multiclass_variant": "stacking",
      "risk_score": 84.2,
      "alert_level": "Attack",
      "is_alert": true,
      "alert_id": "SNT-ATTACK-00000000",
      "recommended_action": "Escalate immediately and isolate affected assets.",
      "alert_message": "Attack alert ..."
    }
  ]
}
```

Minimal `curl` shape:

```bash
curl -X POST http://127.0.0.1:8000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"records\":[{\"source_file\":\"manual\",\"features\":{\"packet_length_variance\":0,\"bwd_packet_length_mean\":0,\"total_length_of_fwd_packets\":0,\"bwd_packet_length_max\":0,\"init_win_bytes_backward\":0,\"max_packet_length\":0,\"fwd_packet_length_max\":0,\"flow_iat_max\":0,\"fwd_header_length\":0,\"flow_duration\":0,\"fwd_packet_length_mean\":0,\"fwd_iat_mean\":0,\"destination_port\":443,\"flow_bytes_per_s\":0,\"flow_iat_mean\":0,\"fwd_iat_std\":0,\"flow_iat_std\":0,\"fwd_packet_length_std\":0,\"rolling_unique_destination_ports_w20\":0,\"bwd_iat_max\":0,\"forward_payload_efficiency\":0,\"bwd_iat_total\":0,\"bwd_iat_mean\":0,\"total_fwd_packets\":0,\"burstiness_score\":0,\"active_mean\":0,\"active_min\":0,\"bwd_iat_min\":0,\"min_packet_length\":0,\"flow_iat_min\":0}}]}"
```

The current real-artifact Phase 14 smoke check completed successfully and returned:

- `/health`: `ready`
- `/alerts?limit=1`: `1` alert row returned
- `/stream?limit=1`: `1` replay row returned
- `/metrics`: `504,472` streamed rows and binary ROC-AUC `0.999858`
- `/predict`: `1` record scored with binary variant `weighted_scoring` and multiclass variant `stacking`

## Run React SOC Frontend

```bash
cd sentinelnet-frontend
npm install
npm run dev
```

The Vite app uses a dev proxy from `/api` to `http://127.0.0.1:8000`, so start the FastAPI backend first:

```bash
uvicorn api.fastapi_app:app --host 0.0.0.0 --port 8000
```

Frontend routes:

- `/` command-center metrics dashboard
- `/live` live stream replay with `limit`, `offset`, `alerts_only`, and `alert_level` controls backed by `GET /stream`
- `/alerts` alert center with `limit`, `offset`, `alert_level`, and `min_risk_score` controls backed by `GET /alerts`
- `/models` model and explainability insights from `GET /metrics`
- `/system` API/system performance views from `GET /metrics` and `GET /health`
- `/predict` inference lab for manual `POST /predict` validation

The frontend is intentionally wired to real backend responses. It does not hardcode fake telemetry; empty or validation-error states indicate missing API data, backend validation failures, or incomplete `/predict` feature payloads.

The FastAPI app also includes CORS middleware for local Vite origins. Override allowed origins with:

```bash
set SENTINELNET_CORS_ORIGINS=http://localhost:5173,http://localhost:5174,http://127.0.0.1:5173,http://127.0.0.1:5174
```

## Run Phase 15

```bash
python -m src.deployment.run_phase15 --config config/performance_config.json
```

Phase 15 benchmarks the real deployment path and writes tuned recommendations for streaming and API serving. It reuses:

- the persisted Phase 11 predictor stack for direct batch inference timing
- the Phase 14 API service for request-path profiling
- the Phase 13 metrics aggregation layer for refresh-versus-cache benchmarking

Primary artifacts:

- report: `models/saved_models/phase15_performance/performance_report.json`
- streaming batch benchmarks: `models/saved_models/phase15_performance/streaming_batch_benchmarks.csv`
- API predict benchmarks: `models/saved_models/phase15_performance/api_predict_benchmarks.csv`
- API read benchmarks: `models/saved_models/phase15_performance/api_read_benchmarks.csv`
- optimized streaming config: `models/saved_models/phase15_performance/optimized_streaming_config.json`
- optimized API config: `models/saved_models/phase15_performance/optimized_api_config.json`

The latest real Phase 15 rerun completed successfully on the persisted CICIDS2017 artifacts with:

- `2,048` benchmark rows across `30` selected features
- predictor load time of `10.54 s`
- API predictor warmup time of `7.28 s`
- recommended streaming `inference_batch_size`: `256`
- recommended API predict batch size: `8`
- recommended `/stream` and `/alerts` page size: `1000`
- metrics refresh latency: `4477.18 ms`
- cached metrics latency: `0.47 ms`
- metrics cache speedup: `9487.56x`

The generated optimized configs are still preserved as benchmark artifacts, and the active runtime defaults have now been aligned with the tuned recommendations:

- `config/streaming_config.json`: `inference_batch_size=256`
- `config/api_config.json`: `default_stream_limit=1000`, `default_alert_limit=1000`, `preload_predictor_on_startup=true`

## Run Phase 16

```bash
python -m src.deployment.run_phase16 --config config/phase16_config.json
```

Phase 16 consumes the Phase 11 replay order, the Phase 12 enriched event stream, and the persisted Phase 8 anomaly detectors to produce:

- operational attack-family classification for streamed events
- zero-day candidate detection from anomaly consensus and ensemble disagreement
- simulated auto-block actions with blocking scopes and repeat-offender tracking
- feature-drift summaries against the training baseline
- continuous-learning queues and a retraining recommendation manifest

Primary artifacts:

- report: `data/streaming/phase16_advanced/phase16_advanced_features_report.json`
- classified event stream: `data/streaming/phase16_advanced/phase16_classified_predictions.csv`
- zero-day candidates: `data/streaming/phase16_advanced/phase16_zero_day_candidates.csv`
- auto-block actions: `data/streaming/phase16_advanced/phase16_autoblock_actions.csv`
- continuous-learning queue: `data/streaming/phase16_advanced/phase16_continuous_learning_queue.csv`
- feature drift summary: `data/streaming/phase16_advanced/phase16_feature_drift_summary.csv`
- retraining manifest: `data/streaming/phase16_advanced/phase16_retraining_manifest.json`

The current real Phase 16 run completed successfully on the persisted CICIDS2017 artifacts with:

- `504,472` streamed rows processed across `30` selected features
- `0` zero-day candidates emitted under the current thresholds
- `0` simulated auto-block actions emitted under the current thresholds
- `5,000` continuous-learning queue entries retained at the configured cap
- `1` drifted feature detected against the `100,000`-row training baseline
- retraining recommendation: `true`, triggered by `continuous_learning_queue_rows>=1500`
- dominant attack families in the classified stream: `Normal Traffic`, `Availability Disruption`, and `Reconnaissance`

## Tests

```bash
pytest tests/test_data_pipeline.py tests/test_cleaning_pipeline.py tests/test_label_pipeline.py tests/test_preprocessing_pipeline.py tests/test_feature_engineering_pipeline.py tests/test_ml_training_pipeline.py tests/test_deep_learning_pipeline.py tests/test_anomaly_detection_pipeline.py tests/test_ensemble_pipeline.py tests/test_explainability_pipeline.py tests/test_streaming_pipeline.py tests/test_alerting_pipeline.py tests/test_dashboard_pipeline.py tests/test_api_pipeline.py tests/test_performance_pipeline.py tests/test_phase16_pipeline.py
```

## Full File Structure

Operational repository tree, excluding transient cache and VCS directories:

```text
SentinelNet/
|-- api/
|   |-- __init__.py
|   |-- config.py
|   |-- fastapi_app.py
|   |-- schemas.py
|   `-- service.py
|-- config/
|   |-- alerting_config.json
|   |-- anomaly_detection_config.json
|   |-- api_config.json
|   |-- cleaning_config.json
|   |-- dashboard_config.json
|   |-- deep_learning_config.json
|   |-- ensemble_config.json
|   |-- explainability_config.json
|   |-- feature_engineering_config.json
|   |-- ingestion_config.json
|   |-- label_config.json
|   |-- ml_training_config.json
|   |-- performance_config.json
|   |-- phase16_config.json
|   |-- preprocessing_config.json
|   `-- streaming_config.json
|-- dashboard/
|   |-- __init__.py
|   |-- config.py
|   |-- dashboard_data.py
|   `-- streamlit_app.py
|-- data/
|   |-- interim/
|   |   |-- cleaned.csv
|   |   |-- cleaned_report.json
|   |   |-- combined.csv
|   |   `-- combined_report.json
|   |-- processed/
|   |   |-- feature_engineered/
|   |   |-- labeled_dataset.csv
|   |   |-- label_mappings.json
|   |   |-- label_report.json
|   |   `-- preprocessed/
|   |-- raw/
|   `-- streaming/
|       |-- alerting_report.json
|       |-- alerts.csv
|       |-- phase16_advanced/
|       |   |-- phase16_advanced_features_report.json
|       |   |-- phase16_autoblock_actions.csv
|       |   |-- phase16_classified_predictions.csv
|       |   |-- phase16_continuous_learning_queue.csv
|       |   |-- phase16_feature_drift_summary.csv
|       |   |-- phase16_retraining_manifest.json
|       |   `-- phase16_zero_day_candidates.csv
|       |-- stream_predictions.csv
|       |-- stream_predictions_with_alerts.csv
|       `-- streaming_report.json
|-- logs/
|   |-- phase10_explainability.log
|   |-- phase11_streaming.log
|   |-- phase12_alerting.log
|   |-- phase14_api.log
|   |-- phase15_performance.log
|   |-- phase16_advanced_features.log
|   |-- phase1_ingestion.log
|   |-- phase2_cleaning.log
|   |-- phase3_label_handling.log
|   |-- phase4_preprocessing.log
|   |-- phase5_feature_engineering.log
|   |-- phase6_ml_training.log
|   |-- phase7_deep_learning.log
|   |-- phase8_anomaly_detection.log
|   |-- phase8_test_stderr.txt
|   |-- phase8_test_stdout.txt
|   `-- phase9_ensemble.log
|-- models/
|   `-- saved_models/
|       |-- phase10_explainability/
|       |-- phase15_performance/
|       |-- phase6_ml/
|       |-- phase7_deep_learning/
|       |-- phase8_anomaly_detection/
|       `-- phase9_ensemble/
|-- notebooks/
|   |-- EDA.ipynb
|   `-- feature_engineering.ipynb
|-- sentinelnet-frontend/
|   |-- index.html
|   |-- package.json
|   |-- vite.config.ts
|   `-- src/
|       |-- App.jsx
|       |-- components/
|       |-- hooks/
|       |-- layout/
|       |-- pages/
|       |-- services/
|       |-- styles/
|       `-- utils/
|-- src/
|   |-- __init__.py
|   |-- anomaly_detection/
|   |   |-- __init__.py
|   |   |-- __main__.py
|   |   |-- config.py
|   |   |-- data.py
|   |   |-- registry.py
|   |   |-- run_phase8.py
|   |   `-- training.py
|   |-- data_pipeline/
|   |   |-- __init__.py
|   |   |-- __main__.py
|   |   |-- cleaning.py
|   |   |-- config.py
|   |   |-- ingestion.py
|   |   |-- label_handling.py
|   |   |-- logging_utils.py
|   |   |-- optimizer.py
|   |   |-- preprocessing.py
|   |   |-- resampling.py
|   |   |-- run_phase1.py
|   |   |-- run_phase2.py
|   |   |-- run_phase3.py
|   |   |-- run_phase4.py
|   |   `-- schema.py
|   |-- deployment/
|   |   |-- __init__.py
|   |   |-- __main__.py
|   |   |-- alerting.py
|   |   |-- alerting_config.py
|   |   |-- config.py
|   |   |-- data.py
|   |   |-- performance.py
|   |   |-- performance_config.py
|   |   |-- phase16.py
|   |   |-- phase16_config.py
|   |   |-- predictor.py
|   |   |-- run_phase11.py
|   |   |-- run_phase12.py
|   |   |-- run_phase15.py
|   |   |-- run_phase16.py
|   |   `-- streaming.py
|   |-- evaluation/
|   |   |-- __init__.py
|   |   |-- anomaly_metrics.py
|   |   `-- classification_metrics.py
|   |-- explainability/
|   |   |-- __init__.py
|   |   |-- __main__.py
|   |   |-- attribution.py
|   |   |-- config.py
|   |   |-- data.py
|   |   |-- pipeline.py
|   |   `-- run_phase10.py
|   |-- feature_engineering/
|   |   |-- __init__.py
|   |   |-- __main__.py
|   |   |-- config.py
|   |   |-- feature_builder.py
|   |   |-- pipeline.py
|   |   |-- run_phase5.py
|   |   `-- selection.py
|   `-- models/
|       |-- deep_learning/
|       |   |-- __init__.py
|       |   |-- __main__.py
|       |   |-- architectures.py
|       |   |-- config.py
|       |   |-- data.py
|       |   |-- run_phase7.py
|       |   `-- training.py
|       |-- ensemble/
|       |   |-- __init__.py
|       |   |-- __main__.py
|       |   |-- config.py
|       |   |-- data.py
|       |   |-- run_phase9.py
|       |   `-- training.py
|       `-- ml/
|           |-- __init__.py
|           |-- __main__.py
|           |-- config.py
|           |-- data.py
|           |-- registry.py
|           |-- run_phase6.py
|           `-- training.py
|-- tests/
|   |-- conftest.py
|   |-- test_alerting_pipeline.py
|   |-- test_anomaly_detection_pipeline.py
|   |-- test_api_pipeline.py
|   |-- test_cleaning_pipeline.py
|   |-- test_dashboard_pipeline.py
|   |-- test_data_pipeline.py
|   |-- test_deep_learning_pipeline.py
|   |-- test_ensemble_pipeline.py
|   |-- test_explainability_pipeline.py
|   |-- test_feature_engineering_pipeline.py
|   |-- test_label_pipeline.py
|   |-- test_ml_training_pipeline.py
|   |-- test_performance_pipeline.py
|   |-- test_phase16_pipeline.py
|   |-- test_preprocessing_pipeline.py
|   `-- test_streaming_pipeline.py
|-- project_report.txt
|-- pytest.ini
|-- README.md
`-- requirements.txt
```
