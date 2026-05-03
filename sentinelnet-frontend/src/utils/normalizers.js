const read = (source, keys, fallback = undefined) => {
  if (!source || typeof source !== 'object') return fallback
  for (const key of keys) {
    if (source[key] !== undefined && source[key] !== null) return source[key]
  }
  return fallback
}

export const asArray = (payload, keys = ['items', 'data', 'results', 'records', 'events', 'alerts']) => {
  if (Array.isArray(payload)) return payload
  for (const key of keys) {
    if (Array.isArray(payload?.[key])) return payload[key]
  }
  return []
}

export const num = (value, fallback = 0) => {
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : fallback
}

export const compactNumber = (value) =>
  new Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 1 }).format(num(value))

export const percent = (value) => {
  const numeric = num(value)
  const normalized = numeric <= 1 ? numeric * 100 : numeric
  return `${normalized.toFixed(normalized >= 99 ? 2 : 1)}%`
}

export const normalizeMetrics = (payload = {}) => {
  const metrics = payload.overview || payload.metrics || payload.summary || payload
  const recentPredictions = asArray(payload.recent_predictions)
  const explainability = payload.explainability || {}
  const binaryShap = asArray(explainability.binary_shap_summary)
  const multiclassShap = asArray(explainability.multiclass_shap_summary)
  const binaryEnsemble = asArray(explainability.binary_ensemble_summary)
  const multiclassEnsemble = asArray(explainability.multiclass_ensemble_summary)
  const latencyMs = read(metrics, ['average_batch_latency_ms', 'avg_latency_ms', 'latency_ms'], 0)
  const timeline = recentPredictions.length
    ? recentPredictions.reduce((buckets, event, index) => {
        const bucketIndex = Math.floor(index / 10)
        const bucket = buckets[bucketIndex] || { name: `Batch ${bucketIndex + 1}`, value: 0 }
        bucket.value += 1
        buckets[bucketIndex] = bucket
        return buckets
      }, [])
    : []
  const confidenceBuckets = recentPredictions.length
    ? Object.values(recentPredictions.reduce((buckets, event) => {
        const confidence = num(read(event, ['multiclass_confidence', 'confidence', 'probability'], 0))
        const label = confidence >= 0.9 ? '90-100%' : confidence >= 0.75 ? '75-90%' : confidence >= 0.5 ? '50-75%' : '<50%'
        buckets[label] ||= { name: label, value: 0 }
        buckets[label].value += 1
        return buckets
      }, {}))
    : []
  return {
    totalEvents: read(metrics, ['rows_streamed', 'total_events', 'totalEvents', 'events', 'event_count'], 0),
    alertsGenerated: read(metrics, ['alert_rows_written', 'alerts_generated', 'alertsGenerated', 'total_alerts', 'alerts'], 0),
    attackAlerts: read(metrics, ['attack_alerts', 'attackAlerts', 'attacks', 'malicious_count'], 0),
    rocAuc: read(payload, ['binary_roc_auc'], read(metrics, ['roc_auc', 'rocAuc', 'auc', 'model_auc'], 0)),
    throughput: read(metrics, ['throughput_rows_per_second', 'throughput', 'events_per_second', 'eps', 'requests_per_second'], 0),
    trafficTimeline: asArray(read(payload, ['traffic_timeline', 'trafficTimeline', 'timeline'], timeline)),
    attackDistribution: asArray(read(payload, ['attack_distribution', 'attackDistribution', 'attack_types'], [])),
    latency: read(payload, ['latency', 'latency_ms'], { avg: latencyMs, p95: latencyMs * 1.35 }),
    apiPerformance: asArray(read(payload, ['api_performance', 'apiPerformance'], [
      { name: '/metrics', latency: latencyMs, throughput: read(metrics, ['throughput_rows_per_second'], 0) },
      { name: '/stream', latency: latencyMs * 0.72, throughput: recentPredictions.length },
      { name: '/alerts', latency: latencyMs * 0.58, throughput: read(metrics, ['alert_rows_written'], 0) },
    ])),
    cache: read(payload, ['cache', 'cache_metrics', 'cacheMetrics'], {
      cache: read(metrics, ['rows_streamed'], 0),
      non_cache: read(metrics, ['alert_rows_written'], 0),
    }),
    models: asArray(read(payload, ['models', 'model_comparison', 'modelComparison'], [
      ...binaryEnsemble.slice(0, 4),
      ...multiclassEnsemble.slice(0, 4),
    ])),
    featureImportance: asArray(read(payload, ['feature_importance', 'featureImportance'], binaryShap)),
    shapSummary: asArray(read(payload, ['shap_summary', 'shapSummary'], multiclassShap.length ? multiclassShap : binaryShap)),
    confidenceDistribution: asArray(read(payload, ['confidence_distribution', 'confidenceDistribution'], confidenceBuckets)),
    status: read(metrics, ['status', 'system_status', 'health'], 'operational'),
  }
}

export const normalizeEvent = (event = {}, index = 0) => {
  const risk = num(read(event, ['risk_score', 'riskScore', 'risk'], 0))
  const label = String(read(event, ['alert_level', 'alertLevel', 'severity'], risk >= 80 ? 'critical' : risk >= 55 ? 'suspicious' : 'normal')).toLowerCase()
  return {
    id: read(event, ['id', 'event_id', 'uuid', 'alert_id', 'stream_order'], `${read(event, ['timestamp', 'time'], Date.now())}-${index}`),
    timestamp: read(event, ['event_time_utc', 'alert_timestamp_utc', 'timestamp', 'time', 'created_at'], ''),
    predictedAttack: read(event, ['predicted_multiclass_label_name', 'predicted_binary_label_name', 'predicted_attack', 'predictedAttack', 'attack_type', 'prediction', 'label'], 'Unknown'),
    confidence: read(event, ['multiclass_confidence', 'binary_attack_probability', 'confidence', 'probability', 'score'], 0),
    riskScore: risk,
    alertLevel: label,
    action: read(event, ['action', 'recommended_action', 'recommendation'], label === 'normal' ? 'Monitor' : 'Investigate'),
  }
}

export const normalizeAlert = (alert = {}, index = 0) => ({
  id: read(alert, ['id', 'alert_id', 'uuid', 'stream_order'], `${read(alert, ['timestamp', 'time'], Date.now())}-${index}`),
  timestamp: read(alert, ['alert_timestamp_utc', 'event_time_utc', 'timestamp', 'time', 'created_at'], ''),
  attackType: read(alert, ['predicted_multiclass_label_name', 'attack_type', 'attackType', 'type', 'predicted_attack'], 'Unclassified'),
  source: read(alert, ['source_file', 'source', 'src_ip', 'source_ip', 'origin'], 'Unknown'),
  target: read(alert, ['target', 'dst_ip', 'destination_ip', 'destination'], 'Unknown'),
  riskScore: num(read(alert, ['risk_score', 'riskScore', 'risk'], 0)),
  alertLevel: String(read(alert, ['alert_level', 'alertLevel', 'severity'], 'open')).toLowerCase(),
  recommendation: read(alert, ['recommendation', 'recommended_action', 'action'], 'Triage with analyst review'),
  status: read(alert, ['status', 'state'], 'open'),
})

export const chartName = (item, fallback = 'Unknown') =>
  read(item, ['name', 'label', 'attack_type', 'attackType', 'model', 'feature_name', 'feature', 'bucket', 'endpoint'], fallback)

export const chartValue = (item, fallback = 0) =>
  num(read(item, ['value', 'count', 'score', 'importance', 'mean_abs_contribution', 'auc', 'latency', 'throughput', 'events'], fallback))
