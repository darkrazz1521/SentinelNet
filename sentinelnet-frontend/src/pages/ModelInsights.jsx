import { useMemo } from 'react'
import { Bar, BarChart, CartesianGrid, Cell, Line, LineChart, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis } from 'recharts'
import Panel from '../components/Panel.jsx'
import EmptyState from '../components/EmptyState.jsx'
import { getMetrics } from '../services/api.js'
import { usePolling } from '../hooks/usePolling.js'
import { chartName, chartValue, normalizeMetrics, percent } from '../utils/normalizers.js'

export default function ModelInsights() {
  const { data, error } = usePolling(getMetrics, { interval: 90000, initialData: {} })
  const metrics = useMemo(() => normalizeMetrics(data), [data])
  const models = useMemo(() => metrics.models.map((item) => ({ name: chartName(item), value: chartValue(item) })), [metrics.models])
  const features = useMemo(() => metrics.featureImportance.map((item) => ({ name: chartName(item), value: chartValue(item) })).slice(0, 12), [metrics.featureImportance])
  const shap = useMemo(() => metrics.shapSummary.map((item, index) => ({
    feature: chartName(item, `feature_${index}`),
    impact: chartValue(item),
    magnitude: chartValue(item, 1),
  })), [metrics.shapSummary])
  const confidence = useMemo(() => metrics.confidenceDistribution.map((item) => ({ bucket: chartName(item), value: chartValue(item) })), [metrics.confidenceDistribution])

  return (
    <div className="space-y-6">
      {error ? <EmptyState title="Model telemetry unavailable" message="GET /metrics is not returning model insight fields yet." /> : null}

      <section className="grid gap-6 xl:grid-cols-[1fr_1fr]">
        <Panel title="Model Comparison" subtitle="RF, XGBoost, Ensemble, or backend-provided model scores">
          {models.length ? (
            <div className="grid gap-3">
              {models.map((model, index) => (
                <div key={model.name} className="rounded-md border border-slate-800 bg-slate-950/50 p-4">
                  <div className="flex items-center justify-between">
                    <span className="font-semibold text-slate-100">{model.name}</span>
                    <span className="font-mono text-cyan-200">{percent(model.value)}</span>
                  </div>
                  <div className="mt-3 h-2 rounded-full bg-slate-800">
                    <div className="h-full rounded-full shadow-[0_0_16px_rgba(56,189,248,.45)]" style={{ width: `${Math.min(model.value <= 1 ? model.value * 100 : model.value, 100)}%`, background: index % 2 ? '#F59E0B' : '#38BDF8' }} />
                  </div>
                </div>
              ))}
            </div>
          ) : <EmptyState title="No model comparison" message="Add models or model_comparison to /metrics to populate this panel." />}
        </Panel>

        <Panel title="Confidence Distribution" subtitle="Prediction certainty across inference buckets">
          {confidence.length ? (
            <div className="h-72">
              <ResponsiveContainer>
                <LineChart data={confidence}>
                  <CartesianGrid stroke="#1E293B" strokeDasharray="3 3" />
                  <XAxis dataKey="bucket" stroke="#64748B" tickLine={false} axisLine={false} />
                  <YAxis stroke="#64748B" tickLine={false} axisLine={false} />
                  <Tooltip contentStyle={{ background: '#0F172A', border: '1px solid rgba(56,189,248,.25)', borderRadius: 8 }} />
                  <Line type="monotone" dataKey="value" stroke="#38BDF8" strokeWidth={3} dot={{ fill: '#38BDF8', r: 4 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          ) : <EmptyState title="No confidence buckets" message="Confidence distribution will appear when /metrics returns confidence_distribution." />}
        </Panel>
      </section>

      <section className="grid gap-6 xl:grid-cols-[1.25fr_.9fr]">
        <Panel title="Feature Importance" subtitle="Top drivers influencing model decisions">
          {features.length ? (
            <div className="h-96">
              <ResponsiveContainer>
                <BarChart data={features} layout="vertical" margin={{ left: 28 }}>
                  <XAxis type="number" stroke="#64748B" tickLine={false} axisLine={false} />
                  <YAxis type="category" dataKey="name" stroke="#64748B" tickLine={false} axisLine={false} width={130} />
                  <Tooltip contentStyle={{ background: '#0F172A', border: '1px solid rgba(56,189,248,.25)', borderRadius: 8 }} />
                  <Bar dataKey="value" fill="#38BDF8" radius={[0, 6, 6, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : <EmptyState title="No feature importance" message="Add feature_importance to /metrics to populate this chart." />}
        </Panel>

        <Panel title="SHAP Summary" subtitle="Explainability impact and magnitude">
          {shap.length ? (
            <div className="h-96">
              <ResponsiveContainer>
                <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 0 }}>
                  <XAxis dataKey="impact" name="Impact" stroke="#64748B" tickLine={false} axisLine={false} />
                  <YAxis dataKey="feature" type="category" name="Feature" stroke="#64748B" tickLine={false} axisLine={false} width={120} />
                  <ZAxis dataKey="magnitude" range={[60, 360]} />
                  <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ background: '#0F172A', border: '1px solid rgba(56,189,248,.25)', borderRadius: 8 }} />
                  <Scatter data={shap} fill="#F59E0B">
                    {shap.map((item, index) => <Cell key={item.feature} fill={index % 2 ? '#38BDF8' : '#F59E0B'} />)}
                  </Scatter>
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          ) : <EmptyState title="No SHAP summary" message="SHAP impact points will appear when /metrics returns shap_summary." />}
        </Panel>
      </section>
    </div>
  )
}
