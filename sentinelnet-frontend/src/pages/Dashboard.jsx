import { useMemo } from 'react'
import { Area, AreaChart, Cell, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { Activity, Bell, Gauge, RadioTower, ShieldAlert } from 'lucide-react'
import MetricCard from '../components/MetricCard.jsx'
import Panel from '../components/Panel.jsx'
import StatusPill from '../components/StatusPill.jsx'
import EmptyState from '../components/EmptyState.jsx'
import { getMetrics } from '../services/api.js'
import { usePolling } from '../hooks/usePolling.js'
import { chartName, chartValue, compactNumber, normalizeMetrics, percent } from '../utils/normalizers.js'

const colors = ['#38BDF8', '#EF4444', '#F59E0B', '#22C55E', '#A78BFA']

export default function Dashboard() {
  const { data, error } = usePolling(getMetrics, { interval: 60000, initialData: {} })
  const metrics = useMemo(() => normalizeMetrics(data), [data])
  const timeline = useMemo(() => metrics.trafficTimeline.map((item, index) => ({
    time: chartName(item, `T-${index}`),
    events: chartValue(item),
  })), [metrics.trafficTimeline])
  const distribution = useMemo(() => metrics.attackDistribution.map((item) => ({
    name: chartName(item),
    value: chartValue(item),
  })), [metrics.attackDistribution])

  return (
    <div className="space-y-6">
      {error ? <EmptyState title="Backend connection degraded" message="Metrics could not be loaded through the Vite /api proxy. The interface will recover automatically when the API responds." /> : null}

      <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-5">
        <MetricCard title="Total Events" value={compactNumber(metrics.totalEvents)} detail="Streamed IDS observations" icon={RadioTower} />
        <MetricCard title="Alerts Generated" value={compactNumber(metrics.alertsGenerated)} detail="Risk scored detections" icon={Bell} tone="orange" />
        <MetricCard title="Attack Alerts" value={compactNumber(metrics.attackAlerts)} detail="Malicious predictions" icon={ShieldAlert} tone="red" />
        <MetricCard title="ROC AUC" value={percent(metrics.rocAuc)} detail="Current model quality" icon={Gauge} tone="green" />
        <MetricCard title="Throughput" value={`${compactNumber(metrics.throughput)}/s`} detail="Inference pipeline rate" icon={Activity} />
      </section>

      <section className="grid gap-6 xl:grid-cols-[1.6fr_1fr]">
        <Panel title="Traffic Timeline" subtitle="Event volume over the latest telemetry window">
          {timeline.length ? (
            <div className="h-80">
              <ResponsiveContainer>
                <AreaChart data={timeline}>
                  <defs>
                    <linearGradient id="trafficGlow" x1="0" x2="0" y1="0" y2="1">
                      <stop offset="5%" stopColor="#38BDF8" stopOpacity={0.42} />
                      <stop offset="95%" stopColor="#38BDF8" stopOpacity={0.02} />
                    </linearGradient>
                  </defs>
                  <XAxis dataKey="time" stroke="#64748B" tickLine={false} axisLine={false} />
                  <YAxis stroke="#64748B" tickLine={false} axisLine={false} width={44} />
                  <Tooltip contentStyle={{ background: '#0F172A', border: '1px solid rgba(56,189,248,.25)', borderRadius: 8 }} />
                  <Area type="monotone" dataKey="events" stroke="#38BDF8" strokeWidth={3} fill="url(#trafficGlow)" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          ) : <EmptyState title="No timeline points" message="The metrics endpoint has not returned traffic_timeline data yet." />}
        </Panel>

        <Panel title="Attack Distribution" subtitle="Observed attack classes in the active window">
          {distribution.length ? (
            <div className="grid gap-4 md:grid-cols-[220px_1fr] xl:grid-cols-1">
              <div className="h-64">
                <ResponsiveContainer>
                  <PieChart>
                    <Pie data={distribution} innerRadius={68} outerRadius={100} paddingAngle={4} dataKey="value">
                      {distribution.map((entry, index) => <Cell key={entry.name} fill={colors[index % colors.length]} />)}
                    </Pie>
                    <Tooltip contentStyle={{ background: '#0F172A', border: '1px solid rgba(56,189,248,.25)', borderRadius: 8 }} />
                  </PieChart>
                </ResponsiveContainer>
              </div>
              <div className="space-y-3">
                {distribution.map((item, index) => (
                  <div key={item.name} className="flex items-center justify-between rounded-md bg-slate-950/50 px-3 py-2">
                    <span className="flex items-center gap-2 text-sm text-slate-300"><span className="h-2 w-2 rounded-full" style={{ background: colors[index % colors.length] }} />{item.name}</span>
                    <span className="font-mono text-sm text-slate-100">{compactNumber(item.value)}</span>
                  </div>
                ))}
              </div>
            </div>
          ) : <EmptyState title="No distribution data" message="The metrics endpoint has not returned attack_distribution data yet." />}
        </Panel>
      </section>

      <Panel title="System Status" subtitle="Operational readiness from backend health telemetry">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <StatusPill value={metrics.status} />
          <div className="text-sm text-slate-400">Inference, alerting, explainability, and streaming channels are monitored through FastAPI telemetry.</div>
        </div>
      </Panel>
    </div>
  )
}
