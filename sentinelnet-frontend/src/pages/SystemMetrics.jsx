import { useMemo } from 'react'
import { Area, AreaChart, Bar, BarChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { Cpu, Gauge, Network, Timer } from 'lucide-react'
import MetricCard from '../components/MetricCard.jsx'
import Panel from '../components/Panel.jsx'
import EmptyState from '../components/EmptyState.jsx'
import { getMetrics } from '../services/api.js'
import { usePolling } from '../hooks/usePolling.js'
import { chartName, chartValue, compactNumber, normalizeMetrics, num } from '../utils/normalizers.js'

export default function SystemMetrics() {
  const { data, error } = usePolling(getMetrics, { interval: 60000, initialData: {} })
  const metrics = useMemo(() => normalizeMetrics(data), [data])
  const latency = metrics.latency || {}
  const apiPerformance = useMemo(() => metrics.apiPerformance.map((item) => ({
    name: chartName(item),
    latency: chartValue(item),
    throughput: num(item.throughput || item.requests || item.rps),
  })), [metrics.apiPerformance])
  const cache = metrics.cache || {}
  const cacheChart = [
    { name: 'Cache', value: num(cache.cache || cache.cached || cache.cache_hits || cache.hits) },
    { name: 'Non-cache', value: num(cache.non_cache || cache.uncached || cache.cache_misses || cache.misses) },
  ].filter((item) => item.value > 0)

  return (
    <div className="space-y-6">
      {error ? <EmptyState title="System metrics unavailable" message="GET /metrics is not returning system performance data yet." /> : null}

      <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <MetricCard title="Avg Latency" value={`${num(latency.avg || latency.average || latency.avg_ms).toFixed(1)} ms`} detail="Mean inference response" icon={Timer} />
        <MetricCard title="P95 Latency" value={`${num(latency.p95 || latency.p95_ms).toFixed(1)} ms`} detail="Tail latency guardrail" icon={Gauge} tone="orange" />
        <MetricCard title="Throughput" value={`${compactNumber(metrics.throughput)}/s`} detail="Processed events" icon={Network} tone="green" />
        <MetricCard title="API Health" value={String(metrics.status).toUpperCase()} detail="Backend reported status" icon={Cpu} />
      </section>

      <section className="grid gap-6 xl:grid-cols-[1.4fr_.9fr]">
        <Panel title="API Performance" subtitle="Endpoint latency and request pressure">
          {apiPerformance.length ? (
            <div className="h-80">
              <ResponsiveContainer>
                <AreaChart data={apiPerformance}>
                  <defs>
                    <linearGradient id="latencyFill" x1="0" x2="0" y1="0" y2="1">
                      <stop offset="5%" stopColor="#F59E0B" stopOpacity={0.38} />
                      <stop offset="95%" stopColor="#F59E0B" stopOpacity={0.03} />
                    </linearGradient>
                  </defs>
                  <XAxis dataKey="name" stroke="#64748B" tickLine={false} axisLine={false} />
                  <YAxis stroke="#64748B" tickLine={false} axisLine={false} />
                  <Tooltip contentStyle={{ background: '#0F172A', border: '1px solid rgba(56,189,248,.25)', borderRadius: 8 }} />
                  <Area dataKey="latency" name="Latency ms" stroke="#F59E0B" strokeWidth={3} fill="url(#latencyFill)" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          ) : <EmptyState title="No API performance series" message="Add api_performance to /metrics to populate endpoint performance." />}
        </Panel>

        <Panel title="Cache vs Non-cache" subtitle="Inference path efficiency">
          {cacheChart.length ? (
            <div className="h-80">
              <ResponsiveContainer>
                <BarChart data={cacheChart}>
                  <XAxis dataKey="name" stroke="#64748B" tickLine={false} axisLine={false} />
                  <YAxis stroke="#64748B" tickLine={false} axisLine={false} />
                  <Tooltip contentStyle={{ background: '#0F172A', border: '1px solid rgba(56,189,248,.25)', borderRadius: 8 }} />
                  <Bar dataKey="value" fill="#38BDF8" radius={[6, 6, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : <EmptyState title="No cache telemetry" message="Cache and non-cache bars will appear when /metrics returns cache metrics." />}
        </Panel>
      </section>
    </div>
  )
}
