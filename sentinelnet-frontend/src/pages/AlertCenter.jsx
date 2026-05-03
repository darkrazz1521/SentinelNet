import { useCallback, useMemo, useState } from 'react'
import { Bar, BarChart, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import Panel from '../components/Panel.jsx'
import ThreatTable from '../components/ThreatTable.jsx'
import EmptyState from '../components/EmptyState.jsx'
import { getAlerts } from '../services/api.js'
import { usePolling } from '../hooks/usePolling.js'
import { asArray, chartName, chartValue, normalizeAlert } from '../utils/normalizers.js'

export default function AlertCenter() {
  const [filters, setFilters] = useState({ limit: 120, offset: 0, alert_level: '', min_risk_score: '' })
  const request = useCallback(() => {
    const params = {
      limit: filters.limit,
      offset: filters.offset,
    }
    if (filters.alert_level) params.alert_level = filters.alert_level
    if (filters.min_risk_score !== '') params.min_risk_score = Number(filters.min_risk_score)
    return getAlerts(params)
  }, [filters])
  const { data, error, refresh } = usePolling(request, { interval: 10000, initialData: [] })
  const alerts = useMemo(() => asArray(data).map(normalizeAlert).sort((a, b) => b.riskScore - a.riskScore), [data])
  const hasMore = Boolean(data?.has_more)
  const distribution = useMemo(() => {
    const explicit = asArray(data?.attack_distribution || data?.attackDistribution || [])
    if (explicit.length) return explicit.map((item) => ({ name: chartName(item), value: chartValue(item) }))
    return Object.values(alerts.reduce((acc, alert) => {
      acc[alert.attackType] ||= { name: alert.attackType, value: 0 }
      acc[alert.attackType].value += 1
      return acc
    }, {}))
  }, [alerts, data])

  return (
    <div className="grid gap-6 xl:grid-cols-[1.4fr_.9fr]">
      <Panel title="Alert Filters" subtitle="Backed by GET /alerts filters: limit, offset, alert_level, and min_risk_score" className="xl:col-span-2">
        <div className="grid gap-3 md:grid-cols-5">
          <label className="control-field">
            <span>Limit</span>
            <select value={filters.limit} onChange={(event) => setFilters((current) => ({ ...current, limit: Number(event.target.value), offset: 0 }))}>
              {[50, 120, 250, 500].map((value) => <option key={value} value={value}>{value} alerts</option>)}
            </select>
          </label>
          <label className="control-field">
            <span>Level</span>
            <select value={filters.alert_level} onChange={(event) => setFilters((current) => ({ ...current, alert_level: event.target.value, offset: 0 }))}>
              <option value="">All alerts</option>
              <option value="Suspicious">Suspicious</option>
              <option value="Attack">Attack</option>
            </select>
          </label>
          <label className="control-field">
            <span>Min Risk</span>
            <input type="number" min="0" max="100" step="5" value={filters.min_risk_score} placeholder="Any" onChange={(event) => setFilters((current) => ({ ...current, min_risk_score: event.target.value, offset: 0 }))} />
          </label>
          <div className="flex items-end gap-2 md:col-span-2">
            <button className="icon-button w-full justify-center" onClick={() => setFilters((current) => ({ ...current, offset: Math.max(0, current.offset - current.limit) }))}>Prev</button>
            <button className="icon-button w-full justify-center" disabled={!hasMore} onClick={() => setFilters((current) => ({ ...current, offset: current.offset + current.limit }))}>Next</button>
            <button className="icon-button w-full justify-center" onClick={refresh}>Apply</button>
          </div>
        </div>
      </Panel>

      <Panel title="High-Risk Alerts" subtitle="Prioritized incidents with high-risk rows highlighted" className="xl:row-span-2">
        {error ? <EmptyState title="Alerts endpoint offline" message="GET /alerts is not responding. Polling continues in the background." /> : alerts.length ? <ThreatTable rows={alerts} mode="alerts" /> : <EmptyState title="No alerts returned" message="Waiting for alert records from GET /alerts." />}
      </Panel>

      <Panel title="Risk Score Focus" subtitle="Highest active incidents">
        <div className="space-y-3">
          {alerts.slice(0, 5).map((alert) => (
            <div key={alert.id} className="rounded-md border border-slate-800 bg-slate-950/50 p-3">
              <div className="flex items-center justify-between gap-3">
                <span className="text-sm font-semibold text-slate-100">{alert.attackType}</span>
                <span className="font-mono text-sm text-red-300">{alert.riskScore.toFixed(0)}</span>
              </div>
              <div className="mt-3 h-1.5 rounded-full bg-slate-800">
                <div className="h-full rounded-full bg-red-400 shadow-[0_0_14px_rgba(239,68,68,.55)]" style={{ width: `${Math.min(alert.riskScore, 100)}%` }} />
              </div>
            </div>
          ))}
          {!alerts.length ? <EmptyState title="No risk scores" message="Risk visualization appears once alerts are available." /> : null}
        </div>
      </Panel>

      <Panel title="Attack Type Distribution" subtitle="Alert volume by detected class">
        {distribution.length ? (
          <div className="h-72">
            <ResponsiveContainer>
              <BarChart data={distribution} layout="vertical" margin={{ left: 24 }}>
                <XAxis type="number" stroke="#64748B" tickLine={false} axisLine={false} />
                <YAxis type="category" dataKey="name" stroke="#64748B" tickLine={false} axisLine={false} width={110} />
                <Tooltip contentStyle={{ background: '#0F172A', border: '1px solid rgba(56,189,248,.25)', borderRadius: 8 }} />
                <Bar dataKey="value" radius={[0, 6, 6, 0]}>
                  {distribution.map((item, index) => <Cell key={item.name} fill={index % 2 ? '#F59E0B' : '#38BDF8'} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        ) : <EmptyState title="No attack classes" message="Attack type distribution will populate from alerts telemetry." />}
      </Panel>
    </div>
  )
}
