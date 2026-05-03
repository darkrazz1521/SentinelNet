import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Pause, Play, RefreshCw, ShieldAlert, Trash2 } from 'lucide-react'
import Panel from '../components/Panel.jsx'
import ThreatTable from '../components/ThreatTable.jsx'
import EmptyState from '../components/EmptyState.jsx'
import { getStream } from '../services/api.js'
import { asArray, normalizeEvent } from '../utils/normalizers.js'

const isAttackRow = (row) =>
  row.alertLevel === 'attack' ||
  row.predictedAttack?.toLowerCase() === 'attack' ||
  (row.predictedAttack && row.predictedAttack.toLowerCase() !== 'benign' && row.riskScore >= 60)

export default function LiveFeed() {
  const [filters, setFilters] = useState({ limit: 120, offset: 0, alerts_only: false, alert_level: '' })
  const [paused, setPaused] = useState(false)
  const [continuous, setContinuous] = useState(true)
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [realTimeAlerts, setRealTimeAlerts] = useState([])
  const [totalScanned, setTotalScanned] = useState(0)
  const [lastBatchAt, setLastBatchAt] = useState(null)
  const inFlight = useRef(false)
  const filtersRef = useRef(filters)
  const pausedRef = useRef(paused)
  const continuousRef = useRef(continuous)

  useEffect(() => {
    filtersRef.current = filters
  }, [filters])

  useEffect(() => {
    pausedRef.current = paused
  }, [paused])

  useEffect(() => {
    continuousRef.current = continuous
  }, [continuous])

  const rows = useMemo(() => asArray(data, ['records', 'events', 'stream', 'predictions', 'data']).map(normalizeEvent), [data])
  const count = data?.count ?? rows.length
  const hasMore = Boolean(data?.has_more)

  const loadBatch = useCallback(async ({ manual = false } = {}) => {
    if (inFlight.current || (pausedRef.current && !manual)) return
    inFlight.current = true
    const activeFilters = filtersRef.current
    const params = {
      limit: activeFilters.limit,
      offset: activeFilters.offset,
      alerts_only: activeFilters.alerts_only,
    }
    if (activeFilters.alert_level) params.alert_level = activeFilters.alert_level

    try {
      const response = await getStream(params)
      const normalizedRows = asArray(response, ['records', 'events', 'stream', 'predictions', 'data']).map(normalizeEvent)
      const attackRows = normalizedRows.filter(isAttackRow)

      setData(response)
      setError(null)
      setLastBatchAt(new Date())
      setTotalScanned((value) => value + normalizedRows.length)

      if (attackRows.length) {
        setRealTimeAlerts((current) => {
          const seen = new Set(current.map((item) => item.id))
          const incoming = attackRows.filter((item) => !seen.has(item.id))
          return [...incoming, ...current].slice(0, 80)
        })
      }

      if (!manual && continuousRef.current) {
        setFilters((current) => ({
          ...current,
          offset: response?.has_more ? current.offset + current.limit : 0,
        }))
      }
    } catch (err) {
      setError(err)
    } finally {
      inFlight.current = false
    }
  }, [])

  useEffect(() => {
    loadBatch()
    const timer = window.setInterval(() => loadBatch(), 4000)
    return () => window.clearInterval(timer)
  }, [loadBatch])

  return (
    <div className="space-y-4">
      <Panel
        title="Streaming Controls"
        subtitle="Continuous replay scanner. Each tick reads 120 rows and pushes attacks into Real-Time Attack Alerts."
        action={
          <div className="flex flex-wrap gap-2">
            <button onClick={() => setPaused((value) => !value)} className="icon-button">
              {paused ? <Play size={16} /> : <Pause size={16} />}
              {paused ? 'Resume' : 'Pause'}
            </button>
            <button onClick={() => loadBatch({ manual: true })} className="icon-button"><RefreshCw size={16} />Refresh</button>
          </div>
        }
      >
        <div className="grid gap-3 md:grid-cols-5">
          <label className="control-field">
            <span>Limit</span>
            <select value={filters.limit} onChange={(event) => setFilters((current) => ({ ...current, limit: Number(event.target.value), offset: 0 }))}>
              {[120, 250, 500].map((value) => <option key={value} value={value}>{value} rows</option>)}
            </select>
          </label>
          <label className="control-field">
            <span>Alert Level</span>
            <select value={filters.alert_level} onChange={(event) => setFilters((current) => ({ ...current, alert_level: event.target.value, offset: 0 }))}>
              <option value="">All traffic</option>
              <option value="Normal">Normal</option>
              <option value="Suspicious">Suspicious</option>
              <option value="Attack">Attack</option>
            </select>
          </label>
          <label className="control-field">
            <span>Mode</span>
            <select value={filters.alerts_only ? 'alerts' : 'all'} onChange={(event) => setFilters((current) => ({ ...current, alerts_only: event.target.value === 'alerts', offset: 0 }))}>
              <option value="all">All predictions</option>
              <option value="alerts">Alerts only</option>
            </select>
          </label>
          <label className="control-field">
            <span>Scanner</span>
            <select value={continuous ? 'continuous' : 'manual'} onChange={(event) => setContinuous(event.target.value === 'continuous')}>
              <option value="continuous">Auto advance</option>
              <option value="manual">Manual offset</option>
            </select>
          </label>
          <div className="flex items-end gap-2">
            <button className="icon-button w-full justify-center" onClick={() => setFilters((current) => ({ ...current, offset: Math.max(0, current.offset - current.limit) }))}>Prev</button>
            <button className="icon-button w-full justify-center" disabled={!hasMore} onClick={() => setFilters((current) => ({ ...current, offset: current.offset + current.limit }))}>Next</button>
          </div>
        </div>
      </Panel>

      <Panel
        title="Real-Time Attack Alerts"
        subtitle={`${realTimeAlerts.length} attack rows captured while scanning ${totalScanned.toLocaleString()} streamed records${lastBatchAt ? ` - last batch ${lastBatchAt.toLocaleTimeString()}` : ''}`}
        action={<button className="icon-button" onClick={() => setRealTimeAlerts([])}><Trash2 size={16} />Clear</button>}
      >
        {realTimeAlerts.length ? (
          <div className="grid gap-3 xl:grid-cols-2">
            {realTimeAlerts.slice(0, 8).map((alert) => (
              <div key={alert.id} className="rounded-md border border-red-400/25 bg-red-500/10 p-4 shadow-[0_0_24px_rgba(239,68,68,.12)]">
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <p className="flex items-center gap-2 text-sm font-semibold text-red-200"><ShieldAlert size={16} />{alert.predictedAttack}</p>
                    <p className="mt-1 font-mono text-xs text-slate-500">{alert.timestamp}</p>
                  </div>
                  <div className="rounded-md border border-red-300/30 bg-red-400/10 px-2 py-1 font-mono text-sm text-red-200">
                    {alert.riskScore.toFixed(0)}
                  </div>
                </div>
                <p className="mt-3 text-sm text-slate-300">{alert.action}</p>
              </div>
            ))}
          </div>
        ) : (
          <EmptyState title="No attack alerts captured yet" message="The scanner is reading the stream in 120-row batches. Attack-level rows will appear here automatically." />
        )}
      </Panel>

      <Panel
        title="Streaming Predictions"
        subtitle={`${count} records loaded from offset ${filters.offset}${paused ? ' - polling paused' : ''}${continuous ? ' - auto advancing' : ''}`}
      >
        {error ? <EmptyState title="Stream endpoint offline" message="GET /stream is not responding. Polling continues in the background." /> : rows.length ? <ThreatTable rows={rows} /> : <EmptyState title="No live events" message="The stream endpoint returned no records for the selected filters." />}
      </Panel>
    </div>
  )
}
