import { useState } from 'react'
import { Play, ShieldAlert } from 'lucide-react'
import Panel from '../components/Panel.jsx'
import EmptyState from '../components/EmptyState.jsx'
import ThreatTable from '../components/ThreatTable.jsx'
import { normalizeEvent } from '../utils/normalizers.js'
import { predictRecords } from '../services/api.js'

const examplePayload = `{
  "records": [
    {
      "source_file": "manual-soc-validation",
      "features": {
        "destination_port": 443,
        "flow_duration": 1200
      }
    }
  ]
}`

export default function PredictLab() {
  const [input, setInput] = useState(examplePayload)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [submitting, setSubmitting] = useState(false)

  const submit = async () => {
    setSubmitting(true)
    setError('')
    try {
      const payload = JSON.parse(input)
      const response = await predictRecords(payload)
      setResult(response)
    } catch (err) {
      const detail = err.response?.data?.detail
      setError(typeof detail === 'string' ? detail : err.message)
      setResult(null)
    } finally {
      setSubmitting(false)
    }
  }

  const rows = (result?.results || []).map(normalizeEvent)

  return (
    <div className="grid gap-6 xl:grid-cols-[0.95fr_1.3fr]">
      <Panel
        title="Prediction Request"
        subtitle="POST /predict with model-ready feature records"
        action={<button className="icon-button" onClick={submit} disabled={submitting}><Play size={16} />{submitting ? 'Running' : 'Run'}</button>}
      >
        <textarea
          value={input}
          onChange={(event) => setInput(event.target.value)}
          className="min-h-[460px] w-full resize-y rounded-md border border-slate-800 bg-slate-950/70 p-4 font-mono text-sm leading-6 text-slate-200 outline-none transition focus:border-cyan-300/60 focus:ring-4 focus:ring-cyan-400/10"
          spellCheck="false"
        />
        {error ? (
          <div className="mt-4 rounded-md border border-red-400/25 bg-red-400/10 p-3 text-sm text-red-200">
            {error}
          </div>
        ) : null}
      </Panel>

      <Panel title="Prediction Results" subtitle="Ensemble labels, confidence, risk score, and recommended action">
        {rows.length ? (
          <ThreatTable rows={rows} />
        ) : (
          <EmptyState
            title="No prediction response yet"
            message="Submit a valid model-ready payload. The backend will reject records missing required selected features."
          />
        )}
        {result ? (
          <div className="mt-4 grid gap-3 md:grid-cols-3">
            <div className="rounded-md border border-slate-800 bg-slate-950/50 p-4">
              <p className="text-xs uppercase tracking-[0.14em] text-slate-500">Records</p>
              <p className="mt-2 text-2xl font-semibold text-slate-100">{result.record_count}</p>
            </div>
            <div className="rounded-md border border-slate-800 bg-slate-950/50 p-4">
              <p className="text-xs uppercase tracking-[0.14em] text-slate-500">Features</p>
              <p className="mt-2 text-2xl font-semibold text-slate-100">{result.feature_count}</p>
            </div>
            <div className="rounded-md border border-cyan-400/20 bg-cyan-400/10 p-4">
              <p className="flex items-center gap-2 text-xs uppercase tracking-[0.14em] text-cyan-300"><ShieldAlert size={14} />Variants</p>
              <p className="mt-2 text-sm text-slate-200">{result.selected_binary_variant} / {result.selected_multiclass_variant}</p>
            </div>
          </div>
        ) : null}
      </Panel>
    </div>
  )
}
