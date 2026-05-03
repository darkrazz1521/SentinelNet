import StatusPill from './StatusPill.jsx'
import { percent } from '../utils/normalizers.js'

export default function ThreatTable({ rows, mode = 'stream' }) {
  return (
    <div className="overflow-hidden rounded-md border border-slate-800/80">
      <div className="max-h-[620px] overflow-auto">
        <table className="min-w-full divide-y divide-slate-800 text-left text-sm">
          <thead className="sticky top-0 z-10 bg-slate-950/95 text-xs uppercase tracking-[0.14em] text-slate-500 backdrop-blur">
            <tr>
              <th className="px-4 py-3 font-semibold">Time</th>
              <th className="px-4 py-3 font-semibold">{mode === 'alerts' ? 'Attack Type' : 'Prediction'}</th>
              {mode === 'alerts' ? <th className="px-4 py-3 font-semibold">Source</th> : <th className="px-4 py-3 font-semibold">Confidence</th>}
              {mode === 'alerts' ? <th className="px-4 py-3 font-semibold">Target</th> : null}
              <th className="px-4 py-3 font-semibold">Risk</th>
              <th className="px-4 py-3 font-semibold">Level</th>
              <th className="px-4 py-3 font-semibold">Action</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/80">
            {rows.map((row) => {
              const hot = row.riskScore >= 80
              const warm = row.riskScore >= 55
              return (
                <tr
                  key={row.id}
                  className={`transition duration-300 ${hot ? 'animate-threatPulse bg-red-500/10' : warm ? 'bg-orange-500/10' : 'bg-emerald-500/[0.03]'} hover:bg-cyan-400/10`}
                >
                  <td className="whitespace-nowrap px-4 py-3 font-mono text-xs text-slate-400">{row.timestamp || 'live'}</td>
                  <td className="px-4 py-3 font-medium text-slate-100">{row.predictedAttack || row.attackType}</td>
                  {mode === 'alerts' ? (
                    <td className="px-4 py-3 font-mono text-xs text-slate-300">{row.source}</td>
                  ) : (
                    <td className="px-4 py-3 text-slate-300">{percent(row.confidence)}</td>
                  )}
                  {mode === 'alerts' ? <td className="px-4 py-3 font-mono text-xs text-slate-300">{row.target}</td> : null}
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-3">
                      <div className="h-1.5 w-20 rounded-full bg-slate-800">
                        <div className={`h-full rounded-full ${hot ? 'bg-red-400' : warm ? 'bg-orange-400' : 'bg-emerald-400'}`} style={{ width: `${Math.min(row.riskScore, 100)}%` }} />
                      </div>
                      <span className="font-mono text-xs text-slate-300">{row.riskScore.toFixed(0)}</span>
                    </div>
                  </td>
                  <td className="px-4 py-3"><StatusPill value={row.alertLevel || (hot ? 'critical' : warm ? 'suspicious' : 'normal')} /></td>
                  <td className="px-4 py-3 text-slate-300">{row.action || row.recommendation}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}
