const statusTone = {
  normal: 'border-emerald-400/30 bg-emerald-400/10 text-emerald-300',
  low: 'border-emerald-400/30 bg-emerald-400/10 text-emerald-300',
  operational: 'border-emerald-400/30 bg-emerald-400/10 text-emerald-300',
  suspicious: 'border-orange-400/30 bg-orange-400/10 text-orange-300',
  medium: 'border-orange-400/30 bg-orange-400/10 text-orange-300',
  high: 'border-red-400/30 bg-red-400/10 text-red-300',
  critical: 'border-red-400/30 bg-red-400/10 text-red-300',
  attack: 'border-red-400/30 bg-red-400/10 text-red-300',
}

export default function StatusPill({ value }) {
  const normalized = String(value || 'unknown').toLowerCase()
  return (
    <span className={`inline-flex items-center gap-2 rounded-md border px-2.5 py-1 text-xs font-semibold uppercase tracking-[0.14em] ${statusTone[normalized] || 'border-slate-500/30 bg-slate-500/10 text-slate-300'}`}>
      <span className="h-1.5 w-1.5 rounded-full bg-current shadow-[0_0_12px_currentColor]" />
      {normalized}
    </span>
  )
}
