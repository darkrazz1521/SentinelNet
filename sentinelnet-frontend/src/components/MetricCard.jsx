const toneClasses = {
  cyan: 'from-cyan-400/20 to-sky-500/5 text-cyan-300 shadow-cyan-500/10',
  orange: 'from-orange-400/20 to-amber-500/5 text-orange-300 shadow-orange-500/10',
  red: 'from-red-500/20 to-rose-500/5 text-red-300 shadow-red-500/10',
  green: 'from-emerald-400/20 to-teal-500/5 text-emerald-300 shadow-emerald-500/10',
}

export default function MetricCard({ title, value, detail, icon: Icon, tone = 'cyan' }) {
  return (
    <section className="glass-panel group min-h-[134px] p-5 transition duration-300 hover:-translate-y-1 hover:border-cyan-300/40 hover:shadow-2xl">
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">{title}</p>
          <div className="mt-4 text-3xl font-semibold text-slate-50">{value}</div>
        </div>
        {Icon ? (
          <div className={`rounded-md bg-gradient-to-br p-3 shadow-lg ${toneClasses[tone] || toneClasses.cyan}`}>
            <Icon size={20} />
          </div>
        ) : null}
      </div>
      {detail ? <p className="mt-4 text-sm text-slate-400">{detail}</p> : null}
    </section>
  )
}
