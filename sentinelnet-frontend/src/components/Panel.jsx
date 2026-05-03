export default function Panel({ title, subtitle, action, children, className = '' }) {
  return (
    <section className={`glass-panel p-5 ${className}`}>
      {(title || subtitle || action) && (
        <header className="mb-5 flex items-start justify-between gap-4">
          <div>
            {title ? <h2 className="text-base font-semibold text-slate-100">{title}</h2> : null}
            {subtitle ? <p className="mt-1 text-sm text-slate-500">{subtitle}</p> : null}
          </div>
          {action}
        </header>
      )}
      {children}
    </section>
  )
}
