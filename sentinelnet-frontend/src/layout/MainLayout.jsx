import { NavLink, Outlet, useLocation } from 'react-router-dom'
import { Activity, Bell, BrainCircuit, ChevronLeft, ChevronRight, FlaskConical, Gauge, LayoutDashboard, Radar, Search, ShieldCheck, TerminalSquare } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { getHealth } from '../services/api.js'
import { usePolling } from '../hooks/usePolling.js'

const navItems = [
  { to: '/', label: 'Command Center', icon: LayoutDashboard },
  { to: '/live', label: 'Live Feed', icon: Radar },
  { to: '/alerts', label: 'Alert Center', icon: Bell },
  { to: '/models', label: 'Model Insights', icon: BrainCircuit },
  { to: '/system', label: 'System Metrics', icon: Gauge },
  { to: '/predict', label: 'Inference Lab', icon: FlaskConical },
]

const titles = {
  '/': ['Command Center', 'Unified threat posture and model telemetry'],
  '/live': ['Live Feed', 'Real-time predictions from SentinelNet v2'],
  '/alerts': ['Alert Center', 'Risk-ranked incidents and response actions'],
  '/models': ['Model Insights', 'ML, anomaly, ensemble, and explainability signals'],
  '/system': ['System Metrics', 'API performance and inference health'],
  '/predict': ['Inference Lab', 'Manual validation against the FastAPI /predict endpoint'],
}

export default function MainLayout() {
  const location = useLocation()
  const [now, setNow] = useState(new Date())
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [collapsed, setCollapsed] = useState(false)
  const { data: health, error: healthError } = usePolling(getHealth, { interval: 30000, initialData: null })

  useEffect(() => {
    const timer = window.setInterval(() => setNow(new Date()), 1000)
    return () => window.clearInterval(timer)
  }, [])

  const page = useMemo(() => titles[location.pathname] || titles['/'], [location.pathname])

  return (
    <div className="min-h-screen bg-[#0B0F17] text-slate-200">
      <aside className={`fixed inset-y-0 left-0 z-40 border-r border-cyan-400/10 bg-slate-950/90 backdrop-blur-xl transition-all duration-300 lg:translate-x-0 ${collapsed ? 'lg:w-24' : 'lg:w-72'} ${sidebarOpen ? 'w-72 translate-x-0' : 'w-72 -translate-x-full'}`}>
        <div className="flex h-full flex-col">
          <div className={`border-b border-slate-800 py-6 ${collapsed ? 'px-4' : 'px-6'}`}>
            <div className={`flex items-center ${collapsed ? 'justify-center' : 'gap-3'}`}>
              <div className="grid h-11 w-11 place-items-center rounded-md border border-cyan-300/30 bg-cyan-400/10 text-cyan-300 shadow-[0_0_30px_rgba(56,189,248,0.22)]">
                <ShieldCheck size={23} />
              </div>
              <div className={collapsed ? 'hidden' : ''}>
                <p className="text-lg font-semibold text-white">SentinelNet v2</p>
                <p className="text-xs uppercase tracking-[0.24em] text-cyan-300">AI IDS SOC</p>
              </div>
            </div>
          </div>

          <nav className={`flex-1 space-y-2 py-6 ${collapsed ? 'px-3' : 'px-4'}`}>
            {navItems.map(({ to, label, icon: Icon }) => (
              <NavLink
                key={to}
                to={to}
                end={to === '/'}
                onClick={() => setSidebarOpen(false)}
                className={({ isActive }) =>
                  `group flex items-center rounded-md py-3 text-sm font-medium transition ${collapsed ? 'justify-center px-3' : 'gap-3 px-4'} ${
                    isActive
                      ? 'border border-cyan-300/30 bg-cyan-400/10 text-cyan-200 shadow-[0_0_24px_rgba(56,189,248,0.14)]'
                      : 'text-slate-400 hover:bg-slate-900 hover:text-slate-100'
                  }`
                }
              >
                <Icon size={18} />
                <span className={collapsed ? 'sr-only' : ''}>{label}</span>
              </NavLink>
            ))}
          </nav>

          <div className="border-t border-slate-800 p-4">
            <button
              className="mb-3 hidden w-full items-center justify-center gap-2 rounded-md border border-slate-700 bg-slate-900/70 px-3 py-2 text-sm font-semibold text-slate-300 transition hover:border-cyan-300/40 hover:text-cyan-200 lg:flex"
              onClick={() => setCollapsed((value) => !value)}
              aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
            >
              {collapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
              <span className={collapsed ? 'sr-only' : ''}>Collapse</span>
            </button>
            <div className={`rounded-md border border-emerald-400/20 bg-emerald-400/10 ${collapsed ? 'grid place-items-center p-3' : 'p-4'}`}>
              <div className={`flex items-center gap-2 text-sm font-semibold text-emerald-300 ${collapsed ? 'justify-center' : ''}`}>
                <Activity size={16} />
                <span className={collapsed ? 'sr-only' : ''}>Backend Link</span>
              </div>
              <p className={`mt-2 text-xs text-slate-400 ${collapsed ? 'hidden' : ''}`}>/api proxy to FastAPI :8000</p>
            </div>
          </div>
        </div>
      </aside>

      <div className={`transition-all duration-300 ${collapsed ? 'lg:pl-24' : 'lg:pl-72'}`}>
        <header className="sticky top-0 z-30 border-b border-slate-800/80 bg-[#0B0F17]/86 backdrop-blur-xl">
          <div className="flex min-h-20 items-center justify-between gap-4 px-4 py-4 sm:px-6 lg:px-8">
            <div className="flex items-center gap-4">
              <button className="rounded-md border border-slate-700 p-2 text-slate-300 lg:hidden" onClick={() => setSidebarOpen(true)} aria-label="Open navigation">
                <TerminalSquare size={20} />
              </button>
              <div>
                <h1 className="text-xl font-semibold text-white sm:text-2xl">{page[0]}</h1>
                <p className="mt-1 text-sm text-slate-500">{page[1]}</p>
              </div>
            </div>

            <div className="hidden items-center gap-3 md:flex">
              <div className="flex items-center gap-2 rounded-md border border-slate-800 bg-slate-950/70 px-3 py-2 text-sm text-slate-400">
                <Search size={16} />
                Threat search
              </div>
              <div className="rounded-md border border-cyan-300/20 bg-cyan-400/10 px-3 py-2 font-mono text-sm text-cyan-200">
                {now.toLocaleTimeString()}
              </div>
              <div className={`rounded-md border px-3 py-2 text-sm font-semibold ${healthError ? 'border-red-400/20 bg-red-400/10 text-red-300' : 'border-emerald-400/20 bg-emerald-400/10 text-emerald-300'}`}>
                {healthError ? 'OFFLINE' : String(health?.status || 'ONLINE').toUpperCase()}
              </div>
            </div>
          </div>
        </header>

        <main className="px-4 py-6 sm:px-6 lg:px-8">
          <Outlet />
        </main>
      </div>
    </div>
  )
}
