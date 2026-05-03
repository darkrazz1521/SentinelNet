import { WifiOff } from 'lucide-react'

export default function EmptyState({ title = 'Telemetry unavailable', message = 'Waiting for the FastAPI backend to return live SOC data.' }) {
  return (
    <div className="flex min-h-[220px] flex-col items-center justify-center rounded-md border border-dashed border-slate-700/80 bg-slate-950/30 p-8 text-center">
      <WifiOff className="text-slate-500" size={28} />
      <h3 className="mt-4 text-sm font-semibold text-slate-300">{title}</h3>
      <p className="mt-2 max-w-md text-sm text-slate-500">{message}</p>
    </div>
  )
}
