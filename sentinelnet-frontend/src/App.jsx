import { Navigate, Route, Routes } from 'react-router-dom'
import MainLayout from './layout/MainLayout.jsx'
import Dashboard from './pages/Dashboard.jsx'
import LiveFeed from './pages/LiveFeed.jsx'
import AlertCenter from './pages/AlertCenter.jsx'
import ModelInsights from './pages/ModelInsights.jsx'
import SystemMetrics from './pages/SystemMetrics.jsx'
import PredictLab from './pages/PredictLab.jsx'

export default function App() {
  return (
    <Routes>
      <Route element={<MainLayout />}>
        <Route index element={<Dashboard />} />
        <Route path="/live" element={<LiveFeed />} />
        <Route path="/alerts" element={<AlertCenter />} />
        <Route path="/models" element={<ModelInsights />} />
        <Route path="/system" element={<SystemMetrics />} />
        <Route path="/predict" element={<PredictLab />} />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}
