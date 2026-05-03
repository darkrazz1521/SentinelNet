import { useCallback, useEffect, useRef, useState } from 'react'

export const usePolling = (request, { interval = 5000, enabled = true, initialData = null } = {}) => {
  const [data, setData] = useState(initialData)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const mounted = useRef(true)
  const inFlight = useRef(false)

  const load = useCallback(async (force = false) => {
    if ((!enabled && !force) || inFlight.current) return
    inFlight.current = true
    try {
      const response = await request()
      if (!mounted.current) return
      setData(response)
      setError(null)
    } catch (err) {
      if (!mounted.current) return
      setError(err)
    } finally {
      inFlight.current = false
      if (mounted.current) setLoading(false)
    }
  }, [enabled, request])

  useEffect(() => {
    mounted.current = true
    load()
    const timer = enabled ? window.setInterval(load, interval) : null
    return () => {
      mounted.current = false
      if (timer) window.clearInterval(timer)
    }
  }, [enabled, interval, load])

  return { data, loading, error, refresh: () => load(true) }
}
