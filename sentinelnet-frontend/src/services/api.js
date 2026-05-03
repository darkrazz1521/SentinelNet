import axios from 'axios'

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || '/api',
  timeout: 90000,
})

export const getMetrics = async () => {
  const { data } = await api.get('/metrics', {
    params: {
      recent_rows: 120,
      explanation_top_n: 8,
      multiclass_top_k: 8,
    },
  })
  return data
}

export const getHealth = async () => {
  const { data } = await api.get('/health')
  return data
}

export const getStream = async (params = {}) => {
  const { data } = await api.get('/stream', {
    params: {
      limit: 120,
      offset: 0,
      alerts_only: false,
      ...params,
    },
  })
  return data
}

export const getAlerts = async (params = {}) => {
  const { data } = await api.get('/alerts', {
    params: {
      limit: 120,
      offset: 0,
      ...params,
    },
  })
  return data
}

export const predictRecords = async (payload) => {
  const { data } = await api.post('/predict', payload)
  return data
}

export default api
