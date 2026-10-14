/** 首屏一把：bootstrap + 重取。 */
import { useCallback, useEffect, useState } from 'react'
import { api } from '../api/endpoints'
import type { Bootstrap } from '../api/types'

export function useBootstrap() {
  const [data, setData] = useState<Bootstrap | null>(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)

  const reload = useCallback(async () => {
    try {
      const d = await api.bootstrap()
      setData(d)
      setError('')
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void reload()
  }, [reload])

  return { data, error, loading, reload }
}
