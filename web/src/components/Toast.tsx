import { useEffect, useState } from 'react'

export interface ToastState {
  text: string
  type: 'success' | 'error'
}

/** Toast 宿主：3.5s 自动消失（对照模版 showToast 行为）。 */
export default function Toast({ state }: { state: ToastState | null }) {
  const [visible, setVisible] = useState(false)

  useEffect(() => {
    if (!state) return
    setVisible(true)
    const t = setTimeout(() => setVisible(false), 3500)
    return () => clearTimeout(t)
  }, [state])

  if (!state || !visible) return null
  return (
    <div
      className={`fixed top-6 left-1/2 -translate-x-1/2 z-50 px-5 py-2.5 rounded-xl text-sm font-medium shadow-lg border backdrop-blur-md ${
        state.type === 'success'
          ? 'bg-emerald-950/90 border-emerald-500/40 text-emerald-200'
          : 'bg-rose-950/90 border-rose-500/40 text-rose-200'
      }`}
    >
      {state.text}
    </div>
  )
}
