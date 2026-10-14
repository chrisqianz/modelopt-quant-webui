/** Toast context：view 通过 useToast() 弹提示，宿主在 App 里。 */
import { createContext, useCallback, useContext, useState, type ReactNode } from 'react'
import Toast, { type ToastState } from './Toast'

interface ToastApi {
  show: (text: string, type?: ToastState['type']) => void
}

const Ctx = createContext<ToastApi>({ show: () => {} })

export function useToast(): ToastApi {
  return useContext(Ctx)
}

export function ToastProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<ToastState | null>(null)
  const show = useCallback((text: string, type: ToastState['type'] = 'success') => {
    setState({ text, type })
  }, [])
  return (
    <Ctx.Provider value={{ show }}>
      {children}
      <Toast state={state} />
    </Ctx.Provider>
  )
}
