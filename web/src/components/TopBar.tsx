/** 顶栏：区标题 + 版本徽章 + GPU 显存指示。 */
import type { Capabilities, HostStatus } from '../api/types'

export default function TopBar({
  title, probe, caps,
}: {
  title: string
  probe: HostStatus | null
  caps: Capabilities | null
}) {
  const reachable = probe?.reachable ?? false
  const busy = probe?.gpu_busy ?? false
  return (
    <header className="h-14 shrink-0 flex items-center justify-between px-6 border-b border-slate-800/80 bg-slate-950/60 backdrop-blur">
      <h1 className="text-base font-semibold text-slate-100">{title}</h1>
      <div className="flex items-center gap-4">
        {caps?.version && (
          <span className="text-[11px] font-mono px-2 py-1 rounded-md bg-slate-900 border border-slate-800 text-slate-400">
            ModelOpt {caps.version}
          </span>
        )}
        <div className="flex items-center gap-2 text-[11px] text-slate-400">
          <span
            className={`w-2 h-2 rounded-full ${reachable ? (busy ? 'bg-amber-400 animate-pulse' : 'bg-emerald-400 animate-pulse') : 'bg-rose-500'}`}
          />
          <span className="font-mono">
            {probe?.gpu_mem || (reachable ? 'GPU 就绪' : '不可达')}
          </span>
        </div>
      </div>
    </header>
  )
}
