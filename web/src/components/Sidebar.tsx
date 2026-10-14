/**
 * 侧栏：logo + 5 区导航 + 底部 GPU 状态卡（真实 probe 数据，无 mock）。
 */
import { Activity, Database, Settings2, SlidersHorizontal, TestTubes } from 'lucide-react'
import type { HostStatus } from '../api/types'

export type ViewKey = 'quantize' | 'eval' | 'recipe' | 'artifacts' | 'settings'

export const VIEWS: { key: ViewKey; label: string; icon: typeof SlidersHorizontal }[] = [
  { key: 'quantize', label: '模型量化', icon: SlidersHorizontal },
  { key: 'eval', label: '性能与精度评测', icon: TestTubes },
  { key: 'recipe', label: '配方对照', icon: Activity },
  { key: 'artifacts', label: '模型仓库与导出', icon: Database },
  { key: 'settings', label: '系统设置与日志', icon: Settings2 },
]

function GpuStatusCard({ probe }: { probe: HostStatus | null }) {
  const busy = probe?.gpu_busy ?? false
  return (
    <div className="mx-3 mb-3 p-3 rounded-xl bg-slate-900/70 border border-slate-800 space-y-2">
      <div className="flex items-center gap-2">
        <span className={`w-2 h-2 rounded-full ${probe?.reachable ? (busy ? 'bg-amber-400' : 'bg-emerald-400') : 'bg-rose-500'}`} />
        <span className="text-xs font-semibold text-slate-300 truncate">{probe?.gpu || (probe?.reachable ? 'GPU' : '不可达')}</span>
      </div>
      {probe ? (
        <>
          <div>
            <div className="flex justify-between text-[11px] text-slate-500 mb-1">
              <span>显存</span>
              <span className="font-mono text-slate-400">{probe.gpu_mem || '—'}</span>
            </div>
            <div className="h-1.5 rounded-full bg-slate-800 overflow-hidden">
              <div
                className={`h-full rounded-full ${busy ? 'bg-amber-500' : 'bg-cyan-500'}`}
                style={{ width: gpuMemPct(probe.gpu_mem) }}
              />
            </div>
          </div>
          <div className="flex justify-between text-[11px] text-slate-500">
            <span>利用率 {probe.gpu_util != null ? `${probe.gpu_util}%` : '—'}</span>
            <span>磁盘 {probe.disk_free || '—'}</span>
          </div>
        </>
      ) : (
        <div className="text-[11px] text-slate-600">探测中…</div>
      )}
    </div>
  )
}

/** 从 "22263 MiB / 32690 MiB" 里算占用百分比（取不到则 0） */
function gpuMemPct(gpuMem: string): string {
  const m = gpuMem.match(/([\d.]+)\s*MiB\s*\/\s*([\d.]+)\s*MiB/)
  if (!m) return '0%'
  const used = parseFloat(m[1]); const total = parseFloat(m[2])
  if (!total) return '0%'
  return `${Math.round((used / total) * 100)}%`
}

export default function Sidebar({
  view, onNavigate, probe,
}: {
  view: ViewKey
  onNavigate: (v: ViewKey) => void
  probe: HostStatus | null
}) {
  return (
    <aside className="w-60 shrink-0 h-full flex flex-col bg-slate-950 border-r border-slate-800/80">
      <div className="px-5 py-5 flex items-center gap-2.5">
        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-violet-600 via-indigo-500 to-cyan-500 flex items-center justify-center text-white text-xs font-bold">
          Q
        </div>
        <div>
          <div className="text-sm font-semibold tracking-wide">ModelOpt Quant</div>
          <div className="text-[10px] text-slate-500">量化 · 评测 · 交付</div>
        </div>
      </div>

      <nav className="flex-1 px-3 space-y-1 overflow-y-auto">
        {VIEWS.map((v) => {
          const Icon = v.icon
          const active = view === v.key
          return (
            <button
              key={v.key}
              onClick={() => onNavigate(v.key)}
              className={`w-full flex items-center gap-2.5 px-3 py-2.5 rounded-xl text-[13px] font-medium transition-all ${
                active
                  ? 'bg-violet-950/40 text-violet-200 border border-violet-500/40'
                  : 'text-slate-400 border border-transparent hover:text-slate-200 hover:bg-slate-900/60'
              }`}
            >
              <Icon size={15} className={active ? 'text-violet-400' : 'text-slate-500'} />
              {v.label}
            </button>
          )
        })}
      </nav>

      <GpuStatusCard probe={probe} />
    </aside>
  )
}
