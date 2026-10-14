/**
 * 产物库区（C1 最小版）：GET /api/artifacts → 卡片列表（dir/model/ts/size/state 徽章）。
 * vLLM 启动命令模板与完整产物库 = Plan 4，顶部只放占位条。
 */
import { useEffect, useState } from 'react'
import { Boxes, RefreshCw } from 'lucide-react'
import { api } from '../api/endpoints'
import type { Artifact } from '../api/types'

function stateBadge(state: string) {
  switch (state) {
    case 'done':
      return 'bg-emerald-500/15 text-emerald-300'
    case 'failed':
      return 'bg-rose-500/15 text-rose-300'
    default:
      return 'bg-slate-500/15 text-slate-400' // manual / 未知
  }
}

function stateLabel(state: string) {
  switch (state) {
    case 'done':
      return '完成'
    case 'failed':
      return '失败'
    case 'running':
      return '运行中'
    case 'aborted':
      return '已中止'
    default:
      return state
  }
}

export default function ArtifactsView() {
  const [list, setList] = useState<Artifact[] | null>(null)
  const [err, setErr] = useState('')
  const [loading, setLoading] = useState(false)

  const load = async () => {
    setLoading(true)
    setErr('')
    try {
      const r = await api.artifacts()
      setList(r.artifacts)
    } catch (e) {
      setList(null)
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { void load() }, [])

  return (
    <div className="max-w-4xl space-y-4">
      {/* 占位条：完整产物库（vLLM 命令模板/下载/清理）在 Plan 4 */}
      <div className="px-4 py-2.5 rounded-xl bg-cyan-500/8 border border-cyan-500/20 text-[12px] text-cyan-300/90">
        vLLM 启动命令模板、产物下载与清理 = Plan 4，当前仅展示 task_dir 产物列表。
      </div>

      <section className="p-4 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-3">
        <div className="flex items-center justify-between">
          <h3 className="text-[12px] font-semibold text-slate-400 tracking-wide uppercase">
            task_dir 产物
          </h3>
          <button
            onClick={() => void load()}
            disabled={loading}
            className="flex items-center gap-1 text-[12px] text-cyan-400 hover:text-cyan-300 disabled:opacity-40"
          >
            <RefreshCw size={12} className={loading ? 'animate-spin' : ''} /> 刷新
          </button>
        </div>

        {err && (
          <div className="p-3 rounded-xl bg-rose-950/40 border border-rose-500/30 text-[12px] text-rose-300">
            {err}
          </div>
        )}

        {list === null && !err ? (
          <div className="flex items-center gap-2 text-[12.5px] text-slate-500">
            <Boxes size={14} /> 加载中…
          </div>
        ) : list && list.length === 0 ? (
          <div className="text-[12.5px] text-slate-500">
            暂无产物（task_dir 下没有带 run.log 的目录）—— 量化跑完一次后刷新
          </div>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {list!.map((a) => (
              <div key={a.dir} className="rounded-xl border border-slate-800 bg-slate-950/60 p-3 space-y-1.5">
                <div className="flex items-center justify-between gap-2">
                  <span className="font-mono text-[12.5px] text-slate-200 truncate" title={a.dir}>
                    {a.dir}
                  </span>
                  <span
                    className={`px-2 py-0.5 rounded-md text-[11px] shrink-0 ${stateBadge(a.state)}`}
                  >
                    {stateLabel(a.state)}
                  </span>
                </div>
                <div className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-[11.5px]">
                  <span className="text-slate-500">模型</span>
                  <span className="font-mono text-slate-400 truncate" title={a.model}>
                    {a.model || '—'}
                  </span>
                  <span className="text-slate-500">时间</span>
                  <span className="font-mono text-slate-400">{a.ts || '—'}</span>
                  <span className="text-slate-500">大小</span>
                  <span className="font-mono text-slate-400">{a.size || '—'}</span>
                </div>
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  )
}
