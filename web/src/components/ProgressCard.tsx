/**
 * 进度卡：ProgressReport → 阶段名 + 进度条（pct 为 null 时显示「进行中 · detail」）
 * + item/detail 行 + error 红字；done 时绿色完成条。
 */
import type { ProgressReport } from '../api/types'

export default function ProgressCard({
  progress, state,
}: {
  progress: ProgressReport | null
  state: string | null
}) {
  const err = progress?.error || ''
  const done = progress?.done === true || state === 'done'

  return (
    <div className="p-4 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-3">
      {/* 阶段行 */}
      <div className="flex items-center justify-between">
        <div className="text-[13px] font-semibold text-slate-200">
          {progress?.stage || (state === 'running' || state === 'starting' ? '准备中…' : '等待进度')}
        </div>
        {progress?.pct != null ? (
          <span className="text-[12px] font-mono text-cyan-300">{progress.pct}%</span>
        ) : (
          <span className="text-[12px] text-slate-500">{progress?.detail || '进行中'}</span>
        )}
      </div>

      {/* 进度条 */}
      <div className="h-1.5 rounded-full bg-slate-800 overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-500 ${
            err ? 'bg-rose-500' : done ? 'bg-emerald-500' : 'bg-gradient-to-r from-violet-500 to-cyan-400'
          }`}
          style={{ width: pct(progress) }}
        />
      </div>

      {/* item / detail / error */}
      {progress?.item && (
        <div className="text-[12px] text-slate-400 truncate" title={progress.item}>{progress.item}</div>
      )}
      {progress?.detail && !progress.pct && (
        <div className="text-[12px] text-slate-500 truncate">{progress.detail}</div>
      )}
      {err && (
        <div className="text-[12px] text-rose-300 bg-rose-950/40 border border-rose-500/30 rounded-lg px-3 py-2 whitespace-pre-wrap">
          {err}
        </div>
      )}
      {done && !err && (
        <div className="text-[12px] text-emerald-300">✓ 完成</div>
      )}
    </div>
  )
}

function pct(p: ProgressReport | null): string {
  if (!p || p.pct == null) return p?.detail ? '6%' : '2%'
  return `${Math.min(100, Math.max(0, p.pct))}%`
}
