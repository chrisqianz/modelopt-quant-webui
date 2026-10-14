/**
 * 任务面板：GET /api/jobs（active + history）列表，点进详情订阅该 job 的 SSE 续看。
 */
import { useCallback, useEffect, useState } from 'react'
import { api } from '../api/endpoints'
import type { JobDetail, JobSummary } from '../api/types'
import { useJobStream } from '../hooks/useJobStream'
import ProgressCard from './ProgressCard'
import LogPanel from './LogPanel'
import { useToast } from './ToastProvider'

const STATE_BADGE: Record<string, string> = {
  starting: 'bg-slate-800 text-slate-300 border-slate-700',
  running: 'bg-cyan-950/60 text-cyan-300 border-cyan-500/40',
  done: 'bg-emerald-950/60 text-emerald-300 border-emerald-500/40',
  failed: 'bg-rose-950/60 text-rose-300 border-rose-500/40',
  aborted: 'bg-amber-950/60 text-amber-300 border-amber-500/40',
}

function Badge({ state }: { state: string }) {
  return (
    <span className={`inline-block px-2 py-0.5 rounded-md text-[11px] font-medium border ${STATE_BADGE[state] ?? 'bg-slate-800 text-slate-300 border-slate-700'}`}>
      {state}
    </span>
  )
}

export default function JobsPanel() {
  const [active, setActive] = useState<JobSummary[]>([])
  const [history, setHistory] = useState<JobSummary[]>([])
  const [selected, setSelected] = useState<JobSummary | null>(null)
  const [detail, setDetail] = useState<JobDetail | null>(null)
  const [err, setErr] = useState('')
  const toast = useToast()
  const stream = useJobStream(selected?.id ?? null, selected?.state ?? null)

  const reload = useCallback(async () => {
    try {
      const r = await api.jobs()
      setActive(r.active)
      setHistory(r.history)
      setErr('')
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    }
  }, [])

  useEffect(() => { void reload() }, [reload])
  // 有活跃任务时 5s 轮询一次（新 job 出现 / state 变化）
  useEffect(() => {
    if (!active.length && !selected) return
    const t = setInterval(() => void reload(), 5000)
    return () => clearInterval(t)
  }, [active.length, selected, reload])

  // 选中 job 是终态时拉快照（终态 job 的 SSE 已短路：state 事件只在 poller
  // 存活时广播，终态后新订阅者永远收不到 ⇒ 流永久挂起，日志/进度只走快照）
  useEffect(() => {
    if (!selected) { setDetail(null); return }
    if (!['done', 'failed', 'aborted'].includes(selected.state)) { setDetail(null); return }
    let cancelled = false
    void (async () => {
      try {
        const d = await api.jobDetail(selected.id)
        if (!cancelled) setDetail(d)
      } catch { /* 快照拉失败不阻塞展示 */ }
    })()
    return () => { cancelled = true }
  }, [selected])

  const onAbort = async (id: string) => {
    try {
      await api.abort(id)
      toast.show('中止信号已发送')
      setSelected((s) => (s && s.id === id ? { ...s, state: 'aborted' } : s))
      void reload()
    } catch (e) {
      toast.show(e instanceof Error ? e.message : String(e), 'error')
    }
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold text-slate-200">任务</h2>
        <button
          onClick={() => void reload()}
          className="text-[12px] text-slate-400 hover:text-slate-200"
        >
          刷新
        </button>
      </div>
      {err && <div className="text-[12px] text-rose-300 font-mono">{err}</div>}

      {/* 列表 */}
      <div className="rounded-2xl border border-slate-800 bg-slate-900/40 divide-y divide-slate-800/80">
        {active.length === 0 && history.length === 0 && (
          <div className="p-4 text-[12px] text-slate-600">暂无任务</div>
        )}
        {[...active, ...history].map((j) => (
          <button
            key={j.id}
            onClick={() => setSelected(j)}
            className={`w-full flex items-center gap-3 px-4 py-2.5 text-left hover:bg-slate-900/70 transition-colors ${selected?.id === j.id ? 'bg-slate-900/70' : ''}`}
          >
            <Badge state={j.state} />
            <span className="text-[12px] text-slate-300 font-medium w-20 shrink-0">{j.kind}</span>
            <span className="text-[11.5px] text-slate-500 font-mono truncate flex-1" title={j.cmd_preview}>
              {j.cmd_preview || j.kind}
            </span>
            <span className="text-[11px] text-slate-600 font-mono shrink-0">{j.started_at}</span>
          </button>
        ))}
      </div>

      {/* 详情：订阅该 job 的 SSE（断线重连自动带 Last-Event-ID） */}
      {selected && (
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <div className="text-[12px] text-slate-400 font-mono truncate">{selected.cmd_preview}</div>
            {(selected.state === 'running' || selected.state === 'starting') && (
              <button
                onClick={() => void onAbort(selected.id)}
                className="shrink-0 ml-3 px-3 py-1 rounded-lg text-[11px] font-medium bg-rose-600/80 text-white hover:bg-rose-500"
              >
                中止
              </button>
            )}
          </div>
          <ProgressCard
            progress={stream.progress ?? detail?.progress ?? null}
            state={stream.state ?? detail?.state ?? selected.state}
          />
          <LogPanel text={stream.log || detail?.log_tail || ''} />
          <div className="text-[11px] text-slate-600">
            {['done', 'failed', 'aborted'].includes(selected.state)
              ? `终态 job（state=${selected.state}），日志/进度来自快照`
              : stream.finished
                ? `流已结束（state=${stream.state ?? selected.state}）`
                : stream.connected
                  ? 'SSE 已连接，实时接收中…'
                  : 'SSE 连接中 / 已断线（自动重连）…'}
          </div>
        </div>
      )}
    </div>
  )
}
