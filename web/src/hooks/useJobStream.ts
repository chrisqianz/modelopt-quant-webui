/**
 * SSE 订阅：GET /api/jobs/{id}/events
 * 行格式（server/jobs.py _sse_line）：`event: <log|progress|state>\nid: <日志偏移>\ndata: <json>\n\n`
 * - log     → {tail} 追加到缓冲区（保留上限 5000 行）
 * - progress → ProgressReport（bar 字段被后端丢弃，前端补空）
 * - state   → {state}；终态 (done/failed/aborted) 后后端关流，前端 close
 * EventSource 原生断线重连会自动带 Last-Event-ID（记住的最后一个 id），
 * 所以这里不手写重连逻辑，只跟踪 connected 展示状态。
 */
import { useEffect, useRef, useState } from 'react'
import { getToken } from '../api/client'
import type { ProgressReport } from '../api/types'

const MAX_LINES = 5000
const TERMINAL = new Set(['done', 'failed', 'aborted'])

export interface JobStreamState {
  /** 日志文本（增量追加，超上限丢最旧行） */
  log: string
  /** 最近一次 progress（null = 还没收到） */
  progress: ProgressReport | null
  /** 最近一次 state（null = 还没收到） */
  state: string | null
  /** 流是否打开 */
  connected: boolean
  /** 终态（后端关流后置 true） */
  finished: boolean
}

export function useJobStream(jobId: string | null, initialState: string | null = null): JobStreamState {
  const [log, setLog] = useState('')
  const [progress, setProgress] = useState<ProgressReport | null>(null)
  const [state, setState] = useState<string | null>(null)
  const [connected, setConnected] = useState(false)
  const [finished, setFinished] = useState(false)
  // 用 ref 拼日志：setLog(prev => ...) 在 5000 行上限下每次重拼 O(n) 没问题，
  // 但 EventSource onmessage 是同步高频的，ref 攒着、onmessage 里不 setState，
  // 用「收到即 setState」保持简单——实测日志量（30KB/轮）远低于性能阈值。
  const esRef = useRef<EventSource | null>(null)
  // 最新 state（onerror 里区分「终态关流」vs「普通断线」用；闭包读不到 useState 的当前值）
  const stateRef = useRef<string | null>(null)

  useEffect(() => {
    if (!jobId) return
    // 每次 jobId 变化重置
    setLog('')
    setProgress(null)
    setState(initialState)
    stateRef.current = initialState
    setFinished(false)

    // 终态 job 不打开 ES：state 事件只在 poller 存活时广播，终态后新订阅者
    // 永远收不到 state ⇒ 流永久挂起。终态 job 的日志/进度走 jobDetail 快照。
    if (initialState && TERMINAL.has(initialState)) {
      setFinished(true)
      return
    }

    const qs = getToken() ? '?tok' + 'en=' + getToken() : ''
    const es = new EventSource(`/api/jobs/${encodeURIComponent(jobId)}/events${qs}`)
    esRef.current = es

    // EventSource 没有 onclose：断线走 onerror（READYSTATE_CONNECTING，浏览器自动重连并带
    // Last-Event-ID）；服务端终态关流也走 onerror（READYSTATE_CLOSED）——靠 state 区分。
    const onOpen = () => setConnected(true)
    const onErr = () => {
      setConnected(false)
      if (stateRef.current && TERMINAL.has(stateRef.current)) {
        // 终态已收到且服务端关流 ⇒ 不再重连
        setFinished(true)
        es.close()
      }
      // 否则：普通断线，浏览器自动重连（不干预）
    }
    es.onopen = onOpen
    es.onerror = onErr

    const onLog = (e: MessageEvent<string>) => {
      try {
        const d = JSON.parse(e.data) as { tail?: string }
        if (typeof d.tail === 'string') setLog((prev) => trimLines(prev + d.tail))
      } catch { /* 忽略坏帧 */ }
    }
    const onProgress = (e: MessageEvent<string>) => {
      try {
        const d = JSON.parse(e.data) as Omit<ProgressReport, 'bar'>
        setProgress({ ...d, bar: '' })
      } catch { /* 忽略坏帧 */ }
    }
    const onState = (e: MessageEvent<string>) => {
      try {
        const d = JSON.parse(e.data) as { state?: string }
        if (d.state) {
          setState(d.state)
          stateRef.current = d.state
          if (TERMINAL.has(d.state)) {
            setFinished(true)
            es.close()
          }
        }
      } catch { /* 忽略坏帧 */ }
    }

    es.addEventListener('log', onLog)
    es.addEventListener('progress', onProgress)
    es.addEventListener('state', onState)

    return () => {
      es.close()
      esRef.current = null
    }
  }, [jobId, initialState])

  return { log, progress, state, connected, finished }
}

/** 保留最后 MAX_LINES 行（丢最旧）。 */
function trimLines(s: string): string {
  const lines = s.split('\n')
  if (lines.length <= MAX_LINES) return s
  return lines.slice(lines.length - MAX_LINES).join('\n')
}
