/**
 * 日志面板：等宽字体、自动滚底（用户上滚则暂停），SSE 增量刷新。
 */
import { useEffect, useRef, useState } from 'react'

export default function LogPanel({
  text, height = 320,
}: {
  text: string
  height?: number
}) {
  const boxRef = useRef<HTMLDivElement>(null)
  const [autoScroll, setAutoScroll] = useState(true)

  // 滚离底部超过 48px 视为「用户在往上翻」，暂停自动滚底
  const onScroll = () => {
    const el = boxRef.current
    if (!el) return
    const gap = el.scrollHeight - el.scrollTop - el.clientHeight
    setAutoScroll(gap < 48)
  }

  useEffect(() => {
    const el = boxRef.current
    if (el && autoScroll) el.scrollTop = el.scrollHeight
  }, [text, autoScroll])

  return (
    <div className="rounded-2xl border border-slate-800 bg-slate-950 overflow-hidden">
      <div className="flex items-center justify-between px-3 py-1.5 border-b border-slate-800/80 bg-slate-900/50">
        <span className="text-[11px] font-semibold text-slate-400 tracking-wide">日志</span>
        <div className="flex items-center gap-3">
          {!autoScroll && (
            <button
              onClick={() => {
                setAutoScroll(true)
                const el = boxRef.current
                if (el) el.scrollTop = el.scrollHeight
              }}
              className="text-[11px] text-cyan-400 hover:text-cyan-300"
            >
              回到最新
            </button>
          )}
          <span className="text-[11px] text-slate-600 font-mono">{text ? text.split('\n').length : 0} 行</span>
        </div>
      </div>
      <div
        ref={boxRef}
        onScroll={onScroll}
        style={{ height }}
        className="overflow-y-auto px-3 py-2 font-mono text-[11.5px] leading-relaxed text-slate-300 whitespace-pre-wrap break-all"
      >
        {text || <span className="text-slate-600">（暂无日志）</span>}
      </div>
    </div>
  )
}
