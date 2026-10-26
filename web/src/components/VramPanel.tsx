/**
 * 显存估算器滑面板（设计稿 §6）：从 TopBar「显存估算」按钮打开，
 * 右侧 fixed 滑入，宽度 360，黑底白字风格对齐模版开场。
 *
 * 输入 model / qformat / kv_qformat / ctx / batch → 调 /api/estimate
 * → 大字「放得下 ✓ / 还差 X GB ✗」+ 分项（权重/量化后/KV/余量）
 * + source 标注 + 真实显存总量与当前占用条。
 */
import { useEffect, useState } from 'react'
import { X } from 'lucide-react'
import { api } from '../api/endpoints'
import type { EstimateResult } from '../api/types'

interface Props {
  open: boolean
  onClose: () => void
  /** 从量化区带入的初值（可空，面板内自填） */
  model?: string
  qformat?: string
}

const KV_OPTIONS = ['none', 'nvfp4', 'fp8']

export default function VramPanel({ open, onClose, model, qformat }: Props) {
  const [m, setM] = useState(model ?? '')
  const [qf, setQf] = useState(qformat ?? '')
  const [kv, setKv] = useState(KV_OPTIONS[0])
  const [ctx, setCtx] = useState(2048)
  const [batch, setBatch] = useState(1)
  const [res, setRes] = useState<EstimateResult | null>(null)
  const [err, setErr] = useState('')
  const [loading, setLoading] = useState(false)

  // 打开时带入量化区当前值
  useEffect(() => {
    if (open) {
      if (model) setM(model)
      if (qformat) setQf(qformat)
    }
  }, [open, model, qformat])

  const run = async () => {
    if (!m.trim()) { setErr('模型不能为空'); return }
    setLoading(true)
    setErr('')
    setRes(null)
    try {
      const r = await api.estimate({
        model: m.trim(), qformat: qf, kv_qformat: kv, ctx, batch,
      })
      setRes(r)
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  if (!open) return null

  const total = res?.vram_total_mib ?? 0
  const used = res?.vram_used_mib ?? 0
  const free = Math.max(0, total - used)
  const usedPct = total > 0 ? (used / total) * 100 : 0

  const gb = (b: number) => (b / 1024 ** 3).toFixed(2)

  return (
    <>
      {/* 遮罩 */}
      <div className="fixed inset-0 bg-black/40 z-40" onClick={onClose} />
      <aside className="fixed top-0 right-0 h-full w-[360px] z-50 bg-slate-950 border-l border-slate-800 shadow-2xl flex flex-col">
        <div className="h-14 shrink-0 flex items-center justify-between px-4 border-b border-slate-800">
          <h2 className="text-sm font-semibold text-slate-100">显存估算</h2>
          <button onClick={onClose} className="p-1.5 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-slate-900">
            <X size={16} />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          <div className="grid grid-cols-2 gap-3">
            <label className="col-span-2 block">
              <span className="text-[11px] text-slate-500">模型</span>
              <input value={m} onChange={(e) => setM(e.target.value)}
                placeholder="Qwen/Qwen2.5-1.5B-Instruct"
                className="mt-1 w-full px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800 text-[12.5px] font-mono text-slate-200 focus:outline-none focus:border-cyan-500/50" />
            </label>
            <label className="block">
              <span className="text-[11px] text-slate-500">量化格式</span>
              <input value={qf} onChange={(e) => setQf(e.target.value)} placeholder="nvfp4"
                className="mt-1 w-full px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800 text-[12.5px] font-mono text-slate-200 focus:outline-none focus:border-cyan-500/50" />
            </label>
            <label className="block">
              <span className="text-[11px] text-slate-500">KV 格式</span>
              <select value={kv} onChange={(e) => setKv(e.target.value)}
                className="mt-1 w-full px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800 text-[12.5px] text-slate-200 focus:outline-none focus:border-cyan-500/50">
                {KV_OPTIONS.map((o) => <option key={o} value={o}>{o}</option>)}
              </select>
            </label>
            <label className="block">
              <span className="text-[11px] text-slate-500">ctx</span>
              <input type="number" min={1} value={ctx} onChange={(e) => setCtx(Math.max(1, Number(e.target.value) || 1))}
                className="mt-1 w-full px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800 text-[12.5px] font-mono text-slate-200 focus:outline-none focus:border-cyan-500/50" />
            </label>
            <label className="block">
              <span className="text-[11px] text-slate-500">batch</span>
              <input type="number" min={1} value={batch} onChange={(e) => setBatch(Math.max(1, Number(e.target.value) || 1))}
                className="mt-1 w-full px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800 text-[12.5px] font-mono text-slate-200 focus:outline-none focus:border-cyan-500/50" />
            </label>
          </div>

          <button onClick={() => void run()} disabled={loading}
            className="w-full flex items-center justify-center gap-2 px-4 py-2 rounded-lg bg-cyan-500/90 hover:bg-cyan-400 text-[#020617] text-[13px] font-semibold disabled:opacity-50 transition-colors">
            {loading ? '估算中…' : '估算显存'}
          </button>

          {err && (
            <div className="p-3 rounded-xl bg-rose-950/40 border border-rose-500/30 text-[12px] text-rose-300">{err}</div>
          )}

          {res && (
            <div className="space-y-4">
              {/* 大字结论 */}
              <div className={`p-4 rounded-2xl border ${res.fits
                ? 'bg-emerald-500/10 border-emerald-500/30'
                : 'bg-rose-500/10 border-rose-500/30'}`}>
                <div className={`text-2xl font-bold ${res.fits ? 'text-emerald-300' : 'text-rose-300'}`}>
                  {res.fits ? '放得下 ✓' : `还差 ${res.short_gib} GB ✗`}
                </div>
                <div className="text-[12px] text-slate-400 mt-1">
                  需要 {res.need_gib} GB · 空闲 {gb((free) * 1024 ** 2)} GB
                </div>
              </div>

              {/* 分项 */}
              <div className="grid grid-cols-2 gap-2 text-[12px]">
                <Row label="权重量" value={res.weight_bytes > 0 ? `${gb(res.weight_bytes)} GB` : '未知'} src={res.source} />
                <Row label="量化后" value={res.quant_bytes > 0 ? `${gb(res.quant_bytes)} GB` : '未知'} />
                <Row label="KV 缓存" value={res.kv_bytes > 0 ? `${gb(res.kv_bytes)} GB` : kv === 'none' ? '未启用' : '—'} />
                <Row label="空闲显存" value={`${(free / 1024).toFixed(1)} GB`} />
              </div>

              {res.note && (
                <div className="p-2.5 rounded-lg bg-amber-950/30 border border-amber-500/25 text-[11.5px] text-amber-300">
                  {res.note}
                </div>
              )}

              {/* 真实显存条 */}
              <div>
                <div className="flex justify-between text-[11px] text-slate-500 mb-1">
                  <span>当前占用 {used} / {total} MiB</span>
                  <span>{usedPct.toFixed(0)}%</span>
                </div>
                <div className="h-2 rounded-full bg-slate-900 overflow-hidden">
                  <div className={`h-full ${usedPct > 90 ? 'bg-rose-500' : usedPct > 70 ? 'bg-amber-400' : 'bg-emerald-400'}`}
                    style={{ width: `${usedPct}%` }} />
                </div>
              </div>
            </div>
          )}
        </div>
      </aside>
    </>
  )
}

function Row({ label, value, src }: { label: string; value: string; src?: string }) {
  return (
    <div className="p-2 rounded-lg bg-slate-900/60 border border-slate-800">
      <div className="text-[10.5px] text-slate-500">{label}{src ? ` · ${src}` : ''}</div>
      <div className="text-[13px] font-mono text-slate-200">{value}</div>
    </div>
  )
}
