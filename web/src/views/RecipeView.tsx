/**
 * 对配方区：选一个已完成的 quant job + 官方 summary 路径（可空）
 * → POST /api/recipe/diff → 本产物 summary 卡 + 量化组列表 + diff 文本 + 文件清单。
 */
import { useState } from 'react'
import { GitCompare, Loader2 } from 'lucide-react'
import { api } from '../api/endpoints'
import type { Bootstrap, JobSummary, RecipeDiffResult } from '../api/types'

const inputCls =
  'w-full px-3 py-1.5 rounded-lg bg-slate-950 border border-slate-800 text-[13px] text-slate-100 outline-none focus:border-violet-500/60 placeholder:text-slate-600'

export default function RecipeView({ data }: { data: Bootstrap | null }) {
  const jobs = data ? [...data.jobs] : []
  const [jobId, setJobId] = useState('')
  const [official, setOfficial] = useState('')
  const [result, setResult] = useState<RecipeDiffResult | null>(null)
  const [err, setErr] = useState('')
  const [busy, setBusy] = useState(false)

  const doDiff = async () => {
    setBusy(true)
    setErr('')
    try {
      setResult(await api.recipeDiff(jobId, official.trim()))
    } catch (e) {
      setResult(null)
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="w-full space-y-4">
      <section className="p-4 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-3">
        <h3 className="text-[12px] font-semibold text-slate-400 tracking-wide uppercase">
          对照来源
        </h3>
        <div className="grid grid-cols-2 gap-x-4 gap-y-3">
          <label className="block space-y-1">
            <span className="text-[12px] text-slate-400">量化 job（已完成/失败均可）</span>
            <select value={jobId} onChange={(e) => setJobId(e.target.value)} className={inputCls}>
              <option value="">— 选择 job —</option>
              {jobs
                .filter((j) => j.kind === 'quant')
                .map((j) => (
                  <option key={j.id} value={j.id}>
                    {j.id}（{stateLabel(j)}）
                  </option>
                ))}
            </select>
          </label>
          <label className="block space-y-1">
            <span className="text-[12px] text-slate-400">
              官方 summary 路径 / HF model id（可空 = 只看本产物）
            </span>
            <input
              value={official}
              onChange={(e) => setOfficial(e.target.value)}
              placeholder="nvidia/Qwen3-8B-NVFP4 或 /abs/path/config.json"
              className={inputCls}
            />
          </label>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={() => void doDiff()}
            disabled={!jobId || busy}
            className="flex items-center gap-1.5 px-5 py-2 rounded-xl text-[13px] font-medium bg-violet-600 text-white hover:bg-violet-500 disabled:opacity-40"
          >
            {busy ? <Loader2 size={14} className="animate-spin" /> : <GitCompare size={14} />}
            对比
          </button>
          {jobs.filter((j) => j.kind === 'quant').length === 0 && (
            <span className="text-[12px] text-slate-500">
              还没有 quant job —— 先到「模型量化」区跑一次
            </span>
          )}
        </div>
      </section>

      {err && (
        <div className="p-3 rounded-xl bg-rose-950/40 border border-rose-500/30 text-[12px] text-rose-300 font-mono whitespace-pre-wrap">
          {err}
        </div>
      )}

      {result && (
        <div className="space-y-4">
          {/* 本产物 summary 卡 */}
          <section className="p-4 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-3">
            <h3 className="text-[12px] font-semibold text-slate-400 tracking-wide uppercase">
              本产物（{result.summary.source || '导出目录'}）
            </h3>
            {result.summary.error ? (
              <div className="text-[12.5px] text-amber-300">{result.summary.error}</div>
            ) : (
              <>
                <div className="flex items-center gap-2">
                  <span className="text-[13px] text-slate-300">quant_method:</span>
                  <span className="px-2 py-0.5 rounded-md bg-violet-500/15 text-violet-300 text-[12px] font-mono">
                    {result.summary.quant_method || '—'}
                  </span>
                </div>
                {result.groups.length > 0 && (
                  <div className="overflow-x-auto rounded-xl border border-slate-800">
                    <table className="w-full text-[12.5px]">
                      <thead>
                        <tr className="bg-slate-900/70 text-slate-400">
                          <th className="px-3 py-2 text-left font-semibold">组</th>
                          <th className="px-3 py-2 text-left font-semibold">bits</th>
                          <th className="px-3 py-2 text-left font-semibold">targets</th>
                          <th className="px-3 py-2 text-left font-semibold">示例</th>
                        </tr>
                      </thead>
                      <tbody>
                        {result.groups.map((g, i) => (
                          <tr key={i} className="border-t border-slate-800/60 font-mono">
                            <td className="px-3 py-1.5 text-slate-200">{g.name || '—'}</td>
                            <td className="px-3 py-1.5 text-slate-300">{g.bits}</td>
                            <td className="px-3 py-1.5 text-slate-300">
                              {g.count}
                              {g.more && <span className="text-slate-500"> {g.more}</span>}
                            </td>
                            <td className="px-3 py-1.5 text-slate-500">
                              {(g.samples ?? []).join(', ')}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
                {result.summary.files && (
                  <pre className="p-3 rounded-xl bg-slate-950 border border-slate-800/80 font-mono text-[11.5px] text-slate-400 whitespace-pre-wrap max-h-56 overflow-y-auto">
                    {result.summary.files}
                  </pre>
                )}
              </>
            )}
          </section>

          {/* 官方对照 diff */}
          {official.trim() && (
            <section className="p-4 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-3">
              <h3 className="text-[12px] font-semibold text-slate-400 tracking-wide uppercase">
                与官方对照（{official.trim()}）
              </h3>
              {result.diff ? (
                <pre className="p-3 rounded-xl bg-slate-950 border border-slate-800/80 font-mono text-[11.5px] text-slate-300 whitespace-pre-wrap max-h-72 overflow-y-auto">
                  {result.diff}
                </pre>
              ) : (
                <div className="text-[12.5px] text-slate-500">（无差异）</div>
              )}
            </section>
          )}
        </div>
      )}
    </div>
  )
}

function stateLabel(j: JobSummary): string {
  const map: Record<string, string> = {
    starting: '启动中', running: '运行中', done: '完成', failed: '失败', aborted: '已中止',
  }
  return `${map[j.state] ?? j.state} · ${new Date(j.started_at * 1000).toLocaleString()}`
}
