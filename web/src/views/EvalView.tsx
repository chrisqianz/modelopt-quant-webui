/**
 * 评测区：表单（quant/baseline/tasks/预设/tp/port/batch）
 * → POST /api/eval → 进度区（pkill 提示的 vllm serve 中止）
 * → 结果区（GET /api/eval/results：表格 + recharts 分组柱状，真实维度 ≥3 才出图）。
 */
import { useEffect, useMemo, useState } from 'react'
import { Play, RefreshCw, Sparkles } from 'lucide-react'
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { api } from '../api/endpoints'
import type { EvalResults } from '../api/types'
import { useToast } from '../components/ToastProvider'
import ProgressCard from '../components/ProgressCard'
import LogPanel from '../components/LogPanel'
import { useJobStream } from '../hooks/useJobStream'
import { EVAL_NOTE, EVAL_PRESETS, EVAL_TASKS } from '../data/evalTasks'

const inputCls =
  'w-full px-3 py-1.5 rounded-lg bg-slate-950 border border-slate-800 text-[13px] text-slate-100 outline-none focus:border-violet-500/60 placeholder:text-slate-600'

export default function EvalView() {
  const [quant, setQuant] = useState('')
  const [baseline, setBaseline] = useState('')
  const [tasks, setTasks] = useState<string[]>([])
  const [tp, setTp] = useState('1')
  const [port, setPort] = useState('8000')
  const [batch, setBatch] = useState('auto')
  const [busy, setBusy] = useState(false)
  const [jobId, setJobId] = useState<string | null>(null)
  const [jobState, setJobState] = useState<string | null>(null)
  const [results, setResults] = useState<EvalResults | null>(null)
  const [resultsNote, setResultsNote] = useState('')
  const toast = useToast()
  const stream = useJobStream(jobId, jobState)

  // 有 quant 路径时拉一次「建议 baseline」（config.json 的 _name_or_path）
  const suggestBaseline = async () => {
    try {
      const r = await api.evalBaseline(quant.trim())
      if (r.baseline) {
        setBaseline(r.baseline)
      } else {
        toast.show(r.message || 'config.json 里没有基线模型名，请手填')
      }
    } catch (e) {
      toast.show(e instanceof Error ? e.message : String(e), 'error')
    }
  }

  const applyPreset = (name: string) => {
    setTasks([...(EVAL_PRESETS[name] ?? [])])
  }

  const toggleTask = (t: string) =>
    setTasks((prev) => (prev.includes(t) ? prev.filter((x) => x !== t) : [...prev, t]))

  const canRun = !!quant.trim() && tasks.length > 0

  const doRun = async () => {
    setBusy(true)
    try {
      const r = await api.eval({
        quant: quant.trim(),
        baseline: baseline.trim(),
        tp: parseInt(tp, 10) || 1,
        port: parseInt(port, 10) || 8000,
        batch: batch.trim() || 'auto',
        tasks,
      })
      setJobId(r.job_id)
      setJobState(r.job.state)
    } catch (e) {
      toast.show(e instanceof Error ? e.message : String(e), 'error')
    } finally {
      setBusy(false)
    }
  }

  const loadResults = async () => {
    setResultsNote('')
    try {
      const r = await api.evalResults()
      setResults(r)
    } catch (e) {
      setResults(null)
      setResultsNote(e instanceof Error ? e.message : String(e))
    }
  }

  // 进入页面先拉一次现有结果（上次跑过就有）
  useEffect(() => { void loadResults() }, [])

  const onAbort = async () => {
    if (!jobId) return
    try {
      await api.abort(jobId)
      toast.show('中止信号已发送（vllm serve 进程会被一起停掉）')
      setJobState('aborted')
    } catch (e) {
      toast.show(e instanceof Error ? e.message : String(e), 'error')
    }
  }

  const running = jobId && (jobState === 'running' || jobState === 'starting' ||
    (stream.state === 'running' || stream.state === 'starting'))

  // 图表数据：按任务聚合（每任务取第一行指标），量化 vs 基线；真实维度 ≥3 才出图
  const chart = useMemo(() => {
    const byTask = new Map<string, { task: string; quant: number; base: number | null; delta: number | null }>()
    if (results) {
      for (const row of results.rows) {
        const [model, task, , q, b] = row
        if (!model || model === '模型') continue
        if (!byTask.has(task)) {
          const qv = parseFloat(q)
          const bv = b ? parseFloat(b) : null
          const dv = !isNaN(qv) && bv !== null && !isNaN(bv) ? qv - bv : null
          byTask.set(task, {
            task, quant: qv,
            base: bv,
            delta: dv,
          })
        }
      }
    }
    return [...byTask.values()]
  }, [results])

  const chartOk = chart.length >= 3

  return (
    <div className="max-w-4xl space-y-4">
      {jobId && running ? (
        <div className="space-y-3">
          <ProgressCard progress={stream.progress} state={stream.state ?? jobState} />
          <LogPanel text={stream.log} />
          <div className="flex justify-end">
            <button
              onClick={() => void onAbort()}
              className="px-4 py-1.5 rounded-lg text-[12px] font-medium bg-rose-600/80 text-white hover:bg-rose-500"
            >
              中止任务（会停掉 vllm serve）
            </button>
          </div>
        </div>
      ) : (
        <>
          <Card title="评测对象">
            <div className="grid grid-cols-2 gap-x-4 gap-y-3">
              <Field label="量化 checkpoint 路径">
                <input value={quant} onChange={(e) => setQuant(e.target.value)}
                  placeholder="~/quant-webui-tasks/xxx/out" className={inputCls} />
              </Field>
              <Field label="基线模型（可空 = 跳过基线评测）">
                <div className="flex gap-2">
                  <input value={baseline} onChange={(e) => setBaseline(e.target.value)}
                    placeholder="Qwen/Qwen2.5-1.5B-Instruct" className={inputCls} />
                  <button
                    onClick={() => void suggestBaseline()}
                    disabled={!quant.trim()}
                    title="从 config.json 抽基线模型名"
                    className="flex items-center gap-1 px-2.5 rounded-lg text-[12px] text-cyan-400 hover:text-cyan-300 disabled:opacity-40 shrink-0"
                  >
                    <Sparkles size={13} /> 建议
                  </button>
                </div>
              </Field>
            </div>
          </Card>

          <Card title="评测任务（lm-eval-harness）">
            <div className="space-y-3">
              <div className="flex flex-wrap gap-2">
                {Object.keys(EVAL_PRESETS).map((name) => (
                  <button
                    key={name}
                    onClick={() => applyPreset(name)}
                    className="px-2.5 py-1 rounded-lg text-[12px] bg-slate-800 text-slate-300 hover:bg-slate-700"
                  >
                    {name}
                  </button>
                ))}
              </div>
              <div className="grid grid-cols-2 gap-x-4 gap-y-1.5">
                {EVAL_TASKS.map((t) => (
                  <label key={t} className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={tasks.includes(t)}
                      onChange={() => toggleTask(t)}
                      className="accent-violet-500"
                    />
                    <span className="text-[12.5px] font-mono text-slate-300">{t}</span>
                  </label>
                ))}
              </div>
              <p className="text-[11.5px] text-slate-500 leading-relaxed">{EVAL_NOTE}</p>
            </div>
          </Card>

          <Card title="服务参数">
            <div className="grid grid-cols-3 gap-x-4 gap-y-3">
              <Field label="tp（tensor parallel）">
                <input value={tp} onChange={(e) => setTp(e.target.value)} className={inputCls} />
              </Field>
              <Field label="port">
                <input value={port} onChange={(e) => setPort(e.target.value)} className={inputCls} />
              </Field>
              <Field label="batch_size">
                <input value={batch} onChange={(e) => setBatch(e.target.value)} className={inputCls} />
              </Field>
            </div>
          </Card>

          <div className="flex items-center gap-3">
            <button
              onClick={() => void doRun()}
              disabled={!canRun || busy}
              className="flex items-center gap-1.5 px-5 py-2 rounded-xl text-[13px] font-medium bg-violet-600 text-white hover:bg-violet-500 disabled:opacity-40"
            >
              <Play size={14} /> 运行评测
            </button>
            {!canRun && (
              <span className="text-[12px] text-slate-500">需填量化路径 + 选至少一个任务</span>
            )}
            <span className="text-[11.5px] text-amber-400/80">
              注意：脚本会 pkill 现有 vllm serve 再部署
            </span>
          </div>

          {jobId && !running && stream.finished && (
            <div className="space-y-3">
              <ProgressCard progress={stream.progress} state={stream.state ?? jobState} />
              <LogPanel text={stream.log} />
              <button
                onClick={() => { setJobId(null); setJobState(null) }}
                className="text-[12px] text-slate-400 hover:text-slate-200"
              >
                ← 返回表单
              </button>
            </div>
          )}
        </>
      )}

      {/* 结果区 */}
      <section className="p-4 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-3">
        <div className="flex items-center justify-between">
          <h3 className="text-[12px] font-semibold text-slate-400 tracking-wide uppercase">
            结果（最近一次评测）
          </h3>
          <button
            onClick={() => void loadResults()}
            className="flex items-center gap-1 text-[12px] text-cyan-400 hover:text-cyan-300"
          >
            <RefreshCw size={12} /> 刷新
          </button>
        </div>
        {resultsNote && (
          <div className="p-3 rounded-xl bg-rose-950/40 border border-rose-500/30 text-[12px] text-rose-300">
            {resultsNote}
          </div>
        )}
        {!results || results.rows.length === 0 ? (
          <div className="text-[12.5px] text-slate-500">
            {results?.note || '（无结果文件）'} — 跑完一次评测后刷新
          </div>
        ) : (
          <div className="space-y-4">
            <div className="overflow-x-auto rounded-xl border border-slate-800">
              <table className="w-full text-[12.5px]">
                <thead>
                  <tr className="bg-slate-900/70 text-slate-400">
                    {results.rows[0].map((h) => (
                      <th key={h} className="px-3 py-2 text-left font-semibold">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {results.rows.slice(1).map((row, i) => (
                    <tr key={i} className="border-t border-slate-800/60 font-mono">
                      {row.map((c, j) => (
                        <td
                          key={j}
                          className={`px-3 py-1.5 ${j === 5 && c.startsWith('-') ? 'text-rose-400' : 'text-slate-300'}`}
                        >
                          {c}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {chartOk ? (
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chart} margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
                    <CartesianGrid stroke="#1e293b" vertical={false} />
                    <XAxis dataKey="task" tick={{ fontSize: 11, fill: '#94a3b8' }} interval={0} angle={-12} height={52} />
                    <YAxis tick={{ fontSize: 11, fill: '#94a3b8' }} domain={['auto', 'auto']} />
                    <Tooltip
                      contentStyle={{ background: '#0f172a', border: '1px solid #334155', borderRadius: 10, fontSize: 12 }}
                      labelStyle={{ color: '#e2e8f0' }}
                    />
                    <Legend wrapperStyle={{ fontSize: 12 }} />
                    <Bar dataKey="quant" name="量化" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="base" name="基线" fill="#22d3ee" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <div className="text-[11.5px] text-slate-600">
                真实评测维度 &lt; 3，只出表格不出图（设计稿 §6）
              </div>
            )}
            {results.note && <div className="text-[11.5px] text-slate-500">{results.note}</div>}
          </div>
        )}
      </section>
    </div>
  )
}

const Card = ({ title, children }: { title: string; children: React.ReactNode }) => (
  <section className="p-4 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-3">
    <h3 className="text-[12px] font-semibold text-slate-400 tracking-wide uppercase">{title}</h3>
    {children}
  </section>
)

const Field = ({ label, children }: { label: string; children: React.ReactNode }) => (
  <label className="block space-y-1">
    <span className="text-[12px] text-slate-400">{label}</span>
    {children}
  </label>
)
