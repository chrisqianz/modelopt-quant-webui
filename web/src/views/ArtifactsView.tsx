/**
 * 产物库区（C1 完整版）：GET /api/artifacts → 卡片列表
 * （模型名 / qformat 徽章 / 时间 / 大小 / 路径 / state 徽章）
 * + 「生成 vLLM 启动命令」纯前端模板拼串 + 复制。
 *
 * 模板（设计稿 §8）：
 *   vllm serve <abs_export_dir> --served-model-name <slug> --dtype auto --gpu-memory-utilization 0.9
 * abs_export_dir = task_dir + '/' + dir；展示层把 `~` 展开成 $HOME，其余原样。
 */
import { useEffect, useState } from 'react'
import { Boxes, Copy, RefreshCw, TerminalSquare } from 'lucide-react'
import { api } from '../api/endpoints'
import { useToast } from '../components/ToastProvider'
import type { Artifact, Bootstrap } from '../api/types'

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

/** task_dir（可能带 ~）拼出绝对导出目录；展示层 ~ → $HOME。 */
function absExportDir(taskDir: string, dir: string): string {
  const base = (taskDir || '').replace(/^~(?=[/\\]|$)/, '$HOME')
  return `${base.replace(/\/+$/, '')}/${dir}`
}

/** served-model-name：模型名把 / 换成 -；没有模型名退回目录名。 */
function slugOf(a: Artifact): string {
  return (a.model || a.dir).replace(/[^A-Za-z0-9._-]+/g, '-').replace(/^-+|-+$/g, '') || a.dir
}

function buildVllmCmd(a: Artifact, taskDir: string): string {
  return (
    `vllm serve ${absExportDir(taskDir, a.dir)} ` +
    `--served-model-name ${slugOf(a)} --dtype auto --gpu-memory-utilization 0.9`
  )
}

function ArtifactCard({ a, taskDir }: { a: Artifact; taskDir: string }) {
  const toast = useToast()
  const [cmd, setCmd] = useState<string | null>(null)

  const copy = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text)
      toast.show('已复制命令')
    } catch {
      toast.show('复制失败', 'error')
    }
  }

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-3 space-y-1.5">
      <div className="flex items-center justify-between gap-2">
        <span className="font-mono text-[12.5px] text-slate-200 truncate" title={a.dir}>
          {a.dir}
        </span>
        <span className={`px-2 py-0.5 rounded-md text-[11px] shrink-0 ${stateBadge(a.state)}`}>
          {stateLabel(a.state)}
        </span>
      </div>
      <div className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-[11.5px]">
        <span className="text-slate-500">模型</span>
        <span className="font-mono text-slate-400 truncate" title={a.model}>
          {a.model || '—'}
        </span>
        <span className="text-slate-500">格式</span>
        {a.qformat ? (
          <span className="px-1.5 py-px rounded bg-cyan-500/10 text-cyan-300 font-mono">
            {a.qformat}
          </span>
        ) : (
          <span className="text-slate-600">—</span>
        )}
        <span className="text-slate-500">时间</span>
        <span className="font-mono text-slate-400">{a.ts || '—'}</span>
        <span className="text-slate-500">大小</span>
        <span className="font-mono text-slate-400">{a.size || '—'}</span>
        <span className="text-slate-500">路径</span>
        <span className="font-mono text-slate-400 truncate" title={absExportDir(taskDir, a.dir)}>
          {taskDir ? absExportDir(taskDir, a.dir) : '—'}
        </span>
      </div>
      <button
        onClick={() => setCmd(cmd === null ? buildVllmCmd(a, taskDir) : null)}
        className="mt-1 flex items-center gap-1 text-[11.5px] text-cyan-400 hover:text-cyan-300"
      >
        <TerminalSquare size={12} /> {cmd === null ? '生成 vLLM 启动命令' : '收起命令'}
      </button>
      {cmd !== null && (
        <div className="relative group">
          <pre className="p-2.5 rounded-lg bg-slate-900 border border-slate-800 text-[11.5px] font-mono text-emerald-300 overflow-x-auto whitespace-pre-wrap break-all">
            {cmd}
          </pre>
          <button
            onClick={() => void copy(cmd)}
            title="复制命令"
            className="absolute top-1.5 right-1.5 p-1 rounded-md bg-slate-800 text-slate-400 hover:text-slate-200 opacity-0 group-hover:opacity-100 transition-opacity"
          >
            <Copy size={12} />
          </button>
        </div>
      )}
    </div>
  )
}

export default function ArtifactsView({ data }: { data: Bootstrap | null }) {
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

  const taskDir = data?.paths?.task_dir ?? ''

  return (
    <div className="w-full space-y-4">
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
              <ArtifactCard key={a.dir} a={a} taskDir={taskDir} />
            ))}
          </div>
        )}
      </section>
    </div>
  )
}
