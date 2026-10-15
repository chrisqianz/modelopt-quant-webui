/**
 * 路径与环境区：四路径卡（PUT /api/paths）+ 环境卡（venv / eval-env job）
 * + 探活卡（POST /api/probe，HostStatus 全字段，✗ 红色高亮）+ 网络卡（PUT /api/net + 测试）。
 */
import { useEffect, useState } from 'react'
import { Activity, FolderOpen, Loader2, Network, Wifi } from 'lucide-react'
import { api } from '../api/endpoints'
import type { Bootstrap, HostStatus, Net, Paths } from '../api/types'
import { useToast } from '../components/ToastProvider'
import ProgressCard from '../components/ProgressCard'
import LogPanel from '../components/LogPanel'
import { useJobStream } from '../hooks/useJobStream'

const inputCls =
  'w-full px-3 py-1.5 rounded-lg bg-slate-950 border border-slate-800 text-[13px] text-slate-100 font-mono outline-none focus:border-violet-500/60 placeholder:text-slate-600'

const PROXY_MODES: { value: string; label: string }[] = [
  { value: 'auto', label: 'auto（先直连，失败走代理）' },
  { value: 'direct', label: 'direct（仅直连）' },
  { value: 'manual', label: 'manual（仅代理）' },
]

/** 探活明细表：[字段, 值, 是否异常] */
function probeRows(p: HostStatus): [string, string, boolean][] {
  return [
    ['reachable', p.reachable ? '可达' : '不可达', !p.reachable],
    ['gpu', p.gpu || '—', !p.gpu],
    ['gpu_util', p.gpu_util == null ? '—' : `${p.gpu_util}%`, false],
    ['gpu_mem', p.gpu_mem || '—', false],
    ['repo_ok', p.repo_ok ? '✓ Model-Optimizer 仓库在' : '✗ 仓库缺失', !p.repo_ok],
    ['venv_ok', p.venv_ok ? '✓ 量化 venv 就绪' : '✗ 量化 venv 未创建', !p.venv_ok],
    ['uv_ok', p.uv_ok ? '✓ uv 可用' : '✗ uv 不可用', !p.uv_ok],
    ['disk_free', p.disk_free || '—', false],
    ['hf_cache', p.hf_cache || '—', false],
    ['hf_direct_ok', p.hf_direct_ok ? '✓ HF 直连通' : '✗ HF 直连不通', !p.hf_direct_ok],
    ['hf_proxy_ok', p.hf_proxy_ok ? '✓ 代理到 HF 通' : '—（auto 模式下未生效）', !p.hf_proxy_ok && p.hf_direct_ok === false],
    ['proxy_note', p.proxy_note || '—', false],
    ['error', p.error || '—', !!p.error],
  ]
}

export default function SettingsView({ data, onSaved }: { data: Bootstrap | null; onSaved: () => void }) {
  const toast = useToast()

  // 路径卡
  const [paths, setPaths] = useState<Paths>({
    repo: '', venv: '', task_dir: '', hf_cache: '',
  })
  // 网络卡
  const [net, setNet] = useState<Net>({ mode: 'auto', proxy: '' })
  const [netResult, setNetResult] = useState<{ ok: boolean; detail: string } | null>(null)
  // 探活卡
  const [probe, setProbe] = useState<HostStatus | null>(null)
  const [probing, setProbing] = useState(false)
  // 环境卡（venv / eval-env job）
  const [venvJob, setVenvJob] = useState<string | null>(null)
  const [venvState, setVenvState] = useState<string | null>(null)
  const [evalEnvJob, setEvalEnvJob] = useState<string | null>(null)
  const [evalEnvState, setEvalEnvState] = useState<string | null>(null)
  const venvStream = useJobStream(venvJob, venvState)
  const evalEnvStream = useJobStream(evalEnvJob, evalEnvState)

  // bootstrap 数据到位后回填（本地编辑值不被旧数据冲掉：只回填一次）
  useEffect(() => {
    if (!data) return
    setPaths(data.paths)
    setNet(data.net)
    setProbe(data.probe)
  }, [data])

  const savePaths = async () => {
    try {
      await api.putPaths(paths)
      toast.show('路径已保存到 config.yaml')
      onSaved()
    } catch (e) {
      toast.show(e instanceof Error ? e.message : String(e), 'error')
    }
  }

  const saveNet = async () => {
    try {
      await api.putNet(net)
      toast.show('网络模式已保存到 net.json')
      onSaved()
    } catch (e) {
      toast.show(e instanceof Error ? e.message : String(e), 'error')
    }
  }

  const testNet = async () => {
    setNetResult(null)
    try {
      setNetResult(await api.netTest())
    } catch (e) {
      setNetResult({ ok: false, detail: e instanceof Error ? e.message : String(e) })
    }
  }

  const doProbe = async () => {
    setProbing(true)
    try {
      const r = await api.probe()
      setProbe(r.probe)
    } catch (e) {
      toast.show(e instanceof Error ? e.message : String(e), 'error')
    } finally {
      setProbing(false)
    }
  }

  const startVenv = async () => {
    try {
      const r = await api.venv()
      setVenvJob(r.job_id)
      setVenvState(r.job.state)
    } catch (e) {
      toast.show(e instanceof Error ? e.message : String(e), 'error')
    }
  }

  const startEvalEnv = async () => {
    try {
      const r = await api.evalEnv()
      setEvalEnvJob(r.job_id)
      setEvalEnvState(r.job.state)
    } catch (e) {
      toast.show(e instanceof Error ? e.message : String(e), 'error')
    }
  }

  const probeNow = data?.probe ?? probe
  const venvOk = probeNow?.venv_ok ?? false
  const envJobBusy = (jobId: string | null, stream: { state: string | null; finished: boolean }) =>
    !!jobId && !stream.finished

  return (
    <div className="w-full space-y-4">
      {/* ① 路径卡 */}
      <section className="p-4 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-3">
        <h3 className="flex items-center gap-2 text-[12px] font-semibold text-slate-400 tracking-wide uppercase">
          <FolderOpen size={13} /> 本机路径（写回 config.yaml）
        </h3>
        <div className="grid grid-cols-2 gap-x-4 gap-y-3">
          <label className="block space-y-1">
            <span className="text-[12px] text-slate-400">repo（Model-Optimizer 仓库）</span>
            <input value={paths.repo} onChange={(e) => setPaths({ ...paths, repo: e.target.value })}
              className={inputCls} placeholder="~/Model-Optimizer" />
          </label>
          <label className="block space-y-1">
            <span className="text-[12px] text-slate-400">venv（量化用）</span>
            <input value={paths.venv} onChange={(e) => setPaths({ ...paths, venv: e.target.value })}
              className={inputCls} placeholder="~/.venvs/modelopt" />
          </label>
          <label className="block space-y-1">
            <span className="text-[12px] text-slate-400">task_dir（任务/日志目录）</span>
            <input value={paths.task_dir} onChange={(e) => setPaths({ ...paths, task_dir: e.target.value })}
              className={inputCls} placeholder="~/quant-webui-tasks" />
          </label>
          <label className="block space-y-1">
            <span className="text-[12px] text-slate-400">hf_cache（HF 缓存）</span>
            <input value={paths.hf_cache} onChange={(e) => setPaths({ ...paths, hf_cache: e.target.value })}
              className={inputCls} placeholder="~/.cache/huggingface" />
          </label>
        </div>
        <div className="flex justify-end">
          <button
            onClick={() => void savePaths()}
            className="px-4 py-1.5 rounded-lg text-[12px] font-medium bg-violet-600 text-white hover:bg-violet-500"
          >
            保存路径
          </button>
        </div>
      </section>

      {/* ② 环境卡 */}
      <section className="p-4 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-3">
        <h3 className="flex items-center gap-2 text-[12px] font-semibold text-slate-400 tracking-wide uppercase">
          <Activity size={13} /> 环境
        </h3>
        <div className="space-y-4">
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-[13px] text-slate-300">
                量化 venv（{paths.venv || '未配置'}）
                {venvOk && <span className="ml-2 px-1.5 py-0.5 rounded-md bg-emerald-500/15 text-emerald-300 text-[11px]">已就绪</span>}
              </span>
              <button
                onClick={() => void startVenv()}
                disabled={venvOk || envJobBusy(venvJob, venvStream)}
                className="px-3.5 py-1.5 rounded-lg text-[12px] font-medium bg-slate-800 text-slate-200 hover:bg-slate-700 disabled:opacity-40"
              >
                {venvOk ? '已就绪' : venvStream.finished ? '重新检查' : '创建量化 venv（5-15 分钟）'}
              </button>
            </div>
            {venvJob && !venvStream.finished && (
              <>
                <ProgressCard progress={venvStream.progress} state={venvStream.state} />
                <LogPanel text={venvStream.log} />
              </>
            )}
            {venvJob && venvStream.finished && (
              <div className={`text-[12px] ${venvStream.state === 'done' ? 'text-emerald-300' : 'text-rose-400'}`}>
                结果：{venvStream.state}（日志已停止更新）
                <button
                  onClick={() => { setVenvJob(null); setVenvState(null); void doProbe() }}
                  className="ml-3 text-cyan-400 hover:text-cyan-300"
                >
                  重新探活
                </button>
              </div>
            )}
          </div>

          <div className="space-y-2 border-t border-slate-800/60 pt-3">
            <div className="flex items-center justify-between">
              <span className="text-[13px] text-slate-300">评测环境（vllm + lm-eval）</span>
              <button
                onClick={() => void startEvalEnv()}
                disabled={envJobBusy(evalEnvJob, evalEnvStream)}
                className="px-3.5 py-1.5 rounded-lg text-[12px] font-medium bg-slate-800 text-slate-200 hover:bg-slate-700 disabled:opacity-40"
              >
                {evalEnvStream.finished ? '再次创建评测环境' : '创建评测环境（首次较慢）'}
              </button>
            </div>
            {evalEnvJob && !evalEnvStream.finished && (
              <>
                <ProgressCard progress={evalEnvStream.progress} state={evalEnvStream.state} />
                <LogPanel text={evalEnvStream.log} />
              </>
            )}
            {evalEnvJob && evalEnvStream.finished && (
              <div className={`text-[12px] ${evalEnvStream.state === 'done' ? 'text-emerald-300' : 'text-rose-400'}`}>
                结果：{evalEnvStream.state}
              </div>
            )}
          </div>
        </div>
      </section>

      {/* ③ 探活卡 */}
      <section className="p-4 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-3">
        <div className="flex items-center justify-between">
          <h3 className="flex items-center gap-2 text-[12px] font-semibold text-slate-400 tracking-wide uppercase">
            <Wifi size={13} /> 探活明细
          </h3>
          <button
            onClick={() => void doProbe()}
            disabled={probing}
            className="flex items-center gap-1 px-3 py-1.5 rounded-lg text-[12px] font-medium bg-slate-800 text-slate-200 hover:bg-slate-700 disabled:opacity-40"
          >
            {probing ? <Loader2 size={13} className="animate-spin" /> : <Activity size={13} />}
            重新探活
          </button>
        </div>
        {probeNow ? (
          <div className="overflow-hidden rounded-xl border border-slate-800">
            <table className="w-full text-[12.5px]">
              <tbody>
                {probeRows(probeNow).map(([k, v, bad]) => (
                  <tr key={k} className="border-t border-slate-800/60 first:border-t-0">
                    <td className="px-3 py-1.5 font-mono text-slate-500 w-32">{k}</td>
                    <td className={`px-3 py-1.5 font-mono ${bad ? 'text-rose-400' : 'text-slate-300'}`}>{v}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="text-[12.5px] text-slate-500">等待首屏数据…</div>
        )}
      </section>

      {/* ④ 网络卡 */}
      <section className="p-4 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-3">
        <h3 className="flex items-center gap-2 text-[12px] font-semibold text-slate-400 tracking-wide uppercase">
          <Network size={13} /> HF 网络（写回 net.json）
        </h3>
        <div className="grid grid-cols-2 gap-x-4 gap-y-3">
          <label className="block space-y-1">
            <span className="text-[12px] text-slate-400">mode</span>
            <select value={net.mode} onChange={(e) => setNet({ ...net, mode: e.target.value })} className={inputCls}>
              {PROXY_MODES.map((m) => <option key={m.value} value={m.value}>{m.label}</option>)}
            </select>
          </label>
          <label className="block space-y-1">
            <span className="text-[12px] text-slate-400">proxy（manual/auto 兜底用）</span>
            <input
              value={net.proxy}
              onChange={(e) => setNet({ ...net, proxy: e.target.value })}
              placeholder="http://192.168.124.24:7890"
              className={inputCls}
            />
          </label>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={() => void saveNet()}
            className="px-4 py-1.5 rounded-lg text-[12px] font-medium bg-violet-600 text-white hover:bg-violet-500"
          >
            保存网络
          </button>
          <button
            onClick={() => void testNet()}
            className="px-4 py-1.5 rounded-lg text-[12px] font-medium bg-slate-800 text-slate-200 hover:bg-slate-700"
          >
            测试连通性
          </button>
        </div>
        {netResult && (
          <div
            className={`p-3 rounded-xl text-[12px] ${
              netResult.ok
                ? 'bg-emerald-950/40 border border-emerald-500/30 text-emerald-300'
                : 'bg-rose-950/40 border border-rose-500/30 text-rose-300'
            }`}
          >
            {netResult.ok ? '✓ ' : '✗ '}{netResult.detail}
          </div>
        )}
      </section>
    </div>
  )
}
