/**
 * 量化区：20 字段表单（分区卡片）+ 命令预览（逐字节对 golden）+ 启动/中止。
 * 默认值：bootstrap.defaults 覆盖 schema 默认；每数据集条数默认 calib_size。
 */
import { useEffect, useState } from 'react'
import { Copy, Play, Plus, Trash2, Wand2 } from 'lucide-react'
import { api } from '../api/endpoints'
import type {
  Bootstrap, LocalModel, PreviewResult, QuantizeParams,
} from '../api/types'
import { useToast } from '../components/ToastProvider'
import ConfirmModal from '../components/ConfirmModal'
import ProgressCard from '../components/ProgressCard'
import LogPanel from '../components/LogPanel'
import { useJobStream } from '../hooks/useJobStream'

interface FormState {
  model: string
  trust_remote_code: boolean
  qformat: string
  recipe: string
  kv_cache_qformat: string
  datasets: string[]
  sizes: Record<string, string>
  custom: { name: string; size: string }[]
  calib_seq: string
  calib_with_images: boolean
  export_path: string
  batch_size: string
  device: string
  gpu_max_mem_percentage: string
  attn_implementation: string
  use_fsdp2: boolean
  cpu_offload: boolean
  low_memory_mode: boolean
  skip_generate: boolean
  extra_args: string
}

function buildInitialForm(data: Bootstrap | null): FormState {
  const d = (data?.defaults ?? {}) as Record<string, unknown>
  const num = (v: unknown, fallback: number) =>
    typeof v === 'number' && Number.isFinite(v) ? v : fallback
  const str = (v: unknown, fallback: string) => (typeof v === 'string' ? v : fallback)
  const defSize = num(d.calib_size, 1024)
  return {
    model: '',
    trust_remote_code: false,
    qformat: str(d.qformat, 'nvfp4'),
    recipe: '',
    kv_cache_qformat: str(d.kv_cache_qformat, 'none'),
    datasets: [],
    sizes: {},
    custom: [],
    calib_seq: String(num(d.calib_seq, 512)),
    calib_with_images: false,
    export_path: '',
    batch_size: '1',
    device: 'cuda',
    gpu_max_mem_percentage: '0',
    attn_implementation: 'sdpa',
    use_fsdp2: false,
    cpu_offload: false,
    low_memory_mode: false,
    skip_generate: false,
    extra_args: '',
  }
  void defSize
}

export default function QuantizeView({ data }: { data: Bootstrap | null }) {
  const [form, setForm] = useState<FormState>(() => buildInitialForm(data))
  const [models, setModels] = useState<LocalModel[]>([])
  const [showAllModels, setShowAllModels] = useState(false)
  const [preview, setPreview] = useState<PreviewResult | null>(null)
  const [previewErr, setPreviewErr] = useState('')
  const [confirmOpen, setConfirmOpen] = useState(false)
  const [jobId, setJobId] = useState<string | null>(null)
  const [jobState, setJobState] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const toast = useToast()
  const stream = useJobStream(jobId, jobState)

  // 首屏数据到位后补 C2 模型列表（datalist 自动补全）
  useEffect(() => {
    void (async () => {
      try {
        const r = await api.modelsLocal()
        setModels(r.models)
      } catch { /* C2 不可用不阻塞 */ }
    })()
  }, [])

  const set = <K extends keyof FormState>(k: K, v: FormState[K]) =>
    setForm((f) => ({ ...f, [k]: v }))

  const vocab = data?.vocab
  const unmappedFlags = data?.capabilities?.unmapped ?? [] // 逃生口：本版本有、无控件的 flag
  const defaultSize = String(
    typeof data?.defaults?.calib_size === 'number' ? data.defaults.calib_size : 1024,
  )

  const sizeOf = (ds: string) => form.sizes[ds] ?? defaultSize

  const toParams = (): QuantizeParams => ({
    model: form.model.trim(),
    trust_remote_code: form.trust_remote_code,
    qformat: form.qformat,
    recipe: form.recipe.trim(),
    kv_cache_qformat: form.kv_cache_qformat,
    datasets: form.datasets,
    dataset_sizes: Object.fromEntries(
      form.datasets.map((d) => [d, parseInt(sizeOf(d), 10) || 1024]),
    ),
    custom_datasets: form.custom
      .filter((c) => c.name.trim())
      .map((c) => [c.name.trim(), parseInt(c.size || '1024', 10) || 1024]),
    calib_seq: parseInt(form.calib_seq, 10) || 512,
    calib_with_images: form.calib_with_images,
    export_path: form.export_path.trim(),
    batch_size: parseInt(form.batch_size, 10) || 1,
    device: form.device,
    gpu_max_mem_percentage: parseFloat(form.gpu_max_mem_percentage) || 0,
    attn_implementation: form.attn_implementation,
    use_fsdp2: form.use_fsdp2,
    cpu_offload: form.cpu_offload,
    low_memory_mode: form.low_memory_mode,
    skip_generate: form.skip_generate,
    extra_args: form.extra_args,
  })

  const canPreview = !!form.model.trim() && !!form.qformat && form.datasets.length > 0

  const doPreview = async () => {
    setPreviewErr('')
    try {
      const r = await api.preview(toParams())
      setPreview(r)
    } catch (e) {
      setPreview(null)
      setPreviewErr(e instanceof Error ? e.message : String(e))
    }
  }

  const doStart = async () => {
    setBusy(true)
    try {
      const r = await api.quantize(toParams())
      if (r.warn_gpu) {
        setConfirmOpen(true)
        return
      }
      setJobId(r.job_id)
      setJobState(r.job.state)
    } catch (e) {
      toast.show(e instanceof Error ? e.message : String(e), 'error')
    } finally {
      setBusy(false)
    }
  }

  const onAbort = async () => {
    if (!jobId) return
    try {
      await api.abort(jobId)
      toast.show('中止信号已发送')
      setJobState('aborted')
    } catch (e) {
      toast.show(e instanceof Error ? e.message : String(e), 'error')
    }
  }

  const copy = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text)
      toast.show('已复制命令')
    } catch {
      toast.show('复制失败', 'error')
    }
  }

  const running = jobId && (jobState === 'running' || jobState === 'starting' ||
    (stream.state === 'running' || stream.state === 'starting'))

  const capErr = preview?.caps_error || ''

  return (
    <div className="max-w-4xl space-y-4">
      {jobId && running ? (
        // 运行态：进度 + 日志 + 中止
        <div className="space-y-3">
          <ProgressCard progress={stream.progress} state={stream.state ?? jobState} />
          <LogPanel text={stream.log} />
          <div className="flex justify-end">
            <button
              onClick={() => void onAbort()}
              className="px-4 py-1.5 rounded-lg text-[12px] font-medium bg-rose-600/80 text-white hover:bg-rose-500"
            >
              中止任务
            </button>
          </div>
        </div>
      ) : (
        <>
          {/* ① 模型 */}
          <Card title="模型">
            <div className="grid grid-cols-[1fr_auto] gap-x-4 gap-y-3 items-center">
              <Field label="模型路径 / HF id" full>
                <input
                  list="local-models"
                  value={form.model}
                  onChange={(e) => set('model', e.target.value)}
                  placeholder="Qwen/Qwen2.5-1.5B-Instruct 或 /abs/path"
                  className={inputCls}
                />
                <datalist id="local-models">
                  {models.map((m) => (
                    <option key={m.id} value={m.id}>{m.size}</option>
                  ))}
                </datalist>
              </Field>
              {models.length > 0 && (
                <div className="flex flex-wrap gap-1.5">
                  <span className="text-[11px] text-slate-500 self-center">本机缓存：</span>
                  {(showAllModels ? models : models.slice(0, 8)).map((m) => {
                    const selected = form.model.trim() === m.id
                    return (
                      <button
                        key={m.id}
                        type="button"
                        title={`${m.id}（${m.size || '未知大小'}${m.revision ? '，rev ' + m.revision.slice(0, 12) : ''}${m.stub ? '，空壳' : ''}）`}
                        onClick={() => set('model', m.id)}
                        className={`flex items-center gap-1 px-2 py-0.5 rounded-md text-[11px] font-mono border transition-colors ${
                          selected
                            ? 'border-violet-500 bg-violet-500/15 text-violet-300'
                            : m.stub
                              ? 'border-slate-800 bg-slate-900 text-slate-500 hover:text-slate-400'
                              : 'border-slate-700 bg-slate-900 text-slate-300 hover:bg-slate-800 hover:border-slate-600'
                        }`}
                      >
                        {m.id}
                        {m.stub && (
                          <span className="px-1 rounded bg-slate-700/60 text-[10px] text-slate-400">空壳</span>
                        )}
                      </button>
                    )
                  })}
                  {models.length > 8 && (
                    <button
                      type="button"
                      onClick={() => setShowAllModels((v) => !v)}
                      className="px-2 py-0.5 rounded-md text-[11px] text-cyan-400 hover:text-cyan-300 border border-cyan-800/40"
                    >
                      {showAllModels ? '收起' : `全部 ${models.length} 个`}
                    </button>
                  )}
                </div>
              )}
              <Toggle label="trust_remote_code" checked={form.trust_remote_code}
                onChange={(v) => set('trust_remote_code', v)} />
            </div>
          </Card>

          {/* ② 量化 */}
          <Card title="量化">
            <div className="grid grid-cols-2 gap-x-4 gap-y-3">
              <Field label="量化格式（qformat）">
                <select value={form.qformat} onChange={(e) => set('qformat', e.target.value)} className={inputCls}>
                  {(vocab?.qformats ?? []).map((q) => <option key={q} value={q}>{q}</option>)}
                </select>
              </Field>
              <Field label="KV 量化">
                <select value={form.kv_cache_qformat} onChange={(e) => set('kv_cache_qformat', e.target.value)} className={inputCls}>
                  {(vocab?.kv ?? []).map(([name, value]) => (
                    <option key={name} value={value}>{name}</option>
                  ))}
                </select>
              </Field>
              <Field label="recipe（可选）" full>
                <input value={form.recipe} onChange={(e) => set('recipe', e.target.value)}
                  placeholder="留空 = 默认 recipe" className={inputCls} />
              </Field>
            </div>
          </Card>

          {/* ③ 校准 */}
          <Card title="校准数据集">
            <div className="space-y-2">
              {(vocab?.datasets ?? []).map((d) => {
                const on = form.datasets.includes(d.key)
                return (
                  <div key={d.key} className="flex items-center gap-3">
                    <input
                      type="checkbox"
                      checked={on}
                      onChange={() =>
                        set('datasets', on ? form.datasets.filter((x) => x !== d.key)
                          : [...form.datasets, d.key])
                      }
                      className="accent-violet-500"
                    />
                    <span className="text-[13px] text-slate-200 w-40 font-mono">{d.key}</span>
                    <span className="text-[12px] text-slate-500 truncate flex-1">{d.desc}</span>
                    {on && (
                      <input
                        value={sizeOf(d.key)}
                        onChange={(e) => set('sizes', { ...form.sizes, [d.key]: e.target.value })}
                        className="w-20 px-2 py-1 rounded-lg bg-slate-950 border border-slate-800 font-mono text-[12px] text-slate-100 outline-none focus:border-violet-500/60"
                      />
                    )}
                  </div>
                )
              })}
              {/* 自定义数据集行 */}
              {form.custom.map((c, i) => (
                <div key={i} className="flex items-center gap-3">
                  <span className="w-5" />
                  <input
                    value={c.name}
                    onChange={(e) => {
                      const next = [...form.custom]
                      next[i] = { ...c, name: e.target.value }
                      set('custom', next)
                    }}
                    placeholder="名称"
                    className="w-40 px-2 py-1 rounded-lg bg-slate-950 border border-slate-800 font-mono text-[12px] text-slate-100 outline-none focus:border-violet-500/60"
                  />
                  <input
                    value={c.size}
                    onChange={(e) => {
                      const next = [...form.custom]
                      next[i] = { ...c, size: e.target.value }
                      set('custom', next)
                    }}
                    placeholder="条数"
                    className="w-20 px-2 py-1 rounded-lg bg-slate-950 border border-slate-800 font-mono text-[12px] text-slate-100 outline-none focus:border-violet-500/60"
                  />
                  <button
                    onClick={() => set('custom', form.custom.filter((_, j) => j !== i))}
                    className="text-slate-600 hover:text-rose-400"
                    title="删除行"
                  >
                    <Trash2 size={13} />
                  </button>
                </div>
              ))}
              <button
                onClick={() => set('custom', [...form.custom, { name: '', size: defaultSize }])}
                className="flex items-center gap-1.5 text-[12px] text-violet-400 hover:text-violet-300"
              >
                <Plus size={13} /> 添加自定义数据集
              </button>
              <div className="grid grid-cols-2 gap-x-4 gap-y-3 pt-2 border-t border-slate-800/60">
                <Field label="calib_seq">
                  <input value={form.calib_seq} onChange={(e) => set('calib_seq', e.target.value)}
                    className={inputCls} />
                </Field>
                <Toggle label="calib_with_images" checked={form.calib_with_images}
                  onChange={(v) => set('calib_with_images', v)} />
              </div>
            </div>
          </Card>

          {/* ④ 导出/资源 */}
          <Card title="导出与资源">
            <div className="grid grid-cols-2 gap-x-4 gap-y-3">
              <Field label="export_path（可选）" full>
                <input value={form.export_path} onChange={(e) => set('export_path', e.target.value)}
                  placeholder="留空 = 模型名_ptq 默认目录" className={inputCls} />
              </Field>
              <Field label="batch_size">
                <input value={form.batch_size} onChange={(e) => set('batch_size', e.target.value)}
                  className={inputCls} />
              </Field>
              <Field label="device">
                <input value={form.device} onChange={(e) => set('device', e.target.value)} className={inputCls} />
              </Field>
              <Field label="gpu_max_mem_percentage（0=不限制）">
                <input value={form.gpu_max_mem_percentage}
                  onChange={(e) => set('gpu_max_mem_percentage', e.target.value)} className={inputCls} />
              </Field>
            </div>
          </Card>

          {/* ⑤ 高级 */}
          <details className="rounded-2xl bg-slate-900/60 border border-slate-800">
            <summary className="px-4 py-3 text-[13px] font-semibold text-slate-300 cursor-pointer select-none">
              高级参数
            </summary>
            <div className="px-4 pb-4 grid grid-cols-2 gap-x-4 gap-y-3">
              <Field label="attn_implementation">
                <select value={form.attn_implementation}
                  onChange={(e) => set('attn_implementation', e.target.value)} className={inputCls}>
                  <option value="sdpa">sdpa</option>
                  <option value="eager">eager</option>
                  <option value="flash_attention_2">flash_attention_2</option>
                </select>
              </Field>
              <div className="space-y-2 pt-5">
                <Toggle label="use_fsdp2" checked={form.use_fsdp2} onChange={(v) => set('use_fsdp2', v)} />
                <Toggle label="cpu_offload" checked={form.cpu_offload} onChange={(v) => set('cpu_offload', v)} />
                <Toggle label="low_memory_mode" checked={form.low_memory_mode} onChange={(v) => set('low_memory_mode', v)} />
                <Toggle label="skip_generate" checked={form.skip_generate} onChange={(v) => set('skip_generate', v)} />
              </div>
              <Field label="extra_args（逃生口：原样拼到命令尾；禁止 ; 与 &&）" full>
                {unmappedFlags.length > 0 && (
                  <div className="flex flex-wrap gap-1.5 mb-2">
                    <span className="text-[11px] text-slate-500 self-center">本版本未映射的参数（点击加入附加参数）：</span>
                    {unmappedFlags.map((f) => {
                      const added = form.extra_args.includes(f)
                      return (
                        <button
                          key={f}
                          type="button"
                          disabled={added}
                          title={added ? '已在附加参数里' : `点击填入 ${f}`}
                          onClick={() => {
                            const cur = form.extra_args.trim()
                            set('extra_args', cur ? `${cur} ${f}` : f)
                          }}
                          className={`px-2 py-0.5 rounded-md text-[11px] font-mono border transition-colors ${
                            added
                              ? 'border-slate-800 bg-slate-900 text-slate-600 cursor-default'
                              : 'border-cyan-800/60 bg-slate-900 text-cyan-300 hover:bg-slate-800 hover:border-cyan-600'
                          }`}
                        >
                          {f}
                        </button>
                      )
                    })}
                  </div>
                )}
                <input value={form.extra_args} onChange={(e) => set('extra_args', e.target.value)}
                  className={inputCls + ' font-mono'} placeholder="留空" />
              </Field>
            </div>
          </details>

          {/* 操作区 */}
          <div className="flex items-center gap-3">
            <button
              onClick={() => void doPreview()}
              disabled={!canPreview}
              className="flex items-center gap-1.5 px-4 py-2 rounded-xl text-[13px] font-medium bg-slate-800 text-slate-200 hover:bg-slate-700 disabled:opacity-40"
            >
              <Wand2 size={14} /> 命令预览
            </button>
            <button
              onClick={() => void doStart()}
              disabled={!canPreview || busy}
              className="flex items-center gap-1.5 px-5 py-2 rounded-xl text-[13px] font-medium bg-violet-600 text-white hover:bg-violet-500 disabled:opacity-40"
            >
              <Play size={14} /> 开始量化
            </button>
            {!canPreview && (
              <span className="text-[12px] text-slate-500">需填模型 + 选至少一个校准数据集</span>
            )}
          </div>

          {/* 预览结果 */}
          {previewErr && (
            <div className="p-3 rounded-xl bg-rose-950/40 border border-rose-500/30 text-[12px] text-rose-300 font-mono whitespace-pre-wrap">
              {previewErr}
            </div>
          )}
          {preview && (
            <div className="space-y-2">
              {capErr && (
                <div className="p-3 rounded-xl bg-amber-950/40 border border-amber-500/30 text-[12px] text-amber-300">
                  capabilities 扫描失败：{capErr}（未做 flag 差集校验，请自行核对命令）
                </div>
              )}
              <div className="rounded-2xl border border-slate-800 bg-slate-950 overflow-hidden">
                <div className="flex items-center justify-between px-3 py-1.5 border-b border-slate-800/80 bg-slate-900/50">
                  <span className="text-[11px] font-semibold text-slate-400">命令预览（与 golden 逐字节同源）</span>
                  <button
                    onClick={() => void copy(preview.cmd)}
                    className="flex items-center gap-1 text-[11px] text-cyan-400 hover:text-cyan-300"
                  >
                    <Copy size={12} /> 复制
                  </button>
                </div>
                <pre className="px-3 py-2.5 font-mono text-[11.5px] leading-relaxed text-slate-300 whitespace-pre-wrap break-all">
                  {preview.pretty}
                </pre>
              </div>
            </div>
          )}

          {/* 已终态的 job：回看快照 */}
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

      <ConfirmModal
        open={confirmOpen}
        title="GPU 当前被占用"
        body="显存利用率已很高，继续可能与别的进程抢显存（可能 OOM）。确认继续？"
        confirmLabel="继续量化"
        onConfirm={() => {
          setConfirmOpen(false)
          void (async () => {
            try {
              const r = await api.quantize(toParams())
              setJobId(r.job_id)
              setJobState(r.job.state)
            } catch (e) {
              toast.show(e instanceof Error ? e.message : String(e), 'error')
            }
          })()
        }}
        onCancel={() => setConfirmOpen(false)}
      />
    </div>
  )
}

const inputCls =
  'w-full px-3 py-1.5 rounded-lg bg-slate-950 border border-slate-800 text-[13px] text-slate-100 outline-none focus:border-violet-500/60 placeholder:text-slate-600'

function Card({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="p-4 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-3">
      <h3 className="text-[12px] font-semibold text-slate-400 tracking-wide uppercase">{title}</h3>
      {children}
    </section>
  )
}

function Field({ label, full, children }: { label: string; full?: boolean; children: React.ReactNode }) {
  return (
    <label className={`block space-y-1 ${full ? 'col-span-2' : ''}`}>
      <span className="text-[12px] text-slate-400">{label}</span>
      {children}
    </label>
  )
}

function Toggle({
  label, checked, onChange,
}: {
  label: string
  checked: boolean
  onChange: (v: boolean) => void
}) {
  return (
    <label className="flex items-center gap-2 pt-5 cursor-pointer">
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        className="accent-violet-500"
      />
      <span className="text-[13px] text-slate-300 font-mono">{label}</span>
    </label>
  )
}
