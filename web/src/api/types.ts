/** 与 server/schemas.py + webui dataclass 逐字对齐的响应类型（2026-10-14 核对）。 */

/** HostStatus（webui/runner.py:73，asdict 原样输出） */
export interface HostStatus {
  reachable: boolean
  gpu: string
  gpu_util: number | null
  gpu_mem: string
  gpu_busy: boolean
  repo_ok: boolean
  venv_ok: boolean
  uv_ok: boolean
  disk_free: string
  hf_cache: string
  hf_direct_ok: boolean
  hf_proxy_ok: boolean
  proxy_note: string
  error: string
}

/** ProgressReport（webui/progress.py:93） */
export interface ProgressReport {
  stage_idx: number
  stage: string
  pct: number | null
  bar: string
  detail: string
  item: string
  done: boolean
  error: string
}

/** GET /api/vocab（server/api.py _scan_vocab） */
export interface VocabData {
  qformats: string[]
  kv: [string, string][]
  datasets: { key: string; desc: string; gated?: boolean }[]
  combos: Record<string, string>
  source?: string
  error?: string
}

/** GET /api/recipes（server/api.py scan_recipes，运行时扫仓库 modelopt_recipes/） */
export interface RecipeEntry {
  name: string
  group: string
  description: string
}

export interface RecipesData {
  recipes: RecipeEntry[]
  error?: string
}

/** GET /api/capabilities（server/capabilities.py） */
export interface Capabilities {
  version: string
  commit: string
  flags_all: string[]
  mapped: string[]
  unmapped: string[]
  missing: string[]
  error: string
}

/** JobSummary（server/api.py _job_summary） */
export interface JobSummary {
  id: string
  kind: string // quant | eval | venv | eval_env
  state: string // starting | running | done | failed | aborted
  started_at: number // epoch 秒
  export_dir: string
  version: string
  commit: string
  cmd_preview: string
}

/** GET /api/jobs（active 在前、各自 started_at 倒序） */
export interface JobList {
  active: JobSummary[]
  history: JobSummary[]
}

/** GET /api/jobs/{id} */
export interface JobDetail {
  job: JobSummary
  state: string
  log_tail: string
  progress: ProgressReport
  result_tables: string[][] | null
}

/** GET /api/paths */
export interface Paths {
  repo: string
  venv: string
  task_dir: string
  hf_cache: string
}

/** GET /api/net */
export interface Net {
  mode: string // auto | direct | manual
  proxy: string
  hf_token_set?: boolean // 后端只回「是否已设置」，永不回显值
}

/** GET /api/bootstrap（首屏一把） */
export interface Bootstrap {
  paths: Paths
  net: Net
  probe: HostStatus
  vocab: VocabData
  recipes: RecipesData
  capabilities: Capabilities
  jobs: JobSummary[]
  defaults: Record<string, unknown>
}

/** QuantizeParams（server/schemas.py，20 字段，extra=forbid） */
export interface QuantizeParams {
  model: string
  trust_remote_code: boolean
  qformat: string
  recipe?: string
  kv_cache_qformat?: string
  datasets: string[]
  dataset_sizes?: Record<string, number>
  /** 每行 [名称, 条数]；条数可以是 number */
  custom_datasets?: [string | number, string | number][]
  calib_seq?: number
  calib_with_images?: boolean
  export_path?: string
  batch_size?: number
  device?: string
  attn_implementation?: string
  use_fsdp2?: boolean
  cpu_offload?: boolean
  low_memory_mode?: boolean
  gpu_max_mem_percentage?: number
  skip_generate?: boolean
  use_seq_device_map?: boolean
  extra_args?: string
}

/** POST /api/preview */
export interface PreviewResult {
  cmd: string
  pretty: string
  missing_flags: string[]
  caps_error: string
}

/** POST /api/quantize · 202 */
export interface QuantizeStart {
  job_id: string
  job: JobSummary
  warn_gpu: boolean
}

/** EvalStart（server/schemas.py） */
export interface EvalStartParams {
  quant: string
  baseline?: string
  tp?: number
  port?: number
  batch?: string
  tasks?: string[]
}

/** POST /api/eval · 202 */
export interface JobStart {
  job_id: string
  job: JobSummary
}

/** POST /api/recipe/diff（summary = asdict(QuantSummary)） */
export interface RecipeGroup {
  name: string
  bits: string | number
  count: number
  samples: string[]
  more: string
}

export interface RecipeDiffResult {
  summary: {
    source: string
    quant_method: string
    groups: RecipeGroup[]
    ignore: string[]
    files: string
    error: string
  }
  /** 后端直接透传 QuantSummary.groups（与 summary.groups 同形） */
  groups: RecipeGroup[]
  /** 官方对照 diff 文本（official 为空时为空串） */
  diff: string
  /** 导出目录文件清单（纯文本） */
  files: string
}

/** POST /api/eval/baseline */
export interface BaselineResult {
  baseline: string
  message: string
}

/** GET /api/eval/results：rows = [模型, 任务, 指标, 量化, 基线, Δ] */
export interface EvalResults {
  rows: string[][]
  note: string
  files: string[]
}

/** GET /api/models/local（C2 完整版） */
export interface LocalModel {
  id: string
  size: string
  stub: boolean
  /** snapshot hash（refs/main 第一行；空壳/不完整为 ""） */
  revision: string
}

/** GET /api/estimate（显存估算器，设计稿 §6） */
export interface EstimateResult {
  weight_bytes: number
  /** cache / hf_api / unknown，可拼 "+kv_estimate" */
  source: string
  quant_bytes: number
  kv_bytes: number
  vram_total_mib: number
  vram_used_mib: number
  need_gib: number
  fits: boolean
  short_gib: number
  note: string
}

/** GET /api/artifacts（C1 完整版：模型名/qformat 从 job.json.cmd 或 run.log 首行提取） */
export interface Artifact {
  dir: string
  model: string
  qformat: string
  ts: string
  size: string
  state: string
}

/** POST /api/net/test */
export interface NetTestResult {
  ok: boolean
  detail: string
}

/** 统一错误 envelope（server/schemas.py ErrorResponse） */
export interface ApiErrorBody {
  code: string
  message: string
  detail: string
}
