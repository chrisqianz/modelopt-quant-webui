/** 23 端点薄封装（形状与 server/api.py 一一对应，2026-10-14 核对）。 */
import { request } from './client'
import type {
  Artifact, BaselineResult, Bootstrap, Capabilities, EvalResults,
  EstimateResult, EvalStartParams, HostStatus, JobDetail, JobList, JobStart, LocalModel,
  Net, NetTestResult, Paths, PreviewResult, QuantizeParams, QuantizeStart,
  RecipeDiffResult, VocabData,
} from './types'

export const api = {
  // 只读
  bootstrap: () => request<Bootstrap>('/bootstrap'),
  probe: () => request<{ probe: HostStatus; vocab: VocabData }>('/probe', { method: 'POST' }),
  vocab: () => request<VocabData>('/vocab'),
  capabilities: () => request<Capabilities>('/capabilities'),
  jobs: () => request<JobList>('/jobs'),
  jobDetail: (id: string) => request<JobDetail>(`/jobs/${encodeURIComponent(id)}`),
  evalResults: () => request<EvalResults>('/eval/results'),
  paths: () => request<{ paths: Paths }>('/paths'),
  net: () => request<{ net: Net }>('/net'),
  modelsLocal: () => request<{ models: LocalModel[]; error: string }>('/models/local'),
  artifacts: () => request<{ artifacts: Artifact[] }>('/artifacts'),
  estimate: (p: { model: string; qformat?: string; kv_qformat?: string; ctx?: number; batch?: number }) =>
    request<EstimateResult>(
      `/estimate?model=${encodeURIComponent(p.model)}&qformat=${encodeURIComponent(p.qformat ?? '')}` +
      `&kv_qformat=${encodeURIComponent(p.kv_qformat ?? '')}&ctx=${p.ctx ?? 2048}&batch=${p.batch ?? 1}`,
    ),

  // 预览 / 校验
  preview: (p: QuantizeParams) => request<PreviewResult>('/preview', { method: 'POST', body: JSON.stringify(p) }),
  recipeDiff: (jobId: string, official: string) =>
    request<RecipeDiffResult>('/recipe/diff', { method: 'POST', body: JSON.stringify({ job_id: jobId, official }) }),
  evalBaseline: (quantPath: string) =>
    request<BaselineResult>('/eval/baseline', { method: 'POST', body: JSON.stringify({ quant_path: quantPath }) }),
  netTest: () => request<NetTestResult>('/net/test', { method: 'POST' }),

  // 起任务（202）
  quantize: (p: QuantizeParams) => request<QuantizeStart>('/quantize', { method: 'POST', body: JSON.stringify(p) }),
  eval: (p: EvalStartParams) => request<JobStart>('/eval', { method: 'POST', body: JSON.stringify(p) }),
  venv: () => request<JobStart>('/venv', { method: 'POST' }),
  evalEnv: () => request<JobStart>('/eval-env', { method: 'POST' }),

  // 控制 / 持久化
  abort: (id: string) => request<{ message: string }>(`/jobs/${encodeURIComponent(id)}/abort`, { method: 'POST' }),
  putPaths: (p: Paths) => request<{ message: string }>('/paths', { method: 'PUT', body: JSON.stringify(p) }),
  // hf_token：省略=保持已存；""=清空；非空=替换。GET 回的 hf_token_set 不能回传（extra=forbid）。
  putNet: (n: { mode: string; proxy: string; hf_token?: string }) => request<{ message: string }>('/net', { method: 'PUT', body: JSON.stringify(n) }),
}
