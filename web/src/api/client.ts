/**
 * fetch 封装：base /api、X-Api-Token（localStorage）、401 触发 token 弹层、
 * 错误 envelope {error:{code,message,detail}} 统一抛 ApiError。
 */
import type { ApiErrorBody } from './types'

const TOKEN_KEY = 'mow.token'

/** 401 时的 token 弹层事件（App 监听后弹输入框） */
export const TOKEN_REQUIRED_EVENT = 'mow:token-required'

export function getToken(): string {
  return localStorage.getItem(TOKEN_KEY) ?? ''
}

export function setToken(t: string): void {
  if (t) localStorage.setItem(TOKEN_KEY, t)
  else localStorage.removeItem(TOKEN_KEY)
}

/** token 弹层出现时的回调（App 注入，用于重放当前请求） */
let _onTokenRequired: (() => void) | null = null
export function onTokenRequired(fn: (() => void) | null): void {
  _onTokenRequired = fn
}

export class ApiError extends Error {
  code: string
  detail: string

  constructor(code: string, message: string, detail: string = '') {
    super(message)
    this.code = code
    this.detail = detail
  }
}

export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers)
  headers.set('Content-Type', 'application/json')
  const token = getToken()
  if (token) headers.set('X-Api-Token', token)

  let resp: Response
  try {
    resp = await fetch('/api' + path, { ...init, headers })
  } catch (e) {
    throw new ApiError('network', `请求失败：${String(e)}`)
  }

  if (resp.status === 401) {
    window.dispatchEvent(new CustomEvent(TOKEN_REQUIRED_EVENT))
    _onTokenRequired?.()
    throw new ApiError('bad_token', '需要 API token')
  }

  const body = await resp.json().catch(() => null)
  if (!resp.ok) {
    const err = (body as { error?: ApiErrorBody } | null)?.error
    if (err) throw new ApiError(err.code, err.message, err.detail)
    throw new ApiError('http_' + resp.status, resp.statusText)
  }
  return body as T
}
