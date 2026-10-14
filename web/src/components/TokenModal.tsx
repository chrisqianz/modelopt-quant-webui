/** token 输入弹层（401 bad_token 时出现；输入后存 localStorage 并重放）。 */
import { useState } from 'react'
import { setToken } from '../api/client'

export default function TokenModal({
  open, onDone,
}: {
  open: boolean
  onDone: (ok: boolean) => void
}) {
  const [val, setVal] = useState('')
  if (!open) return null

  const submit = () => {
    setToken(val.trim())
    onDone(true)
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="w-96 p-6 rounded-2xl bg-slate-900 border border-slate-800 space-y-4">
        <div className="text-sm font-semibold text-slate-100">需要 API token</div>
        <div className="text-[12px] text-slate-500">
          服务器开启了 <span className="font-mono">server.token</span>。输入 token 后可继续（存本机 localStorage）。
        </div>
        <input
          autoFocus
          value={val}
          onChange={(e) => setVal(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && submit()}
          placeholder="X-Api-Token"
          className="w-full px-3 py-2 rounded-lg bg-slate-950 border border-slate-800 font-mono text-[13px] text-slate-100 outline-none focus:border-violet-500/60"
        />
        <div className="flex justify-end gap-2">
          <button
            onClick={() => onDone(false)}
            className="px-3 py-1.5 rounded-lg text-[12px] text-slate-400 hover:text-slate-200"
          >
            取消
          </button>
          <button
            onClick={submit}
            disabled={!val.trim()}
            className="px-4 py-1.5 rounded-lg text-[12px] font-medium bg-violet-600 text-white hover:bg-violet-500 disabled:opacity-40"
          >
            确定
          </button>
        </div>
      </div>
    </div>
  )
}
