/** 确认弹窗：warn_gpu 等危险动作前的二次确认。 */
import { AlertTriangle } from 'lucide-react'

export default function ConfirmModal({
  open, title, body, confirmLabel = '继续', onConfirm, onCancel,
}: {
  open: boolean
  title: string
  body: string
  confirmLabel?: string
  onConfirm: () => void
  onCancel: () => void
}) {
  if (!open) return null
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="w-[26rem] p-5 rounded-2xl bg-slate-900 border border-amber-500/40 space-y-4">
        <div className="flex items-center gap-2 text-amber-300">
          <AlertTriangle size={16} />
          <span className="text-sm font-semibold">{title}</span>
        </div>
        <div className="text-[13px] text-slate-300 leading-relaxed">{body}</div>
        <div className="flex justify-end gap-2">
          <button
            onClick={onCancel}
            className="px-3 py-1.5 rounded-lg text-[12px] text-slate-400 hover:text-slate-200"
          >
            取消
          </button>
          <button
            onClick={onConfirm}
            className="px-4 py-1.5 rounded-lg text-[12px] font-medium bg-amber-600 text-white hover:bg-amber-500"
          >
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  )
}
