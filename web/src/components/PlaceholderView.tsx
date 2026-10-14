/** 占位视图（Task 3-6 逐个替换）。 */
export default function PlaceholderView({ name, note }: { name: string; note: string }) {
  return (
    <div className="mt-16 max-w-md mx-auto p-5 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-2">
      <div className="text-sm font-semibold text-slate-200">{name}</div>
      <div className="text-[12px] text-slate-500">{note}</div>
    </div>
  )
}
