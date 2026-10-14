/**
 * App 外壳：侧栏（5 区 + GPU 真值卡）+ 顶栏（区标题 + 版本徽章 + 显存指示）
 * + 5 区视图 + token 弹层（401 时出现，输入后重放 bootstrap）。
 */
import { useEffect, useState } from 'react'
import { onTokenRequired, TOKEN_REQUIRED_EVENT } from './api/client'
import { useBootstrap } from './hooks/useBootstrap'
import Sidebar, { VIEWS, type ViewKey } from './components/Sidebar'
import TopBar from './components/TopBar'
import { ToastProvider } from './components/ToastProvider'
import TokenModal from './components/TokenModal'
import PlaceholderView from './components/PlaceholderView'
import JobsPanel from './components/JobsPanel'
import QuantizeView from './views/QuantizeView'
import EvalView from './views/EvalView'
import RecipeView from './views/RecipeView'
import ArtifactsView from './views/ArtifactsView'

export default function App() {
  const [view, setView] = useState<ViewKey>('quantize')
  const { data, error, reload } = useBootstrap()
  const [tokenOpen, setTokenOpen] = useState(false)

  // token 弹层：401 时出现；输入后重放首屏
  useEffect(() => {
    const onReq = () => setTokenOpen(true)
    window.addEventListener(TOKEN_REQUIRED_EVENT, onReq)
    onTokenRequired(() => setTokenOpen(true))
    return () => {
      window.removeEventListener(TOKEN_REQUIRED_EVENT, onReq)
      onTokenRequired(null)
    }
  }, [])

  const title = VIEWS.find((v) => v.key === view)?.label ?? ''
  const probe = data?.probe ?? null
  const caps = data?.capabilities ?? null

  return (
    <div className="flex h-screen bg-slate-950 text-slate-100 font-sans overflow-hidden">
      <Sidebar view={view} onNavigate={setView} probe={probe} />
      <div className="flex-1 flex flex-col min-w-0">
        <TopBar title={title} probe={probe} caps={caps} />
        <main className="flex-1 overflow-y-auto p-6">
          <ToastProvider>
            {error && !data ? (
              <div className="mt-10 mx-auto max-w-lg p-5 rounded-2xl bg-rose-950/40 border border-rose-500/30 space-y-3">
                <div className="text-sm font-semibold text-rose-200">无法连接后端</div>
                <div className="text-[13px] text-rose-300/80 font-mono break-all">{error}</div>
                <div className="text-[12px] text-slate-500">
                  确认 serve.py（7862）在运行；dev 模式下 vite 会把 /api 代理到 127.0.0.1:7862。
                </div>
                <button
                  onClick={() => void reload()}
                  className="px-4 py-1.5 rounded-lg text-[12px] font-medium bg-rose-600/80 text-white hover:bg-rose-500"
                >
                  重试
                </button>
              </div>
            ) : view === 'quantize' ? (
              <div className="space-y-6">
                <QuantizeView data={data} />
                <JobsPanel />
              </div>
            ) : view === 'eval' ? (
              <EvalView />
            ) : view === 'recipe' ? (
              <RecipeView data={data} />
            ) : view === 'artifacts' ? (
              <ArtifactsView />
            ) : (
              <PlaceholderView name="系统设置与日志" note="Task 6：路径 / venv / 代理 / 探活明细" />
            )}
          </ToastProvider>
        </main>
      </div>
      <TokenModal
        open={tokenOpen}
        onDone={(ok) => {
          setTokenOpen(false)
          if (ok) void reload()
        }}
      />
    </div>
  )
}
