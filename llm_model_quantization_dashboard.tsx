import React, { useState, useEffect, useRef } from 'react';
import { 
  Zap, BarChart3, Server, Package, Settings, Cpu, HardDrive, Play, Square, 
  RefreshCw, CheckCircle2, AlertCircle, HelpCircle, Download, Terminal, 
  Sliders, Database, Layers, ArrowRight, Copy, Check, Info, ShieldCheck, 
  Activity, ExternalLink, MessageSquare, ChevronRight, Calculator, Sparkles
} from 'lucide-react';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, 
  RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, Legend, LineChart, Line
} from 'recharts';

const MODELS_LIST = [
  { id: 'llama-3.1-8b', name: 'Meta-Llama-3.1-8B-Instruct', params: 8, defaultVRAM: 16.0, hfRepo: 'meta-llama/Meta-Llama-3.1-8B-Instruct' },
  { id: 'qwen-2.5-14b', name: 'Qwen2.5-14B-Instruct', params: 14, defaultVRAM: 28.0, hfRepo: 'Qwen/Qwen2.5-14B-Instruct' },
  { id: 'deepseek-r1-distill-32b', name: 'DeepSeek-R1-Distill-Qwen-32B', params: 32, defaultVRAM: 64.0, hfRepo: 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B' },
  { id: 'mistral-7b-v0.3', name: 'Mistral-7B-Instruct-v0.3', params: 7, defaultVRAM: 14.5, hfRepo: 'mistralai/Mistral-7B-Instruct-v0.3' },
];

const QUANT_ALGORITHMS = [
  { id: 'AWQ', name: 'AWQ (Activation-aware Weight Quantization)', desc: '保护显著权重，适合高吞吐推理', category: 'PTQ', bits: [4, 8], recommended: true },
  { id: 'GPTQ', name: 'GPTQ (Accurate Post-Training Quantization)', desc: '基于二阶Hessian矩阵信息，高精度压缩', category: 'PTQ', bits: [2, 3, 4, 8] },
  { id: 'GGUF', name: 'GGUF (llama.cpp / K-Quants)', desc: '适合 CPU/混合架构、端侧部署 (Q4_K_M, Q5_K_S)', category: 'PTQ', bits: [2, 3, 4, 5, 6, 8], recommended: true },
  { id: 'EXL2', name: 'EXL2 (ExLlamaV2 Quantization)', desc: '支持任意位宽(例如 3.5-bit)，极致速度', category: 'PTQ', bits: [3, 4, 5, 6] },
  { id: 'SmoothQuant', name: 'SmoothQuant (W8A8 / FP8)', desc: '平滑激活异常值，支持权重与激活同时量化', category: 'FP8/INT8', bits: [8] },
];

const CALIBRATION_DATASETS = [
  { id: 'wikitext2', name: 'WikiText-2 (通用英文测试)', samples: 128, length: 2048 },
  { id: 'c4', name: 'C4 (Colossal Clean Crawled Corpus)', samples: 256, length: 2048 },
  { id: 'ceval_zh', name: 'C-Eval (中文综合知识测试)', samples: 200, length: 2048 },
  { id: 'custom', name: '自定义数据集 (JSONL/Parquet)', samples: 500, length: 4096 },
];

export default function App() {
  const [activeTab, setActiveTab] = useState('quantize'); // quantize, eval, connect, hub, settings
  const [vramCalcOpen, setVramCalcOpen] = useState(false);
  const [toastMessage, setToastMessage] = useState(null);

  const showToast = (msg, type = 'success') => {
    setToastMessage({ text: msg, type });
    setTimeout(() => setToastMessage(null), 3500);
  };

  return (
    <div className="flex h-screen bg-slate-950 text-slate-100 font-sans overflow-hidden select-none">
      {/* Toast Notification */}
      {toastMessage && (
        <div className={`fixed top-5 right-5 z-50 flex items-center gap-3 px-4 py-3 rounded-lg border shadow-xl backdrop-blur-md transition-all duration-300 animate-slide-in ${
          toastMessage.type === 'error' ? 'bg-rose-950/90 border-rose-500/50 text-rose-200' : 'bg-emerald-950/90 border-emerald-500/50 text-emerald-200'
        }`}>
          {toastMessage.type === 'error' ? <AlertCircle className="w-5 h-5 text-rose-400" /> : <CheckCircle2 className="w-5 h-5 text-emerald-400" />}
          <span className="text-sm font-medium">{toastMessage.text}</span>
        </div>
      )}

      {/* Sidebar Navigation */}
      <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} openVramCalc={() => setVramCalcOpen(true)} />

      {/* Main Workspace Area */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
        {/* Top Header Bar */}
        <header className="h-16 border-b border-slate-800/80 bg-slate-900/50 backdrop-blur-md px-6 flex items-center justify-between z-10">
          <div className="flex items-center gap-3">
            <h1 className="text-lg font-bold bg-gradient-to-r from-violet-400 via-indigo-300 to-cyan-400 bg-clip-text text-transparent">
              {activeTab === 'quantize' && '⚡ 大模型量化配置与执行 (Quantization Workspace)'}
              {activeTab === 'eval' && '📊 性能与精度评测 (Evaluation & Benchmark)'}
              {activeTab === 'connect' && '🔌 引擎与集群连接 (Engine & Cluster Connections)'}
              {activeTab === 'hub' && '📦 模型仓库与导出 (Model Hub & Artifacts)'}
              {activeTab === 'settings' && '⚙️ 系统设置与运行日志 (Settings & Runtime Logs)'}
            </h1>
            <span className="px-2 py-0.5 text-xs font-mono rounded-full bg-violet-500/10 text-violet-400 border border-violet-500/30">
              v2.4.0-CUDA12.2
            </span>
          </div>

          <div className="flex items-center gap-3">
            {/* Quick VRAM Calculator Trigger Button */}
            <button
              onClick={() => setVramCalcOpen(true)}
              className="flex items-center gap-2 px-3 py-1.5 text-xs font-medium rounded-lg bg-slate-800/80 hover:bg-slate-700 text-slate-300 border border-slate-700/60 transition-all hover:border-violet-500/50 hover:text-violet-300"
            >
              <Calculator className="w-4 h-4 text-violet-400" />
              <span>VRAM 显存估算器</span>
            </button>

            {/* GPU Node Status Quick Indicator */}
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-800/40 border border-slate-800 text-xs">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
              </span>
              <span className="text-slate-400">Node-A100:</span>
              <span className="font-mono text-emerald-400 font-medium">32.4 / 80 GB</span>
            </div>
          </div>
        </header>

        {/* View Router */}
        <main className="flex-1 overflow-y-auto p-6 relative">
          {activeTab === 'quantize' && <QuantizationView showToast={showToast} />}
          {activeTab === 'eval' && <EvaluationView showToast={showToast} />}
          {activeTab === 'connect' && <ConnectionsView showToast={showToast} />}
          {activeTab === 'hub' && <ModelHubView showToast={showToast} />}
          {activeTab === 'settings' && <SettingsView showToast={showToast} />}
        </main>
      </div>

      {/* Slide-over VRAM Calculator Modal */}
      {vramCalcOpen && <VRAMCalculatorModal onClose={() => setVramCalcOpen(false)} />}
    </div>
  );
}

function Sidebar({ activeTab, setActiveTab }) {
  const navItems = [
    { id: 'quantize', label: '模型量化', sub: 'Quantization', icon: Zap },
    { id: 'eval', label: '性能与精度评测', sub: 'Benchmark', icon: BarChart3 },
    { id: 'connect', label: '引擎与集群连接', sub: 'Connections', icon: Server },
    { id: 'hub', label: '模型仓库与导出', sub: 'Model Hub', icon: Package },
    { id: 'settings', label: '系统设置与日志', sub: 'Settings', icon: Settings },
  ];

  return (
    <aside className="w-64 border-r border-slate-800/80 bg-slate-900/80 flex flex-col justify-between p-4 z-20 backdrop-blur-lg">
      <div className="space-y-6">
        {/* Logo Section */}
        <div className="flex items-center gap-3 px-2 py-1">
          <div className="p-2 rounded-xl bg-gradient-to-tr from-violet-600 to-cyan-500 shadow-lg shadow-violet-500/20">
            <Cpu className="w-6 h-6 text-white" />
          </div>
          <div>
            <div className="font-bold text-sm tracking-wide text-slate-100">QuantStudio AI</div>
            <div className="text-[11px] text-slate-400">大模型量化管理平台</div>
          </div>
        </div>

        {/* Navigation Links */}
        <nav className="space-y-1.5">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`w-full flex items-center justify-between px-3 py-2.5 rounded-xl text-left transition-all group ${
                  isActive 
                    ? 'bg-gradient-to-r from-violet-600/30 to-indigo-600/10 text-white border border-violet-500/40 shadow-sm' 
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
                }`}
              >
                <div className="flex items-center gap-3">
                  <Icon className={`w-4 h-4 transition-colors ${isActive ? 'text-violet-400' : 'text-slate-400 group-hover:text-slate-200'}`} />
                  <div>
                    <div className="text-xs font-semibold">{item.label}</div>
                    <div className="text-[10px] text-slate-500">{item.sub}</div>
                  </div>
                </div>
                {isActive && <ChevronRight className="w-4 h-4 text-violet-400" />}
              </button>
            );
          })}
        </nav>
      </div>

      {/* Hardware Summary Box at Footer */}
      <div className="p-3.5 rounded-xl bg-slate-950/60 border border-slate-800/80 text-xs space-y-2">
        <div className="flex items-center justify-between text-slate-400">
          <span className="flex items-center gap-1.5">
            <HardDrive className="w-3.5 h-3.5 text-cyan-400" /> GPU 集群状态
          </span>
          <span className="text-[10px] text-emerald-400 font-mono">ONLINE</span>
        </div>
        <div className="space-y-1.5 font-mono text-[11px]">
          <div className="flex justify-between text-slate-300">
            <span>NVIDIA A100-80G</span>
            <span>40%</span>
          </div>
          <div className="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
            <div className="bg-gradient-to-r from-cyan-500 to-violet-500 h-full rounded-full" style={{ width: '40%' }}></div>
          </div>
          <div className="text-[10px] text-slate-500 flex justify-between pt-0.5">
            <span>Temp: 48°C</span>
            <span>Power: 180W</span>
          </div>
        </div>
      </div>
    </aside>
  );
}

function QuantizationView({ showToast }) {
  const [selectedModel, setSelectedModel] = useState(MODELS_LIST[0]);
  const [selectedAlgo, setSelectedAlgo] = useState(QUANT_ALGORITHMS[0]);
  const [selectedBits, setSelectedBits] = useState(4);
  const [groupSize, setGroupSize] = useState(128);
  const [symmetric, setSymmetric] = useState(true);
  const [calibrationDataset, setCalibrationDataset] = useState(CALIBRATION_DATASETS[0]);
  const [ggufArch, setGgufArch] = useState('Q4_K_M');

  // Job Execution State
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [currentStep, setCurrentStep] = useState(0);
  const [logs, setLogs] = useState([]);
  const logsEndRef = useRef(null);

  const steps = [
    '加载模型权重 (Weights Loading)',
    '校准数据集采提 (Calibration Data Feed)',
    'Hessian 矩阵/激活分布计算',
    '权重矩阵量化与反量化误差校准',
    '生成并导出量化 Artifact 文件'
  ];

  const handleStartQuantization = () => {
    setIsProcessing(true);
    setProgress(0);
    setCurrentStep(0);
    setLogs([`[System] 开始准备大模型量化任务: ${selectedModel.name}`]);

    let currentProgress = 0;
    const interval = setInterval(() => {
      currentProgress += 2;
      setProgress(currentProgress);

      if (currentProgress === 10) {
        setCurrentStep(0);
        addLog(`[Info] 正在加载基础模型权重 (${selectedModel.hfRepo})...`);
      } else if (currentProgress === 30) {
        setCurrentStep(1);
        addLog(`[Info] 载入校准数据集 [${calibrationDataset.name}]，计算激活范围...`);
      } else if (currentProgress === 55) {
        setCurrentStep(2);
        addLog(`[Algo] 运行算法: ${selectedAlgo.id} (${selectedBits}-bit, Group Size: ${groupSize})...`);
        addLog(`[CUDA] 计算 Hessian 二阶偏导误差矩阵中...`);
      } else if (currentProgress === 80) {
        setCurrentStep(3);
        addLog(`[Quant] 对 Transformer Block 权重进行量化，最小化 FP16/INT4 距离...`);
      } else if (currentProgress === 95) {
        setCurrentStep(4);
        addLog(`[IO] 写入量化权重文件 -> ./models/${selectedModel.id}-${selectedAlgo.id.toLowerCase()}-${selectedBits}bit.safetensors`);
      } else if (currentProgress >= 100) {
        clearInterval(interval);
        setIsProcessing(false);
        addLog(`[Success] 🎉 量化任务完美完成！压降显存 ~68%，就绪可用于推理服务。`);
        showToast(`模型 ${selectedModel.name} 量化成功！`);
      }
    }, 150);
  };

  const addLog = (msg) => {
    setLogs((prev) => [...prev, `${new Date().toLocaleTimeString()} ${msg}`]);
  };

  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  // VRAM Savings Calculation Estimation
  const originalVRAM = selectedModel.defaultVRAM;
  const quantRatio = (selectedBits / 16) * 1.15; // + Overhead
  const estimatedQuantVRAM = (originalVRAM * quantRatio).toFixed(1);
  const savedVramPercent = Math.round((1 - estimatedQuantVRAM / originalVRAM) * 100);

  return (
    <div className="grid grid-cols-12 gap-6 pb-8">
      {/* Left Configuration Panel (7 columns) */}
      <div className="col-span-12 lg:col-span-7 space-y-6">
        
        {/* Step 1: Select Model */}
        <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-4 backdrop-blur-md">
          <div className="flex items-center justify-between">
            <label className="text-sm font-semibold flex items-center gap-2 text-slate-200">
              <span className="w-6 h-6 rounded-full bg-violet-600/30 text-violet-400 text-xs flex items-center justify-center font-mono border border-violet-500/40">1</span>
              选择待量化大模型 (Baseline Model)
            </label>
            <span className="text-xs text-slate-400 font-mono">支持 HuggingFace / Local</span>
          </div>

          <div className="grid grid-cols-2 gap-3">
            {MODELS_LIST.map((model) => (
              <div
                key={model.id}
                onClick={() => setSelectedModel(model)}
                className={`p-3.5 rounded-xl border cursor-pointer transition-all ${
                  selectedModel.id === model.id
                    ? 'border-violet-500/80 bg-violet-950/20 ring-1 ring-violet-500/40'
                    : 'border-slate-800 bg-slate-950/40 hover:border-slate-700 hover:bg-slate-800/40'
                }`}
              >
                <div className="font-semibold text-xs text-slate-100 truncate">{model.name}</div>
                <div className="text-[11px] text-slate-400 mt-1 flex justify-between">
                  <span>{model.params}B 参数量</span>
                  <span className="font-mono text-cyan-400">FP16 ~{model.defaultVRAM}GB</span>
                </div>
              </div>
            ))}
          </div>

          <div className="pt-2">
            <input
              type="text"
              placeholder="或输入 HuggingFace Repo ID (例如: deepseek-ai/DeepSeek-V3)"
              className="w-full px-3.5 py-2 rounded-xl bg-slate-950 border border-slate-800 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-violet-500/60 transition-colors"
            />
          </div>
        </div>

        {/* Step 2: Select Quantization Algorithm */}
        <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-4 backdrop-blur-md">
          <div className="flex items-center justify-between">
            <label className="text-sm font-semibold flex items-center gap-2 text-slate-200">
              <span className="w-6 h-6 rounded-full bg-violet-600/30 text-violet-400 text-xs flex items-center justify-center font-mono border border-violet-500/40">2</span>
              选择量化算法 (Quantization Algorithm)
            </label>
          </div>

          <div className="space-y-2.5">
            {QUANT_ALGORITHMS.map((algo) => (
              <div
                key={algo.id}
                onClick={() => {
                  setSelectedAlgo(algo);
                  if (!algo.bits.includes(selectedBits)) {
                    setSelectedBits(algo.bits[0]);
                  }
                }}
                className={`p-3.5 rounded-xl border cursor-pointer transition-all flex items-start justify-between ${
                  selectedAlgo.id === algo.id
                    ? 'border-violet-500/80 bg-violet-950/20 ring-1 ring-violet-500/40'
                    : 'border-slate-800 bg-slate-950/40 hover:border-slate-700'
                }`}
              >
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <span className="font-bold text-xs text-slate-100">{algo.name}</span>
                    {algo.recommended && (
                      <span className="px-1.5 py-0.5 text-[9px] font-semibold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 rounded">
                        推荐
                      </span>
                    )}
                  </div>
                  <div className="text-[11px] text-slate-400">{algo.desc}</div>
                </div>
                <span className="text-[10px] font-mono px-2 py-0.5 bg-slate-800 text-slate-300 rounded">
                  {algo.category}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Step 3: Parameter Tuning */}
        <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-4 backdrop-blur-md">
          <label className="text-sm font-semibold flex items-center gap-2 text-slate-200">
            <span className="w-6 h-6 rounded-full bg-violet-600/30 text-violet-400 text-xs flex items-center justify-center font-mono border border-violet-500/40">3</span>
            细粒度参数调优 (Hyperparameters)
          </label>

          <div className="grid grid-cols-2 gap-4 pt-1">
            {/* Bit Precision */}
            <div className="space-y-2">
              <div className="flex justify-between text-xs text-slate-300">
                <span>目标位宽 (Bit Precision)</span>
                <span className="font-mono text-cyan-400 font-bold">{selectedBits}-bit</span>
              </div>
              <div className="flex gap-2">
                {selectedAlgo.bits.map((b) => (
                  <button
                    key={b}
                    onClick={() => setSelectedBits(b)}
                    className={`flex-1 py-1.5 text-xs font-mono rounded-lg border transition-all ${
                      selectedBits === b
                        ? 'bg-violet-600 text-white border-violet-500'
                        : 'bg-slate-950 text-slate-400 border-slate-800 hover:border-slate-700'
                    }`}
                  >
                    {b}-bit
                  </button>
                ))}
              </div>
            </div>

            {/* Group Size */}
            {selectedAlgo.id !== 'GGUF' ? (
              <div className="space-y-2">
                <div className="flex justify-between text-xs text-slate-300">
                  <span>Group Size (分组大小)</span>
                  <span className="font-mono text-cyan-400 font-bold">{groupSize}</span>
                </div>
                <div className="flex gap-2">
                  {[32, 64, 128, -1].map((gs) => (
                    <button
                      key={gs}
                      onClick={() => setGroupSize(gs)}
                      className={`flex-1 py-1.5 text-xs font-mono rounded-lg border transition-all ${
                        groupSize === gs
                          ? 'bg-violet-600 text-white border-violet-500'
                          : 'bg-slate-950 text-slate-400 border-slate-800 hover:border-slate-700'
                      }`}
                    >
                      {gs === -1 ? 'None' : gs}
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              <div className="space-y-2">
                <div className="flex justify-between text-xs text-slate-300">
                  <span>GGUF 预设方案 (K-Quants)</span>
                </div>
                <select
                  value={ggufArch}
                  onChange={(e) => setGgufArch(e.target.value)}
                  className="w-full px-3 py-1.5 rounded-lg bg-slate-950 border border-slate-800 text-xs font-mono text-slate-200 focus:outline-none"
                >
                  <option value="Q4_K_M">Q4_K_M (中度平衡 - 推荐)</option>
                  <option value="Q4_K_S">Q4_K_S (小型体积)</option>
                  <option value="Q5_K_M">Q5_K_M (高精度，稍大体积)</option>
                  <option value="Q8_0">Q8_0 (近乎无损 FP16 精度)</option>
                </select>
              </div>
            )}
          </div>

          {/* Calibration Dataset */}
          <div className="space-y-2 pt-2">
            <label className="text-xs text-slate-300 flex justify-between">
              <span>校准数据集 (Calibration Dataset)</span>
              <span className="text-[11px] text-slate-500">影响 Hessian 矩阵采样精确度</span>
            </label>
            <select
              value={calibrationDataset.id}
              onChange={(e) => setCalibrationDataset(CALIBRATION_DATASETS.find(d => d.id === e.target.value))}
              className="w-full px-3 py-2 rounded-xl bg-slate-950 border border-slate-800 text-xs text-slate-200 focus:outline-none focus:border-violet-500/60"
            >
              {CALIBRATION_DATASETS.map((ds) => (
                <option key={ds.id} value={ds.id}>
                  {ds.name} — {ds.samples} 样本 / Context {ds.length}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Right Column: Dynamic Preview, VRAM Gauge, Exec Panel (5 columns) */}
      <div className="col-span-12 lg:col-span-5 space-y-6">
        
        {/* Estimated VRAM Savings Card */}
        <div className="p-5 rounded-2xl bg-gradient-to-br from-slate-900 to-slate-950 border border-slate-800 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-bold text-slate-300 uppercase tracking-wider flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-violet-400" /> 显存与效果估算 Preview
            </h3>
            <span className="text-[11px] font-mono text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20">
              节约 ~{savedVramPercent}% VRAM
            </span>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="p-3.5 rounded-xl bg-slate-950 border border-slate-800 text-center">
              <div className="text-[10px] text-slate-500 uppercase font-medium">原始 FP16 显存需求</div>
              <div className="text-xl font-mono font-bold text-slate-300 mt-1">{originalVRAM} GB</div>
            </div>
            <div className="p-3.5 rounded-xl bg-violet-950/30 border border-violet-500/30 text-center">
              <div className="text-[10px] text-violet-300 uppercase font-medium">量化后显存 ({selectedBits}-bit)</div>
              <div className="text-xl font-mono font-bold text-cyan-400 mt-1">{estimatedQuantVRAM} GB</div>
            </div>
          </div>

          {/* Visual Gauge Bar */}
          <div className="space-y-1.5 pt-1">
            <div className="flex justify-between text-[11px] text-slate-400">
              <span>显存占用压降对比</span>
              <span>{estimatedQuantVRAM}GB / {originalVRAM}GB</span>
            </div>
            <div className="w-full bg-slate-800 rounded-full h-3 p-0.5 overflow-hidden flex">
              <div 
                className="bg-gradient-to-r from-cyan-500 to-violet-500 h-full rounded-full transition-all duration-500" 
                style={{ width: `${(estimatedQuantVRAM / originalVRAM) * 100}%` }}
              ></div>
            </div>
          </div>
        </div>

        {/* Real-time Job Execution Widget */}
        <div className="p-5 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-4 shadow-xl">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Terminal className="w-4 h-4 text-cyan-400" />
              <h3 className="text-sm font-semibold text-slate-200">任务构建与运行终端</h3>
            </div>
            {isProcessing && (
              <span className="flex items-center gap-1.5 text-xs text-amber-400 font-mono animate-pulse">
                <RefreshCw className="w-3.5 h-3.5 animate-spin" /> 正在量化...
              </span>
            )}
          </div>

          {/* Action Button */}
          <button
            onClick={handleStartQuantization}
            disabled={isProcessing}
            className={`w-full py-3 px-4 rounded-xl font-semibold text-sm flex items-center justify-center gap-2 transition-all shadow-lg ${
              isProcessing
                ? 'bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-700'
                : 'bg-gradient-to-r from-violet-600 via-indigo-600 to-cyan-600 hover:from-violet-500 hover:to-cyan-500 text-white shadow-violet-600/25 border border-violet-400/30'
            }`}
          >
            {isProcessing ? (
              <>
                <RefreshCw className="w-4 h-4 animate-spin" />
                正在执行 CUDA 量化计算 ({progress}%)
              </>
            ) : (
              <>
                <Play className="w-4 h-4 fill-current" />
                启动大模型量化任务 (Start Quantization)
              </>
            )}
          </button>

          {/* Progress Timeline Pipeline */}
          <div className="space-y-2 pt-2">
            <div className="text-xs text-slate-400 font-medium">处理流水线 Phase</div>
            <div className="space-y-1.5">
              {steps.map((step, idx) => (
                <div key={idx} className="flex items-center gap-2 text-xs">
                  <div className={`w-4 h-4 rounded-full flex items-center justify-center text-[9px] font-mono ${
                    currentStep > idx ? 'bg-emerald-500 text-slate-950 font-bold' :
                    currentStep === idx && isProcessing ? 'bg-amber-500 text-slate-950 animate-bounce font-bold' :
                    'bg-slate-800 text-slate-500'
                  }`}>
                    {currentStep > idx ? '✓' : idx + 1}
                  </div>
                  <span className={currentStep === idx && isProcessing ? 'text-amber-300 font-medium' : currentStep > idx ? 'text-slate-300 line-through opacity-75' : 'text-slate-500'}>
                    {step}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Simulated Terminal Screen */}
          <div className="bg-slate-950 rounded-xl p-3 border border-slate-800 font-mono text-[11px] h-40 overflow-y-auto space-y-1 text-slate-400 scrollbar-thin">
            {logs.length === 0 ? (
              <div className="text-slate-600 italic">点击“启动大模型量化任务”按钮观察实时 CUDA 执行日志...</div>
            ) : (
              logs.map((log, index) => (
                <div key={index} className={log.includes('Success') ? 'text-emerald-400 font-bold' : log.includes('Algo') ? 'text-cyan-300' : ''}>
                  {log}
                </div>
              ))
            )}
            <div ref={logsEndRef} />
          </div>
        </div>

      </div>
    </div>
  );
}

function EvaluationView({ showToast }) {
  const [promptInput, setPromptInput] = useState('请用中文简要解释量子计算的基本原理，以及为什么它在特定领域能远超经典计算机？');
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [origOutput, setOrigOutput] = useState('');
  const [quantOutput, setQuantOutput] = useState('');
  const [origStats, setOrigStats] = useState(null);
  const [quantStats, setQuantStats] = useState(null);

  // Radar chart metrics data
  const radarData = [
    { subject: 'MMLU 综合推理', FP16: 92, INT4_AWQ: 89.5, GGUF_Q4: 87.2 },
    { subject: 'GSM8K 数学逻辑', FP16: 88, INT4_AWQ: 86.0, GGUF_Q4: 83.8 },
    { subject: 'HumanEval 代码', FP16: 85, INT4_AWQ: 84.1, GGUF_Q4: 81.0 },
    { subject: 'C-Eval 中文能力', FP16: 90, INT4_AWQ: 88.8, GGUF_Q4: 86.5 },
    { subject: '长文本关联理解', FP16: 94, INT4_AWQ: 91.2, GGUF_Q4: 88.0 },
  ];

  // Perplexity vs Compression bar data
  const pplData = [
    { name: 'FP16 (基准)', PPL: 6.12, MemoryGB: 16.0, SpeedTPS: 38 },
    { name: 'INT8 SmoothQuant', PPL: 6.15, MemoryGB: 8.5, SpeedTPS: 55 },
    { name: 'INT4 AWQ', PPL: 6.28, MemoryGB: 4.8, SpeedTPS: 82 },
    { name: 'GGUF Q4_K_M', PPL: 6.34, MemoryGB: 4.6, SpeedTPS: 76 },
    { name: 'EXL2 3.5bit', PPL: 6.65, MemoryGB: 3.9, SpeedTPS: 95 },
  ];

  const handleRunComparison = () => {
    if (!promptInput.trim()) return;
    setIsEvaluating(true);
    setOrigOutput('');
    setQuantOutput('');
    setOrigStats(null);
    setQuantStats(null);

    const fullOrigText = "量子计算是一种利用量子力学规律（如量子叠加态和量子纠缠）进行数据处理的全新计算模式。经典计算机以二进制位（0或1）处理数据，而量子计算机使用量子位（Qubit），可同时处于0和1的叠加状态。因此，N个量子位能并行表示2^N种状态，在如大数因子分解、分子模拟和复杂优化问题上呈现指数级加速。";
    const fullQuantText = "量子计算是基于量子力学特征（量子叠加与纠缠）的新型计算技术。不同于传统二进制（非0即1）的比特，量子比特（Qubit）能以叠加态同时记录0与1。这赋予了量子计算极高的并行处理能力，可以在密码破译、材料科学仿真等高复杂度计算场景中，获得相比传统超级计算机数万倍的速度提升。";

    let idx1 = 0;
    let idx2 = 0;

    const timer1 = setInterval(() => {
      if (idx1 < fullOrigText.length) {
        setOrigOutput((prev) => prev + fullOrigText[idx1]);
        idx1++;
      } else {
        clearInterval(timer1);
        setOrigStats({ tps: 36.2, latency: '420ms', vram: '15.8 GB' });
      }
    }, 40);

    const timer2 = setInterval(() => {
      if (idx2 < fullQuantText.length) {
        setQuantOutput((prev) => prev + fullQuantText[idx2]);
        idx2++;
      } else {
        clearInterval(timer2);
        setQuantStats({ tps: 84.6, latency: '180ms', vram: '4.8 GB' });
        setIsEvaluating(false);
        showToast('同屏实时对比测试完成！');
      }
    }, 20);
  };

  return (
    <div className="space-y-6 pb-8">
      
      {/* Visual Analytics Grid */}
      <div className="grid grid-cols-12 gap-6">
        
        {/* Radar Chart Component */}
        <div className="col-span-12 lg:col-span-6 p-5 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-3">
          <div className="flex justify-between items-center">
            <h3 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
              <Activity className="w-4 h-4 text-violet-400" /> 多维度任务准确率对比 (Accuracy Radar)
            </h3>
            <span className="text-[11px] text-slate-400">测试集: MMLU / GSM8K / C-Eval</span>
          </div>

          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart cx="50%" cy="50%" outerRadius="80%" data={radarData}>
                <PolarGrid stroke="#334155" />
                <PolarAngleAxis dataKey="subject" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                <PolarRadiusAxis angle={30} domain={[60, 100]} tick={{ fill: '#64748b', fontSize: 10 }} />
                <Radar name="FP16 原始" dataKey="FP16" stroke="#818cf8" fill="#818cf8" fillOpacity={0.2} />
                <Radar name="INT4 AWQ" dataKey="INT4_AWQ" stroke="#38bdf8" fill="#38bdf8" fillOpacity={0.3} />
                <Radar name="GGUF Q4_K_M" dataKey="GGUF_Q4" stroke="#34d399" fill="#34d399" fillOpacity={0.2} />
                <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '8px' }} />
                <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '11px' }} />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Perplexity & Memory Bar Chart Component */}
        <div className="col-span-12 lg:col-span-6 p-5 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-3">
          <div className="flex justify-between items-center">
            <h3 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
              <BarChart3 className="w-4 h-4 text-cyan-400" /> 困惑度 PPL 损失 vs 吞吐速度 (Tokens/s)
            </h3>
            <span className="text-[11px] text-slate-400">PPL 越低越好 | TPS 越高越好</span>
          </div>

          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={pplData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 10 }} />
                <YAxis yAxisId="left" orientation="left" stroke="#38bdf8" tick={{ fill: '#94a3b8', fontSize: 10 }} domain={[0, 120]} />
                <YAxis yAxisId="right" orientation="right" stroke="#f43f5e" tick={{ fill: '#94a3b8', fontSize: 10 }} domain={[5, 8]} />
                <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '11px' }} />
                <Bar yAxisId="left" dataKey="SpeedTPS" name="推理速度 (Tokens/s)" fill="#38bdf8" radius={[4, 4, 0, 0]} />
                <Bar yAxisId="right" dataKey="PPL" name="困惑度 PPL (WikiText2)" fill="#f43f5e" radius={[4, 4, 0, 0]} />
                <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '8px' }} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

      </div>

      {/* Side-by-side Live Inference Test Sandbox */}
      <div className="p-5 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-4 shadow-xl">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <MessageSquare className="w-4 h-4 text-emerald-400" />
            <h3 className="text-sm font-semibold text-slate-200">同屏双模型生成效果与延时沙盒 (Side-by-side Live Prompt Sandbox)</h3>
          </div>
          <button
            onClick={handleRunComparison}
            disabled={isEvaluating}
            className={`px-4 py-2 rounded-xl text-xs font-semibold flex items-center gap-2 transition-all ${
              isEvaluating 
                ? 'bg-slate-800 text-slate-500 cursor-not-allowed' 
                : 'bg-emerald-600 hover:bg-emerald-500 text-white shadow-lg shadow-emerald-600/20'
            }`}
          >
            {isEvaluating ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5 fill-current" />}
            同时向两模型发送 Prompt
          </button>
        </div>

        {/* Input Prompt Box */}
        <div>
          <textarea
            value={promptInput}
            onChange={(e) => setPromptInput(e.target.value)}
            rows={2}
            placeholder="输入 Prompt 测试量化前后输出准确度与速度..."
            className="w-full px-4 py-2.5 rounded-xl bg-slate-950 border border-slate-800 text-xs text-slate-200 focus:outline-none focus:border-emerald-500/60"
          ></textarea>
        </div>

        {/* Split Screen Models Comparison */}
        <div className="grid grid-cols-2 gap-4">
          
          {/* Left: FP16 Baseline */}
          <div className="p-4 rounded-xl bg-slate-950 border border-slate-800/80 space-y-3 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between border-b border-slate-800/80 pb-2.5">
                <span className="font-bold text-xs text-indigo-400 flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-indigo-500"></span> 原始 FP16 Baseline
                </span>
                <span className="text-[10px] font-mono text-slate-500">Llama-3.1-8B (FP16)</span>
              </div>
              <div className="mt-3 text-xs text-slate-300 leading-relaxed font-sans min-h-[100px]">
                {origOutput || <span className="text-slate-600 italic">等待发起对比测试...</span>}
              </div>
            </div>

            {origStats && (
              <div className="pt-2 border-t border-slate-800/60 flex items-center justify-between text-[11px] font-mono text-slate-400">
                <span>生成吞吐: <strong className="text-slate-200">{origStats.tps} tok/s</strong></span>
                <span>首 Token 延时: <strong className="text-slate-200">{origStats.latency}</strong></span>
                <span>显存: <strong className="text-slate-200">{origStats.vram}</strong></span>
              </div>
            )}
          </div>

          {/* Right: Quantized Model */}
          <div className="p-4 rounded-xl bg-slate-950 border border-violet-500/30 space-y-3 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between border-b border-slate-800/80 pb-2.5">
                <span className="font-bold text-xs text-cyan-400 flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse"></span> 量化模型 (INT4 AWQ)
                </span>
                <span className="text-[10px] font-mono text-cyan-300/70">Llama-3.1-8B-AWQ-INT4</span>
              </div>
              <div className="mt-3 text-xs text-slate-300 leading-relaxed font-sans min-h-[100px]">
                {quantOutput || <span className="text-slate-600 italic">等待发起对比测试...</span>}
              </div>
            </div>

            {quantStats && (
              <div className="pt-2 border-t border-slate-800/60 flex items-center justify-between text-[11px] font-mono text-cyan-400">
                <span>生成吞吐: <strong className="text-emerald-400">{quantStats.tps} tok/s (2.3x)</strong></span>
                <span>首 Token 延时: <strong className="text-emerald-400">{quantStats.latency}</strong></span>
                <span>显存: <strong className="text-emerald-400">{quantStats.vram} (-70%)</strong></span>
              </div>
            )}
          </div>

        </div>
      </div>

    </div>
  );
}

function ConnectionsView({ showToast }) {
  const [servers, setServers] = useState([
    { id: 1, name: 'Local GPU Cluster - Node A100', type: 'NVIDIA A100 SXM4 80GB', status: 'online', ping: '4ms', vramUsed: '32.4 GB', vramTotal: '80 GB', role: 'Quantization & Benchmark' },
    { id: 2, name: 'vLLM High-Performance Endpoint', type: 'vLLM Serving Engine v0.6.2', status: 'online', ping: '12ms', vramUsed: '18.2 GB', vramTotal: '24 GB', role: 'Inference Backend' },
    { id: 3, name: 'Ollama Edge Runtime', type: 'Ollama Local Daemon', status: 'online', ping: '8ms', vramUsed: '4.8 GB', vramTotal: '16 GB', role: 'Desktop Testing' },
    { id: 4, name: 'Hugging Face Hub Token Sync', type: 'HF API Gateway', status: 'connected', ping: '110ms', vramUsed: 'N/A', vramTotal: 'Cloud', role: 'Model Downloads' },
  ]);

  const [testingId, setTestingId] = useState(null);

  const handleTestConnection = (id) => {
    setTestingId(id);
    setTimeout(() => {
      setTestingId(null);
      showToast('服务器健康检查成功，连接正常！');
    }, 1000);
  };

  return (
    <div className="space-y-6 pb-8">
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-base font-bold text-slate-100">底层推理引擎与算力节点管理</h2>
          <p className="text-xs text-slate-400">配置本地 GPU、vLLM/Ollama 推理引擎及 Hugging Face API 凭证</p>
        </div>
        <button 
          onClick={() => showToast('已成功刷新所有节点状态')}
          className="px-3.5 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-xs font-medium text-slate-200 border border-slate-700 flex items-center gap-2"
        >
          <RefreshCw className="w-3.5 h-3.5" /> 刷新集群状态
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {servers.map((server) => (
          <div key={server.id} className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-4 hover:border-slate-700 transition-all">
            <div className="flex items-start justify-between">
              <div className="flex items-center gap-3">
                <div className="p-2.5 rounded-xl bg-slate-800 border border-slate-700 text-violet-400">
                  <Server className="w-5 h-5" />
                </div>
                <div>
                  <h3 className="font-bold text-xs text-slate-100">{server.name}</h3>
                  <div className="text-[11px] text-slate-400 mt-0.5">{server.type}</div>
                </div>
              </div>
              <span className="flex items-center gap-1.5 text-[10px] font-mono px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-ping"></span>
                {server.status.toUpperCase()}
              </span>
            </div>

            <div className="grid grid-cols-3 gap-2 py-2 px-3 bg-slate-950 rounded-xl text-[11px] font-mono">
              <div>
                <div className="text-slate-500 text-[10px]">网络延时</div>
                <div className="text-slate-200 font-semibold mt-0.5">{server.ping}</div>
              </div>
              <div>
                <div className="text-slate-500 text-[10px]">显存占有</div>
                <div className="text-cyan-400 font-semibold mt-0.5">{server.vramUsed}</div>
              </div>
              <div>
                <div className="text-slate-500 text-[10px]">集群角色</div>
                <div className="text-slate-300 font-semibold mt-0.5 truncate">{server.role}</div>
              </div>
            </div>

            <div className="flex justify-between items-center pt-1">
              <span className="text-[11px] text-slate-500">Node ID: node-gpu-0{server.id}</span>
              <button
                onClick={() => handleTestConnection(server.id)}
                disabled={testingId === server.id}
                className="px-3 py-1.5 text-xs rounded-lg bg-violet-600/20 hover:bg-violet-600/30 text-violet-300 border border-violet-500/30 font-medium transition-all flex items-center gap-1.5"
              >
                {testingId === server.id ? <RefreshCw className="w-3 h-3 animate-spin" /> : <ShieldCheck className="w-3.5 h-3.5" />}
                测试连通性
              </button>
            </div>
          </div>
        ))}
      </div>

      {/* Hugging Face Token Configuration Panel */}
      <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-4">
        <h3 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
          <ExternalLink className="w-4 h-4 text-amber-400" /> Hugging Face API Token & Mirror 设置
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="space-y-1.5">
            <label className="text-xs text-slate-400">HF User Access Token (hf_****)</label>
            <input 
              type="password" 
              defaultValue="hf_x89aFkk20914LkkqM1290aa"
              className="w-full px-3.5 py-2 rounded-xl bg-slate-950 border border-slate-800 text-xs font-mono text-slate-200 focus:outline-none focus:border-violet-500" 
            />
          </div>
          <div className="space-y-1.5">
            <label className="text-xs text-slate-400">HF 镜像源 (HF_ENDPOINT)</label>
            <input 
              type="text" 
              defaultValue="https://hf-mirror.com"
              className="w-full px-3.5 py-2 rounded-xl bg-slate-950 border border-slate-800 text-xs font-mono text-slate-200 focus:outline-none focus:border-violet-500" 
            />
          </div>
        </div>
      </div>
    </div>
  );
}

function ModelHubView({ showToast }) {
  const [copiedScriptId, setCopiedScriptId] = useState(null);

  const artifacts = [
    { id: 'art-1', name: 'Llama-3.1-8B-Instruct-AWQ-INT4', size: '4.8 GB', algo: 'AWQ', precision: 'INT4', format: 'Safetensors', date: '2026-10-12 14:20' },
    { id: 'art-2', name: 'Qwen2.5-14B-GGUF-Q4_K_M', size: '8.2 GB', algo: 'GGUF', precision: 'Q4_K_M', format: '.gguf', date: '2026-10-11 09:15' },
    { id: 'art-3', name: 'DeepSeek-R1-Distill-32B-EXL2-4bit', size: '17.1 GB', algo: 'EXL2', precision: '4.0-bit', format: 'ExLlamaV2', date: '2026-10-08 18:40' },
  ];

  const copyDeploymentScript = (id, scriptText) => {
    navigator.clipboard.writeText(scriptText);
    setCopiedScriptId(id);
    showToast('部署命令已复制到剪贴板！');
    setTimeout(() => setCopiedScriptId(null), 2500);
  };

  return (
    <div className="space-y-6 pb-8">
      <div>
        <h2 className="text-base font-bold text-slate-100">已量化 Artifact 成果库与一键部署 Export</h2>
        <p className="text-xs text-slate-400">查看导出的量化权重文件，快速生成 vLLM、Ollama Modelfile 和 llama.cpp 启动脚本</p>
      </div>

      <div className="space-y-4">
        {artifacts.map((art) => {
          const vllmCmd = `python3 -m vllm.entrypoints.openai.api_server --model ./models/${art.name} --quantization ${art.algo.toLowerCase()} --port 8000`;
          const ollamaModelfile = `FROM ./models/${art.name}.gguf\nPARAMETER stop "<|im_end|>"`;

          return (
            <div key={art.id} className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-4 hover:border-slate-700 transition-all">
              <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800/80 pb-3">
                <div className="flex items-center gap-3">
                  <div className="p-2 rounded-xl bg-violet-600/20 border border-violet-500/30 text-violet-400">
                    <Package className="w-5 h-5" />
                  </div>
                  <div>
                    <h3 className="font-bold text-sm text-slate-100">{art.name}</h3>
                    <div className="text-[11px] text-slate-400 flex items-center gap-3 mt-0.5">
                      <span>格式: <strong className="text-slate-200">{art.format}</strong></span>
                      <span>大小: <strong className="text-cyan-400 font-mono">{art.size}</strong></span>
                      <span>创建时间: {art.date}</span>
                    </div>
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  <span className="px-2.5 py-1 text-xs font-mono rounded-lg bg-slate-800 text-slate-300 border border-slate-700">
                    {art.algo} {art.precision}
                  </span>
                  <button className="px-3 py-1.5 text-xs font-medium rounded-lg bg-violet-600 hover:bg-violet-500 text-white transition-all flex items-center gap-1.5">
                    <Download className="w-3.5 h-3.5" /> 下载权重
                  </button>
                </div>
              </div>

              {/* Deployment Commands Helper */}
              <div className="space-y-2">
                <div className="text-xs text-slate-300 font-semibold flex items-center gap-1.5">
                  <Terminal className="w-3.5 h-3.5 text-cyan-400" /> 一键生成推理服务启动命令 (vLLM Launch Script)
                </div>
                <div className="flex items-center justify-between bg-slate-950 p-3 rounded-xl border border-slate-800 font-mono text-xs text-cyan-300">
                  <span className="truncate mr-4">{vllmCmd}</span>
                  <button
                    onClick={() => copyDeploymentScript(art.id, vllmCmd)}
                    className="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-200 text-[11px] flex items-center gap-1 shrink-0"
                  >
                    {copiedScriptId === art.id ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                    {copiedScriptId === art.id ? '已复制' : '复制命令'}
                  </button>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function SettingsView({ showToast }) {
  return (
    <div className="space-y-6 max-w-4xl pb-8">
      <div>
        <h2 className="text-base font-bold text-slate-100">全局 CUDA 与量化引擎设置</h2>
        <p className="text-xs text-slate-400">调整底层算子编译选项与临时缓存存储位置</p>
      </div>

      <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-4">
        <h3 className="text-sm font-semibold text-slate-200">CUDA & Tensor Core 加速开关</h3>
        
        <div className="space-y-3">
          <div className="flex items-center justify-between py-2 border-b border-slate-800/60">
            <div>
              <div className="text-xs font-semibold text-slate-200">启用 FlashAttention-2 算子加速</div>
              <div className="text-[11px] text-slate-400">提升长文本上下文在校准与评估阶段的吞吐速度</div>
            </div>
            <input type="checkbox" defaultChecked className="w-4 h-4 accent-violet-600 rounded" />
          </div>

          <div className="flex items-center justify-between py-2 border-b border-slate-800/60">
            <div>
              <div className="text-xs font-semibold text-slate-200">自动 CUDA 编译优化的 JIT 缓存</div>
              <div className="text-[11px] text-slate-400">将编译后的 Cutlass 内核持久化存储至磁盘</div>
            </div>
            <input type="checkbox" defaultChecked className="w-4 h-4 accent-violet-600 rounded" />
          </div>

          <div className="flex items-center justify-between py-2">
            <div>
              <div className="text-xs font-semibold text-slate-200">自动清理 Hessian 矩阵中间过程文件</div>
              <div className="text-[11px] text-slate-400">量化任务结束后自动释放磁盘临时 Temp 空间</div>
            </div>
            <input type="checkbox" defaultChecked className="w-4 h-4 accent-violet-600 rounded" />
          </div>
        </div>
      </div>

      <div className="flex justify-end">
        <button 
          onClick={() => showToast('全局配置更新成功！')}
          className="px-5 py-2.5 rounded-xl bg-violet-600 hover:bg-violet-500 text-white font-semibold text-xs shadow-lg shadow-violet-600/20"
        >
          保存全局偏好设置
        </button>
      </div>
    </div>
  );
}

function VRAMCalculatorModal({ onClose }) {
  const [modelSize, setModelSize] = useState(14); // 7, 14, 32, 70B
  const [quantBits, setQuantBits] = useState(4); // 16, 8, 4, 2
  const [contextLen, setContextLen] = useState(8192); // tokens
  const [batchSize, setBatchSize] = useState(1);

  // VRAM Formula: Model Weights VRAM + KV Cache VRAM + Overhead
  // Model Weights = (params * bits) / 8
  // KV Cache ~= 2 * layers * hidden * seq_len * batch * 2 bytes / 10^9
  const weightVRAM = (modelSize * quantBits) / 8;
  const kvCacheVRAM = (modelSize * 0.15 * (contextLen / 2048) * batchSize).toFixed(1);
  const totalEstimatedVRAM = (weightVRAM + parseFloat(kvCacheVRAM) + 1.5).toFixed(1);

  return (
    <div className="fixed inset-0 z-50 bg-slate-950/80 backdrop-blur-sm flex justify-end">
      <div className="w-full max-w-md bg-slate-900 border-l border-slate-800 p-6 flex flex-col justify-between shadow-2xl animate-slide-left">
        <div className="space-y-6">
          <div className="flex items-center justify-between border-b border-slate-800 pb-4">
            <div className="flex items-center gap-2">
              <Calculator className="w-5 h-5 text-violet-400" />
              <h2 className="font-bold text-sm text-slate-100">显存 (VRAM) 需求精准估算器</h2>
            </div>
            <button onClick={onClose} className="p-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-400">✕</button>
          </div>

          {/* Model Params Slider */}
          <div className="space-y-2">
            <div className="flex justify-between text-xs">
              <span className="text-slate-300">模型参数量 (Parameters)</span>
              <span className="font-mono text-violet-400 font-bold">{modelSize} B</span>
            </div>
            <div className="flex gap-2">
              {[7, 14, 32, 70].map((size) => (
                <button
                  key={size}
                  onClick={() => setModelSize(size)}
                  className={`flex-1 py-1.5 text-xs font-mono rounded-lg border ${
                    modelSize === size ? 'bg-violet-600 text-white border-violet-500' : 'bg-slate-950 text-slate-400 border-slate-800'
                  }`}
                >
                  {size}B
                </button>
              ))}
            </div>
          </div>

          {/* Quant Bits Slider */}
          <div className="space-y-2">
            <div className="flex justify-between text-xs">
              <span className="text-slate-300">量化位宽 (Precision)</span>
              <span className="font-mono text-cyan-400 font-bold">{quantBits}-bit</span>
            </div>
            <div className="flex gap-2">
              {[16, 8, 4, 2].map((b) => (
                <button
                  key={b}
                  onClick={() => setQuantBits(b)}
                  className={`flex-1 py-1.5 text-xs font-mono rounded-lg border ${
                    quantBits === b ? 'bg-cyan-600 text-white border-cyan-500' : 'bg-slate-950 text-slate-400 border-slate-800'
                  }`}
                >
                  {b === 16 ? 'FP16' : `${b}-bit`}
                </button>
              ))}
            </div>
          </div>

          {/* Context Window Length */}
          <div className="space-y-2">
            <div className="flex justify-between text-xs">
              <span className="text-slate-300">上下文长度 (Context Tokens)</span>
              <span className="font-mono text-slate-300">{contextLen} tokens</span>
            </div>
            <input
              type="range"
              min={2048}
              max={32768}
              step={2048}
              value={contextLen}
              onChange={(e) => setContextLen(Number(e.target.value))}
              className="w-full accent-violet-500 cursor-pointer"
            />
          </div>

          {/* Results Display */}
          <div className="p-4 rounded-xl bg-slate-950 border border-violet-500/30 space-y-3">
            <div className="text-xs text-slate-400 font-medium">预计 VRAM 显存占用分解</div>
            <div className="space-y-1.5 text-xs font-mono">
              <div className="flex justify-between text-slate-300">
                <span>模型权重 VRAM:</span>
                <span>~{weightVRAM.toFixed(1)} GB</span>
              </div>
              <div className="flex justify-between text-slate-300">
                <span>KV-Cache 显存:</span>
                <span>~{kvCacheVRAM} GB</span>
              </div>
              <div className="flex justify-between text-slate-300">
                <span>CUDA Runtime 驻留:</span>
                <span>~1.5 GB</span>
              </div>
              <div className="border-t border-slate-800 pt-2 flex justify-between text-sm font-bold text-emerald-400">
                <span>总计建议 GPU 显存:</span>
                <span>~{totalEstimatedVRAM} GB</span>
              </div>
            </div>
          </div>
        </div>

        <button
          onClick={onClose}
          className="w-full py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold"
        >
          关闭估算器
        </button>
      </div>
    </div>
  );
}