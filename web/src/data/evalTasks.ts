/**
 * 评测任务与预设常量——与 webui/cmds.py 的 EVAL_TASKS / EVAL_PRESETS 逐字一致。
 * 后端未暴露该词表（Plan 3 只允许 token 一处后端改动），前端本地维护；
 * 改动时需同步 webui/cmds.py。
 */

export const EVAL_TASKS = [
  'gpqa_diamond_cot_zeroshot', 'gpqa_diamond_cot_n_shot', 'gpqa_main_cot_zeroshot',
  'mmlu_pro', 'mmlu', 'arc_challenge', 'hellaswag', 'gsm8k_cot_zeroshot',
  'ifeval', 'humaneval_instruct', 'drop',
] as const

export const EVAL_PRESETS: Record<string, string[]> = {
  'AA 标准（GPQA + MMLU-Pro + IFBench）':
    ['gpqa_diamond_cot_zeroshot', 'mmlu_pro', 'ifeval'],
  '快速检查（GPQA 单项）': ['gpqa_diamond_cot_zeroshot'],
  '量化敏感扩展（+ 数学/知识/代码/指令）':
    ['gpqa_diamond_cot_zeroshot', 'mmlu_pro', 'ifeval',
     'gsm8k_cot_zeroshot', 'hellaswag', 'humaneval_instruct'],
}

/** AA 套件（HLE/SciCode/AA-LCR/Omniscience/τ²/Terminal-Bench）跑不了的说明，与 cmds.py 注释一致 */
export const EVAL_NOTE =
  'AA 套件（HLE/SciCode/AA-LCR/Omniscience/τ²/Terminal-Bench）需要判分端点 + 沙箱/长上下文，单卡裸机跑不动。'
