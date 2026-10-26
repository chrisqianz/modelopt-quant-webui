"""progress.parse 的诚实性单测（2026-10-26 用户：「进度条要么不走动，要么直接显示完成」）。"""
from webui import progress

BAR_100 = "Loading weights: 100%|██████████| 471/471 [00:08<00:00, 52.3it/s]\n"


def test_subbar_100_clamps_to_99_before_export():
    """早期权重条 100% ≠ 整体完成：非导出前总进度封顶 99。"""
    rep = progress.parse(BAR_100)
    assert rep.pct == 99.0
    assert not rep.done


def test_done_keeps_100():
    rep = progress.parse("whatever\nQuantized model exported to: /x/out\n")
    assert rep.done and rep.pct == 100.0


def test_min_stage_floor_prevents_regression():
    """marker 被冲出窗口后，min_stage 保证阶段不回退。"""
    with_marker = "Use calib batch_size 1\nCalibrating: 30%|x| 300/1024 [00:10<00:20]\n"
    assert progress.parse(with_marker).stage_idx == 3
    lost = "Calibrating: 70%|x| 700/1024 [00:30<00:10]\n"
    assert progress.parse(lost).stage_idx == 0          # 无地板：确实会倒退（复现 bug）
    rep = progress.parse(lost, min_stage=3)              # 有地板：钉住
    assert rep.stage_idx == 3 and rep.stage == progress.STAGES[3]
