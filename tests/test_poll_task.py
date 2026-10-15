"""webui.runner.poll_task 单测（2026-10-15 远程误判 failed 事故回归）。

事故复盘：旧代码 `if r != 0` 把 run_shell 的 rc=-1（探测命令本身超时）
当作「进程已死」，进程还在跑的任务被提前判 failed；进程刚死而成功 marker
尚未落盘时也是立即判 failed，漏掉「正在收尾的成功任务」。
回归点：
1. kill 探测超时（rc=-1）→ 状态不变，不当已死
2. tail 读超时（rc=-1）→ 不判 done（空串不是「日志为空」）
3. 进程死 + marker 已落盘 → done
4. 进程死 + 无 marker → 宽限 N 轮后才判 failed，期间保持 running
5. 宽限中进程恢复（pid 复用）→ 宽限计数清零
"""
from webui import runner


def _task(tmp_path, **kw) -> runner.Task:
    base: dict = {"task_id": "t", "remote_cmd": "x",
                  "log_path": str(tmp_path / "run.log"), "pid": "12345"}
    base.update(kw)
    return runner.Task(**base)


class _FakeShell:
    """按序返回预置 (rc, out)；每次 poll_task 调用最多消耗两条
    （kill 探测 + tail 读日志）。队列耗尽后默认「活着」。"""

    def __init__(self, results: list[tuple[int, str]]):
        self.results = list(results)

    def __call__(self, cmd: str, timeout=60, env=None):
        if self.results:
            rc, out = self.results.pop(0)
        else:
            rc, out = 0, "alive"
        return rc, out, ""


def _patch(monkeypatch, results):
    monkeypatch.setattr(runner, "run_shell", _FakeShell(results))


class TestProbePhase:
    def test_kill_timeout_keeps_state(self, monkeypatch, tmp_path):
        t = _task(tmp_path, state="running")
        _patch(monkeypatch, [(-1, "")])
        runner.poll_task(t)
        assert t.state == "running"

    def test_kill_timeout_from_starting(self, monkeypatch, tmp_path):
        t = _task(tmp_path, state="starting")
        _patch(monkeypatch, [(-1, "")])
        runner.poll_task(t)
        assert t.state == "starting"

    def test_alive_sets_running_and_resets_grace(self, monkeypatch, tmp_path):
        t = _task(tmp_path, state="running", death_grace=4)
        _patch(monkeypatch, [(0, "alive")])
        runner.poll_task(t)
        assert t.state == "running"
        assert t.death_grace == 0


class TestDeadPhase:
    def test_dead_with_marker_done(self, monkeypatch, tmp_path):
        t = _task(tmp_path)
        _patch(monkeypatch, [(1, ""), (0, "Quantized model exported to: /x")])
        runner.poll_task(t)
        assert t.state == "done"

    def test_tail_timeout_does_not_judge(self, monkeypatch, tmp_path):
        """tail rc=-1 是读超时，不是日志为空——旧代码会把空串判成 done。"""
        t = _task(tmp_path, state="running")
        _patch(monkeypatch, [(1, ""), (-1, "")])
        runner.poll_task(t)
        assert t.state == "running"

    def test_dead_empty_tail_done_legacy(self, monkeypatch, tmp_path):
        t = _task(tmp_path)
        _patch(monkeypatch, [(1, ""), (0, "  \n")])
        runner.poll_task(t)
        assert t.state == "done"

    def test_dead_no_marker_grace_then_failed(self, monkeypatch, tmp_path):
        t = _task(tmp_path, state="running")
        _patch(monkeypatch, [(1, ""), (0, "no marker here")])
        # 第 1 轮：进入宽限
        runner.poll_task(t)
        assert t.state == "running"
        assert t.death_grace == runner._DEATH_GRACE_ROUNDS
        # 中间轮：递减但不判 failed
        for _ in range(runner._DEATH_GRACE_ROUNDS - 1):
            _patch(monkeypatch, [(1, ""), (0, "still no marker")])
            runner.poll_task(t)
        assert t.state == "running"
        assert t.death_grace == 1
        # 最后一轮：宽限耗尽 → failed
        _patch(monkeypatch, [(1, ""), (0, "still no marker")])
        runner.poll_task(t)
        assert t.state == "failed"

    def test_grace_resets_when_alive_again(self, monkeypatch, tmp_path):
        t = _task(tmp_path, state="running")
        _patch(monkeypatch, [(1, ""), (0, "no marker")])
        runner.poll_task(t)
        assert t.death_grace == runner._DEATH_GRACE_ROUNDS
        _patch(monkeypatch, [(0, "alive")])
        runner.poll_task(t)
        assert t.state == "running"
        assert t.death_grace == 0
