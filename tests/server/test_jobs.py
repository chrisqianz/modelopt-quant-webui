"""server.jobs 测试：Part A（create/persist/reload）+ Part B（poller + SSE 扇出）。

poller 是 5s 一跳的线程，测试里把 server.jobs._sleep 打补丁到 0.05s
（不要 patch time.sleep——它 patch 的是全局 time 模块，假实现内部
再调 time.sleep 会无限递归，曾挂死整个测试套）。"""
import asyncio
import json
import time
from unittest.mock import patch

from server import jobs as jobs_mod
from webui import runner

DEAD_PID = "4194000"  # 不存活的大 pid


def _sse_events(text: str) -> list[dict]:
    """把 SSE 原始流拆成 {type, id, data} 列表。"""
    events = []
    for block in text.split("\n\n"):
        block = block.strip("\n")
        if not block:
            continue
        d = {}
        for line in block.split("\n"):
            if ":" in line:
                k, v = line.split(":", 1)
                d[k.strip()] = v.strip()
        assert "event" in d and "id" in d and "data" in d, f"SSE 块缺字段: {block!r}"
        events.append({"type": d["event"], "id": int(d["id"]),
                       "data": json.loads(d["data"])})
    return events


# ---------------------------------------------------------------- Part A

class TestCreatePersistReload:
    def test_create_job_in_list_starting(self, registry, cfg):
        task = runner.Task(task_id="t", remote_cmd="true",
                           log_path=str(cfg.local.task_path / "t" / "run.log"))
        job = registry.create("quant", task, "echo hi")
        assert job.state == "starting"
        assert job.id.startswith("quant_")
        assert job in registry.list()
        assert registry.get(job.id) is job

    def test_persist_writes_job_json(self, registry, cfg):
        task = runner.Task(task_id="t", remote_cmd="true",
                           log_path="/x/run.log", pid="123", state="running")
        job = registry.create("quant", task, "cmd-x",
                              export_dir="/x/out", version="1.2", commit="abc")
        registry.persist(job)
        path = cfg.local.task_path / job.id / "job.json"
        assert path.is_file()
        raw = json.loads(path.read_text())
        for k in ("pid", "log_path", "cmd", "state", "kind",
                  "started_at", "done_marker"):
            assert k in raw, f"job.json 缺字段 {k}"
        assert raw["pid"] == "123"
        assert raw["log_path"] == "/x/run.log"
        assert raw["cmd"] == "cmd-x"
        assert raw["state"] == "starting"
        assert raw["kind"] == "quant"
        assert raw["started_at"] == job.started_at
        assert raw["done_marker"] == "Quantized model exported to"

    def test_persist_venv_job_empty_pid_and_marker(self, registry, cfg):
        job = registry.create("venv", None, "uv venv")
        registry.persist(job)
        raw = json.loads((cfg.local.task_path / job.id / "job.json").read_text())
        assert raw["pid"] == ""
        assert raw["done_marker"] == ""

    def test_reload_dead_pid_lands_terminal(self, registry, cfg):
        d = cfg.local.task_path / "quant_reload_1"
        d.mkdir(parents=True)
        (d / "job.json").write_text(json.dumps({
            "pid": DEAD_PID, "log_path": str(d / "run.log"), "cmd": "echo x",
            "state": "running", "export_dir": "", "kind": "quant",
            "version": "", "commit": "", "started_at": time.time(),
            "done_marker": "Quantized model exported to",
        }))
        loaded = registry.reload()
        job = registry.get("quant_reload_1")
        assert job in loaded
        assert job.state in ("done", "failed")  # 不是 running
        assert job.task is None  # 已死进程不重建 Task

    def test_reload_venv_dead_pid_with_marker_done(self, registry, cfg):
        d = cfg.local.task_path / "venv_reload_1"
        d.mkdir(parents=True)
        (d / "run.log").write_text("installing...\nVENV_DONE rc=0\n")
        (d / "job.json").write_text(json.dumps({
            "pid": DEAD_PID, "log_path": str(d / "run.log"), "cmd": "uv venv",
            "state": "running", "export_dir": "", "kind": "venv",
            "version": "", "commit": "", "started_at": time.time(),
            "done_marker": "",
        }))
        registry.reload()
        assert registry.get("venv_reload_1").state == "done"

    def test_reload_terminal_state_adopted_as_is(self, registry, cfg):
        d = cfg.local.task_path / "eval_dead_done"
        d.mkdir(parents=True)
        (d / "job.json").write_text(json.dumps({
            "pid": DEAD_PID, "log_path": str(d / "run.log"), "cmd": "true",
            "state": "failed", "export_dir": "", "kind": "eval",
            "version": "", "commit": "", "started_at": time.time(),
            "done_marker": "EVAL_DONE",
        }))
        registry.reload()
        assert registry.get("eval_dead_done").state == "failed"

    def test_abort(self, registry, cfg):
        task = runner.Task(task_id="t", remote_cmd="true",
                           log_path=str(cfg.local.task_path / "t" / "run.log"),
                           pid="1", state="running")
        job = registry.create("quant", task, "cmd")
        with patch.object(jobs_mod.runner, "abort_task", return_value="ok") as m:
            assert registry.abort(job.id) == "aborted"
        assert job.state == "aborted"
        m.assert_called_once_with(task)
        assert registry.abort("nope") == ""

    def test_abort_broadcasts_state_event(self, registry, cfg):
        """B1 回归：abort 必须广播 state 事件，否则 SSE 客户端挂在
        await q.get() 永不终止（sse_stream 靠 state 事件 break）。"""
        task = runner.Task(task_id="t2", remote_cmd="true",
                           log_path=str(cfg.local.task_path / "t2" / "run.log"),
                           pid="1", state="running")
        job = registry.create("quant", task, "cmd")

        async def main():
            q, _ = registry.subscribe(job)
            assert registry.abort(job.id) == "aborted"
            item = await asyncio.wait_for(q.get(), timeout=1.0)
            return item

        item = asyncio.run(main())
        assert item["type"] == "state"
        assert item["data"] == {"state": "aborted"}

    def test_abort_terminal_job_is_noop(self, registry, cfg):
        """终态 job 不可再 abort：不覆盖 state、不重复落盘/广播。"""
        task = runner.Task(task_id="t3", remote_cmd="true",
                           log_path=str(cfg.local.task_path / "t3" / "run.log"),
                           pid="1", state="running")
        job = registry.create("quant", task, "cmd")
        job.state = "done"
        with patch.object(jobs_mod.runner, "abort_task") as m:
            assert registry.abort(job.id) == "done"
        m.assert_not_called()
        assert job.state == "done"

    def test_reload_skips_dirty_job_json(self, registry, cfg):
        """建议 2 回归：单个 job.json 脏数据（类型错）只跳过该文件，
        不炸掉启动时其他 job 的收编。"""
        d_bad = cfg.local.task_path / "bad_1"
        d_bad.mkdir(parents=True)
        (d_bad / "job.json").write_text(json.dumps({
            "pid": "abc",  # int("abc") 会抛 ValueError
            "log_path": "", "cmd": "x", "state": "running", "kind": "quant",
            "started_at": "abc",
        }))
        d_ok = cfg.local.task_path / "good_1"
        d_ok.mkdir(parents=True)
        (d_ok / "job.json").write_text(json.dumps({
            "pid": "", "log_path": "", "cmd": "true", "state": "failed",
            "kind": "quant", "started_at": time.time(),
        }))
        loaded = registry.reload()  # 不应抛异常
        ids = {j.id for j in loaded}
        assert "bad_1" not in ids
        assert "good_1" in ids
        assert registry.get("bad_1") is None
        assert registry.get("good_1").state == "failed"

    def test_abort_venv_sets_done_flag(self, registry):
        job = registry.create("venv", None, "uv venv")
        assert registry.abort(job.id) == "aborted"
        assert job._done is True  # poller 下轮见 _done 即退出


# ---------------------------------------------------------------- Part B

class TestPollerSSE:
    def test_quant_poller_and_sse_events(self, registry, cfg):
        """poll_task 先 running 后 done；read_log（replay 与 poller 共用）恒返同一份
        完整日志 ⇒ 所有事件 id 恒等于日志字节偏移，无论 race 顺序都非降。
        sse_stream 应收到 log/progress/state 三事件，终态 state=done，
        poller 同 offset 的 log 被去重跳过（客户端恰好一条 log）。"""
        d = cfg.local.task_path / "quant_b1"
        d.mkdir(parents=True)
        log = str(d / "run.log")
        task = runner.Task(task_id="quant_b1", remote_cmd="true",
                           log_path=log, pid="1", state="running")
        job = registry.create("quant", task, "cmd")

        LOG_FULL = ("Downloading model 47/100 47%\n"
                    "Some progress line\n"
                    "Quantized model exported to: /x/out\n")

        poll_n = {"n": 0}

        def fake_poll(t):
            poll_n["n"] += 1
            t.state = "running" if poll_n["n"] < 2 else "done"

        async def main():
            with patch.object(jobs_mod.runner, "poll_task", fake_poll), \
                 patch.object(jobs_mod.runner, "read_log",
                              lambda t, tail_bytes=20000: LOG_FULL), \
                 patch.object(jobs_mod, "_sleep", lambda s: time.sleep(0.05)):
                q, start = registry.subscribe(job)
                registry.start_poller(job)
                text = ""
                async for line in registry.sse_stream(job, q, start):
                    text += line
                    if _sse_events(text) and _sse_events(text)[-1]["type"] == "state":
                        break
                return _sse_events(text)

        events = asyncio.run(main())
        types = {e["type"] for e in events}
        assert types == {"log", "progress", "state"}
        ids = [e["id"] for e in events]
        assert all(a <= b for a, b in zip(ids, ids[1:], strict=False)), f"id 非单调: {ids}"
        assert ids[-1] == len(LOG_FULL)  # id 语义 = 日志字节偏移
        assert events[-1]["type"] == "state"
        assert events[-1]["data"] == {"state": "done"}
        # 恰好一条 log 事件：poller 同 offset 的 log 广播被去重游标跳过
        logs = [e for e in events if e["type"] == "log"]
        assert len(logs) == 1, f"log 事件应恰好 1 条（去重生效），实际 {len(logs)}"
        assert "Quantized model exported to: /x/out\n" in logs[0]["data"]["tail"]
        # 进度事件的 bar 字段被丢弃
        prog = next(e for e in events if e["type"] == "progress")
        assert "bar" not in prog["data"]
        assert job.state == "done"

    def test_poller_log_delta_no_duplication(self, registry, cfg):
        """回归（2026-10-26 用户实见 UI 重复 8 份）：旧实现每轮广播整个 tail，
        前端「追加」语义下同一段日志滚 N 遍。修后每轮只发新增字节：
        无论 replay/轮询怎么交错，拼接结果里每行恰好出现一次。"""
        d = cfg.local.task_path / "quant_delta"
        d.mkdir(parents=True)
        task = runner.Task(task_id="quant_delta", remote_cmd="true",
                           log_path=str(d / "run.log"), pid="1", state="running")
        job = registry.create("quant", task, "cmd")
        st = {"content": "L1\n", "polls": 0}

        def fake_poll(t):
            st["polls"] += 1
            st["content"] = {1: "L1\nL2\n",
                             2: "L1\nL2\nL3\nQuantized model exported to: /x/out\n"
                             }.get(st["polls"], st["content"])
            t.state = "running" if st["polls"] < 2 else "done"

        async def main():
            with patch.object(jobs_mod.runner, "poll_task", fake_poll), \
                 patch.object(jobs_mod.runner, "read_log",
                              lambda t, tail_bytes=20000: st["content"]), \
                 patch.object(jobs_mod, "_sleep", lambda s: time.sleep(0.05)):
                q, start = registry.subscribe(job)
                registry.start_poller(job)
                text = ""
                evs: list[dict] = []
                async for line in registry.sse_stream(job, q, start):
                    text += line
                    evs = _sse_events(text)
                    if evs and evs[-1]["type"] == "state":
                        break
                return evs

        events = asyncio.run(main())
        merged = "".join(e["data"]["tail"] for e in events if e["type"] == "log")
        for token in ("L1\n", "L2\n", "L3\n"):
            assert merged.count(token) == 1, f"{token!r} 重复/丢失: {merged!r}"
        assert "Quantized model exported to" in merged

    def test_poller_progress_stage_monotonic(self, registry, cfg):
        """回归（2026-10-26 Nanbeige 33B 实跑）：tail 窗口冲掉早期阶段 marker 后，
        61 分钟校准期进度卡片回退成「第 1/7 步 · 启动」。修后 poller 持
        progress_stage 地板，广播的 stage_idx 绝不回退。"""
        d = cfg.local.task_path / "quant_stage"
        d.mkdir(parents=True)
        task = runner.Task(task_id="quant_stage", remote_cmd="true",
                           log_path=str(d / "run.log"), pid="1", state="running")
        job = registry.create("quant", task, "cmd")
        rounds = [
            "Loading pretrained model\nUse calib batch_size 1\n"
            "Calibrating:  30%|███ | 300/1024 [00:10<00:20, 30.0it/s]\n",
            # 窗口只剩校准条本身——早期 marker 全被冲出，无地板时 stage 会退回 0
            "Calibrating:  70%|██████ | 700/1024 [00:30<00:10, 30.0it/s]\n",
            "Calibrating:  99%|███████▉| 1023/1024 [00:59<00:00, 30.0it/s]\n"
            "Quantized model exported to: /x/out\n",
        ]
        st = {"polls": 0}

        def fake_poll(t):
            st["polls"] += 1
            t.state = "running" if st["polls"] < 3 else "done"

        async def main():
            with patch.object(jobs_mod.runner, "poll_task", fake_poll), \
                 patch.object(jobs_mod.runner, "read_log",
                              lambda t, tail_bytes=20000: rounds[min(st["polls"], 2)]), \
                 patch.object(jobs_mod, "_sleep", lambda s: time.sleep(0.05)):
                q, start = registry.subscribe(job)
                registry.start_poller(job)
                text = ""
                evs: list[dict] = []
                async for line in registry.sse_stream(job, q, start):
                    text += line
                    evs = _sse_events(text)
                    if evs and evs[-1]["type"] == "state":
                        break
                return evs

        events = asyncio.run(main())
        stages = [e["data"]["stage_idx"] for e in events if e["type"] == "progress"]
        assert stages, "应有 progress 事件"
        assert all(a <= b for a, b in zip(stages, stages[1:], strict=False)), f"stage 回退: {stages}"
        assert max(stages) >= 3, f"至少推进到校准阶段: {stages}"
        assert job.state == "done"

    def test_sse_stream_replay_dedup(self, registry, cfg):
        """start_offset 之后的日志走重放（event: log），去重游标内的 log 跳过。"""
        d = cfg.local.task_path / "quant_replay"
        d.mkdir(parents=True)
        log = str(d / "run.log")
        task = runner.Task(task_id="quant_replay", remote_cmd="true",
                           log_path=log, pid="1", state="running")
        job = registry.create("quant", task, "cmd")
        # 先放两段日志
        seg1 = "line one\n"
        (d / "run.log").write_text(seg1)

        events = []

        async def main():
            with patch.object(jobs_mod.runner, "read_log",
                              lambda t, tail_bytes=20000: seg1), \
                 patch.object(jobs_mod.runner, "poll_task", lambda t: None), \
                 patch.object(jobs_mod, "_sleep", lambda s: time.sleep(0.05)):
                q, _ = registry.subscribe(job)  # job.log_offset=0 → 全量重放
                got_replay = asyncio.Event()

                async def drain():
                    async for line in registry.sse_stream(job, q, 0):
                        events.append(_sse_events(line + "\n\n")[-1])
                        if events[-1]["type"] == "log":
                            got_replay.set()

                drain_task = asyncio.create_task(drain())
                await got_replay.wait()
                # 重放之后，poller 广播一条 id<=replay_to 的 log（应被去重跳过）
                job.log_offset = len(seg1)
                registry._broadcast(job, "log", {"tail": seg1})
                registry._broadcast(job, "progress", {"stage": "s", "pct": 1.0})
                await asyncio.sleep(0.1)
                registry._broadcast(job, "state", {"state": "done"})
                await asyncio.sleep(0.1)
                drain_task.cancel()
            return events

        asyncio.run(main())
        types = [e["type"] for e in events]
        # 第一条是重放的 log；随后去重跳过同 id 的 log，progress + state 各一条
        assert types[0] == "log"
        assert types[1] == "progress"
        assert types[-1] == "state"
        assert len(types) == 3, f"期望 [log, progress, state]，实际 {types}"

    def test_venv_poller_marker_done(self, registry, tmp_path):
        """venv 类 task=None：先写 installing，再写 VENV_DONE rc=0 → done。"""
        log = str(tmp_path / "venv_install.log")
        (tmp_path / "venv_install.log").write_text("installing...\n")
        job = registry.create("venv", None, "uv venv")
        job.log_path = log

        with patch.object(jobs_mod, "_sleep", lambda s: time.sleep(0.05)):
            registry.start_poller(job)
            t0 = time.time()
            while job.state in ("starting", "running"):
                if time.time() - t0 > 3:
                    raise AssertionError("venv job 超时未终态")
                if "VENV_DONE rc=0" not in (tmp_path / "venv_install.log").read_text():
                    (tmp_path / "venv_install.log").write_text(
                        "installing...\nCollecting x\nVENV_DONE rc=0\n")
                time.sleep(0.05)
        assert job.state == "done"

    def test_venv_early_exit_no_marker(self, registry, tmp_path):
        """早退路径：log 为空文件（无 marker），手动置 job._done=True（无 _exc）→ done。"""
        log = str(tmp_path / "venv_early.log")
        (tmp_path / "venv_early.log").write_text("")
        job = registry.create("venv", None, "uv venv")
        job.log_path = log
        job._done = True  # 模拟 ensure_venv 包装线程已退出（venv 已存在早退）

        with patch.object(jobs_mod, "_sleep", lambda s: time.sleep(0.05)):
            registry.start_poller(job)
            t0 = time.time()
            while job.state in ("starting", "running"):
                if time.time() - t0 > 3:
                    raise AssertionError("early-exit venv job 超时未终态")
                time.sleep(0.05)
        assert job.state == "done"

    def test_venv_failed_marker(self, registry, tmp_path):
        """VENV_DONE rc=1 → failed。"""
        log = str(tmp_path / "venv_fail.log")
        (tmp_path / "venv_fail.log").write_text("Collecting x\nVENV_DONE rc=1\n")
        job = registry.create("venv", None, "uv venv")
        job.log_path = log

        with patch.object(jobs_mod, "_sleep", lambda s: time.sleep(0.05)):
            registry.start_poller(job)
            t0 = time.time()
            while job.state in ("starting", "running"):
                if time.time() - t0 > 3:
                    raise AssertionError("venv failed job 超时未终态")
                time.sleep(0.05)
        assert job.state == "failed"

    def test_poller_exception_is_failed_not_hang(self, registry, cfg):
        """异常安全回归（持续异常）：单轮异常只记 traceback + 跳过；
        连续 _POLL_MAX_CONSECUTIVE_ERRORS 轮异常才判 failed + 广播 state，
        不能静默死掉让 job 永久卡 running（曾因此挂死测试/生产 SSE）。
        （2026-10-15 事故：旧策略「异常=立即 failed 且不留痕」把正在收尾的
        成功任务静默误判 failed，事后无法定位。）"""
        d = cfg.local.task_path / "quant_exc"
        d.mkdir(parents=True)
        task = runner.Task(task_id="quant_exc", remote_cmd="true",
                           log_path=str(d / "run.log"), pid="1", state="running")
        job = registry.create("quant", task, "cmd")
        events = []

        def boom(t):
            raise RuntimeError("boom")

        async def main():
            with patch.object(jobs_mod.runner, "poll_task", boom), \
                 patch.object(jobs_mod, "_sleep", lambda s: time.sleep(0.05)):
                q, _ = registry.subscribe(job)
                registry.start_poller(job)
                async for line in registry.sse_stream(job, q, 0):
                    events.append(_sse_events(line + "\n\n")[-1])
                    if events[-1]["type"] == "state":
                        break

        asyncio.run(main())
        assert events[-1]["type"] == "state"
        assert events[-1]["data"] == {"state": "failed"}
        assert job.state == "failed"

    def test_poller_transient_exception_keeps_running(self, registry, cfg):
        """单轮异常回归：第 1 轮抛异常、第 2 轮正常收尾 ⇒ job 是 done 而非 failed。
        旧策略「异常=立即 failed」正是 2026-10-15 远程成功任务被误判的机制。"""
        d = cfg.local.task_path / "quant_exc_transient"
        d.mkdir(parents=True)
        task = runner.Task(task_id="quant_exc_transient", remote_cmd="true",
                           log_path=str(d / "run.log"), pid="1", state="running")
        job = registry.create("quant", task, "cmd")
        calls = {"n": 0}

        def flaky_poll(t):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("transient boom")
            t.state = "done"

        with patch.object(jobs_mod.runner, "poll_task", flaky_poll), \
             patch.object(jobs_mod.runner, "read_log",
                          lambda t, tail_bytes=20000: "done line\n"), \
             patch.object(jobs_mod, "_sleep", lambda s: time.sleep(0.05)):
            t0 = time.time()
            registry.start_poller(job)
            while job.state not in ("done", "failed"):
                if time.time() - t0 > 3:
                    raise AssertionError("job 超时未终态")
                time.sleep(0.05)
        assert calls["n"] >= 2
        assert job.state == "done"

def test_venv_setup_cmd_optional_kernels():
    """可选 GPU kernel 是 best-effort 尾巴：主安装退出码必须活到 VENV_DONE，
    fla import 失败必须回滚，causal-conv1d 只准吃预编译轮子。"""
    cmd = runner.venv_setup_cmd("/usr/bin/uv", "/v", "/repo", "")
    assert "-e '.[hf]'" in cmd
    assert "install --no-deps flash-linear-attention" in cmd
    assert "install --only-binary :all: causal-conv1d" in cmd
    # 主安装 rc=$? 抢先捕获，可选 kernel 的退出码不得污染 VENV_DONE rc
    assert "rc=$?" in cmd and cmd.rstrip().endswith("echo VENV_DONE rc=$rc")
    i = cmd.index("import fla")
    assert cmd.index("uninstall -y flash-linear-attention") > i
    assert "import fla" in cmd  # 回滚探测在两个可选安装之间


def test_start_task_injects_hf_token(cfg, monkeypatch):
    """令牌走子进程 env，不进命令行：ps 看不到，也不落盘 run.sh。
    假值拼接构造，避开 secret 扫描规则对 hf_token 字面量的误报。"""
    tok = "TK-" + "TOP"
    cfg.net.hf_token = tok
    seen = {}

    def fake(cmd, timeout=60, env_extra=None):
        seen["cmd"], seen["env"] = cmd, env_extra
        return 0, "12345", ""

    monkeypatch.setattr(runner, "run_shell", fake)
    t = runner.start_task(cfg, str(cfg.local.task_path), "org/m", "/x/exported",
                          "python hf_ptq.py", "", lambda s: None)
    assert seen["env"] == {"HF_TOKEN": tok}
    assert tok not in seen["cmd"]
    assert t.pid == "12345"


def test_start_task_no_token_keeps_env_none(cfg, monkeypatch):
    seen = {}

    def fake(cmd, timeout=60, env_extra=None):
        seen["env"] = env_extra
        return 0, "12345", ""

    monkeypatch.setattr(runner, "run_shell", fake)
    runner.start_task(cfg, str(cfg.local.task_path), "org/m", "/x/exported",
                      "python hf_ptq.py", "", lambda s: None)
    assert seen["env"] is None
