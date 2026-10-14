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
        """异常安全回归：poller 线程里任何一轮抛异常，必须判 failed + 广播 state，
        不能静默死掉让 job 永久卡 running（曾因此挂死测试/生产 SSE）。"""
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
