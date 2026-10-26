"""后台任务 registry + 5s 轮询 poller + SSE 扇出 + 落盘/启动重挂。

量化/评测任务由 webui.runner 起 nohup 进程（有 ``runner.Task``）；
venv/eval_env 安装类任务没有 Task（``task=None``），靠 marker-tail
（``VENV_DONE rc=`` / ``EVAL_ENV_DONE rc=``）+ 包装线程三条件判终态。

线程模型：poller 每 5s 一跳，读日志尾部、解析进度、状态变化落盘
``task_path/<job_id>/job.json``，并把 log/progress/state 三事件
``call_soon_threadsafe`` 扇出到各订阅者 asyncio.Queue；SSE 端先按
Last-Event-ID 重放已读游标之后的日志（``log_offset`` 为去重游标），
再实时消费队列。
"""

from __future__ import annotations

import asyncio
import dataclasses
import glob
import json
import os
import re
import threading
import time
import traceback
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from webui import progress, runner


_TERMINAL = ("done", "failed", "aborted")

# poller 连续多少轮异常后放弃重试、判 job failed（单轮异常只记日志跳过，
# 避免瞬时故障把正在收尾的任务误判成失败）
_POLL_MAX_CONSECUTIVE_ERRORS = 5


def _sleep(seconds: float) -> None:
    """轮询间隔：模块级函数，测试 patch server.jobs._sleep 即可，
    不要 patch time.sleep（它 patch 的是全局 time 模块，测试的假实现
    内部再调 time.sleep 会无限递归）。"""
    time.sleep(seconds)


@dataclass
class Job:
    id: str
    kind: str                 # quant / eval / venv / eval_env
    task: runner.Task | None   # venv/eval_env 类没有 Task，是 None
    state: str                # starting / running / done / failed / aborted
    log_path: str
    cmd: str
    export_dir: str = ""
    version: str = ""
    commit: str = ""
    started_at: float = field(default_factory=time.time)
    log_offset: int = 0       # SSE 续传游标 = 日志已读字节数
    _subscribers: list = field(default_factory=list, repr=False)  # [(queue, loop), ...]
    _done: bool = field(default=False, repr=False)    # venv/eval_env 包装线程已退出
    _exc: BaseException | None = field(default=None, repr=False)


class JobRegistry:
    def __init__(self, cfg):
        self.cfg = cfg
        self._jobs: dict[str, Job] = {}

    # ---------------------------------------------------------------- create/list

    def create(self, kind, task, cmd, export_dir="", version="", commit="") -> Job:
        job = Job(
            id=kind + "_" + time.strftime("%Y%m%d-%H%M%S") + "_" + uuid.uuid4().hex[:6],
            kind=kind,
            task=task,
            state="starting",
            log_path=task.log_path if task is not None else "",
            cmd=cmd,
            export_dir=export_dir,
            version=version,
            commit=commit,
        )
        self._jobs[job.id] = job
        return job

    def get(self, job_id) -> Job | None:
        return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        """活跃(starting/running)在前、历史在后，各自 started_at 倒序。"""
        # list() 快照：其他线程（API 请求）并发 create() 时不触发
        # dictionary changed size during iteration
        active = [j for j in list(self._jobs.values()) if j.state in ("starting", "running")]
        done = [j for j in self._jobs.values() if j.state not in ("starting", "running")]
        active.sort(key=lambda j: j.started_at, reverse=True)
        done.sort(key=lambda j: j.started_at, reverse=True)
        return active + done

    # ---------------------------------------------------------------- abort

    def abort(self, job_id) -> str:
        job = self.get(job_id)
        if job is None:
            return ""
        if job.state in _TERMINAL:
            # 终态不可再 abort：不覆盖、不重复落盘/广播
            return job.state
        if job.task is not None:
            runner.abort_task(job.task)
        job.state = "aborted"
        if job.task is None:
            # 包装线程（venv 类）没有 pid 可杀：置 _done，poller 下轮
            # 见 _done 即退出，不再写 done/failed。
            job._done = True
        self.persist(job)
        # abort 后 poller 循环条件（state in _TERMINAL）直接退出不补发，
        # 这里必须广播 state，否则 SSE 客户端挂在 await q.get() 永不终止。
        self._broadcast(job, "state", {"state": job.state})
        return job.state

    # ---------------------------------------------------------------- 落盘

    def persist(self, job) -> None:
        d = self.cfg.local.task_path / job.id
        d.mkdir(parents=True, exist_ok=True)
        task = job.task
        payload = {
            "id": job.id,
            "pid": (task.pid if task is not None and task.pid else ""),
            "log_path": job.log_path,
            "cmd": job.cmd,
            "state": job.state,
            "export_dir": job.export_dir,
            "kind": job.kind,
            "version": job.version,
            "commit": job.commit,
            "started_at": job.started_at,
            "done_marker": (task.done_marker if task is not None and task.done_marker else ""),
        }
        (d / "job.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    # ---------------------------------------------------------------- 启动重挂

    def reload(self) -> list[Job]:
        """扫 task_path/*/job.json：终态直接收编；pid 存活重挂 poller；
        进程已死且未终态的，读日志尾判终态（venv 看 marker，quant/eval 保守 failed）。"""
        loaded = []
        pattern = str(self.cfg.local.task_path / "*" / "job.json")
        for path in sorted(glob.glob(pattern)):
            try:
                with open(path) as f:
                    raw = json.load(f)
                job = Job(
                    id=raw.get("id") or os.path.basename(os.path.dirname(path)),
                    kind=raw.get("kind", ""),
                    task=None,
                    state=raw.get("state", "failed"),
                    log_path=raw.get("log_path", ""),
                    cmd=raw.get("cmd", ""),
                    export_dir=raw.get("export_dir", ""),
                    version=raw.get("version", ""),
                    commit=raw.get("commit", ""),
                    started_at=float(raw.get("started_at") or time.time()),
                )
                state = job.state
                if state in ("done", "failed", "aborted"):
                    # 1) 已终态：原样收编，不挂 poller。
                    self._jobs[job.id] = job
                    loaded.append(job)
                    continue
                pid = str(raw.get("pid") or "").strip()
                if pid and _pid_alive(int(pid)):
                    # 2) 进程还活着：重建 Task，重新接管轮询。
                    # done_marker 回落 Task 默认值（persist 总会写该键，
                    # 只有手工/旧版文件会缺）——空串会让 poll_task 的
                    # `marker in tail` 恒真，死进程被误判 done。
                    job.task = runner.Task(
                        task_id=job.id,
                        remote_cmd=job.cmd,
                        log_path=job.log_path,
                        pid=pid,
                        state="running",
                        started_at=job.started_at,
                        done_marker=raw.get("done_marker")
                        or runner.Task.__dataclass_fields__["done_marker"].default,
                        pkill_pattern="",
                    )
                    job.state = "running"
                    self._jobs[job.id] = job
                    self.start_poller(job)
                    loaded.append(job)
                    continue
                # 3) pid 空或已死（state 仍是 starting/running）：读日志尾判终态。
                if not job.log_path:
                    job.state = "failed"
                elif job.kind == "venv":
                    tail = runner.tail_log(job.log_path, 4000)
                    job.state = "done" if "VENV_DONE rc=0" in tail else "failed"
                elif job.kind == "eval_env":
                    tail = runner.tail_log(job.log_path, 4000)
                    job.state = "done" if "EVAL_ENV_DONE rc=0" in tail else "failed"
                else:
                    # quant/eval：进程死了但没写 done marker，保守判 failed。
                    job.state = "failed"
                self._jobs[job.id] = job
                loaded.append(job)
            except (OSError, ValueError, TypeError):
                # 单文件脏数据（缺字段/类型错/半截 json）只跳过该文件，
                # 不炸掉启动时全部 job 的重挂。
                continue
        return loaded

    # ---------------------------------------------------------------- poller

    def start_poller(self, job) -> None:
        threading.Thread(target=self._poll, args=(job,), daemon=True,
                         name=f"job-{job.id}").start()

    def _poll(self, job) -> None:
        # 异常安全：poller 是后台线程，任何一轮抛异常（日志被删、
        # 假实现递归等）都不能让线程静默死掉——否则 job 永久卡 running，
        # SSE 客户端永远等不到 state 事件。
        # 单轮异常只记录 + 跳过（日志可能正在写入、磁盘可能繁忙）；
        # 连续 _POLL_MAX_CONSECUTIVE_ERRORS 轮异常才判 failed，且必留 traceback
        # （2026-10-15 远程事故：旧策略「异常=立即 failed 且不留痕」导致成功
        # 任务被静默误判，事后无法定位根因）。
        consecutive_errors = 0
        while job.state not in _TERMINAL:
            terminal = False
            try:
                if job.kind in ("quant", "eval"):
                    runner.poll_task(job.task)
                    # 读全量（至 1MB）：前端 log 事件是「追加」语义，必须发绝对
                    # 偏移才能算增量；只读 tail 时 len(raw) 不是文件位置。
                    raw = runner.read_log(job.task, 2 ** 20)
                    rep = progress.parse(raw)
                    terminal = job.task.state != "running"
                    # 广播：只发新增字节。旧实现每轮广播整个 tail，前端累加后
                    # UI 滚出 N 份重复日志（2026-10-26 用户实见 8 份）。
                    if len(raw) > job.log_offset:
                        delta = raw[job.log_offset:]
                        job.log_offset = len(raw)
                        self._broadcast(job, "log", {"tail": delta})
                    self._broadcast(job, "progress", dataclasses.asdict(rep))
                else:  # venv / eval_env
                    # log_path 为空时先 glob 定位真实日志（ensure_venv 内部自己 mkstemp）
                    if not job.log_path:
                        prefix = "/tmp/modelopt_venv_setup_" if job.kind == "venv" else "/tmp/modelopt_eval_env_"  # noqa: S108
                        cands = [p for p in glob.glob(prefix + "*.log")
                                 if p.endswith(".log") and os.path.getmtime(p) >= job.started_at]
                        if cands:
                            job.log_path = max(cands, key=os.path.getmtime)
                            self.persist(job)
                    if not job.log_path and not job._done:
                        _sleep(5)
                        continue
                    # _done 且无日志（venv 已存在早退不写 /tmp 日志）⇒ raw 当空串，
                    # 直接落到下方 elif job._done 判终态，不再等 glob。
                    raw = runner.tail_log(job.log_path, 2 ** 20) if job.log_path else ""
                    rep = progress.parse_install(raw.splitlines())
                    # 终态判定三选一：(a) 日志 marker (b) 包装线程退出无异常 (c) 有异常
                    m = re.search(r"(VENV_DONE|EVAL_ENV_DONE) rc=(\d+)", raw)
                    if m:
                        terminal = True
                        job.state = "done" if m.group(2) == "0" else "failed"
                    elif job._done:      # 包装线程已退出（venv 已存在早退不写 marker 的情况）
                        terminal = True
                        job.state = "failed" if job._exc is not None else "done"
                    # 广播：只发新增字节（同 quant 分支，前端是累加语义）
                    if len(raw) > job.log_offset:
                        delta = raw[job.log_offset:]
                        job.log_offset = len(raw)
                        self._broadcast(job, "log", {"tail": delta})
                    self._broadcast(job, "progress", dataclasses.asdict(rep))
            except Exception:
                consecutive_errors += 1
                traceback.print_exc()  # 写 app stderr 日志，不再静默
                if consecutive_errors >= _POLL_MAX_CONSECUTIVE_ERRORS:
                    terminal = True
                    job.state = "failed"
            else:
                consecutive_errors = 0
            if terminal:
                if job.kind in ("quant", "eval") and job.state not in _TERMINAL:
                    job.state = "done" if job.task.state == "done" else "failed"
                self.persist(job)
                self._broadcast(job, "state", {"state": job.state})
                break
            _sleep(5)

    # ---------------------------------------------------------------- SSE 扇出

    def subscribe(self, job, last_event_id="") -> tuple[asyncio.Queue, int]:
        q = asyncio.Queue()
        loop = asyncio.get_running_loop()
        job._subscribers.append((q, loop))
        start = int(last_event_id) if last_event_id.isdigit() else job.log_offset
        return q, start

    async def sse_stream(self, job, q, start_offset) -> AsyncIterator[str]:
        # 1) 重放：先读当前日志全量，吐出 start_offset 之后的增量。
        if job.task is not None:
            raw = runner.read_log(job.task, 2 ** 20)
        else:
            raw = runner.tail_log(job.log_path, 2 ** 20) if job.log_path else ""
        if len(raw) > start_offset:
            yield _sse_line("log", len(raw), {"tail": raw[start_offset:]})
            replay_to = len(raw)
        else:
            replay_to = start_offset
        # 2) 实时：消费队列，按去重游标跳过已重放过的 log 事件。
        # 退出（正常 break / 客户端断开 GeneratorExit）必须注销订阅者：
        # 不注销则队列无消费者无限增长，且 loop 关闭后 poller 的
        # call_soon_threadsafe 抛 RuntimeError 被 _poll 的 except 捕获
        # → job 被误判 failed。
        try:
            while True:
                item = await q.get()
                if item["type"] == "log" and item["id"] <= replay_to:
                    continue
                yield _sse_line(item["type"], item["id"], item["data"])
                if item["type"] == "state" and item["data"]["state"] in ("done", "failed", "aborted"):
                    break
        finally:
            try:
                job._subscribers.remove((q, asyncio.get_running_loop()))
            except (ValueError, RuntimeError):
                # RuntimeError：生成器被提前关闭时当前线程可能不在原 loop；
                # 此时按对象身份移除仍安全（_subscribers 存的就是 (q, loop) 对）。
                try:
                    for s in list(job._subscribers):
                        if s[0] is q:
                            job._subscribers.remove(s)
                            break
                except (ValueError, RuntimeError):
                    pass

    def _broadcast(self, job, evt, data) -> None:
        item = {"type": evt, "data": data, "id": job.log_offset}
        for q, loop in list(job._subscribers):
            loop.call_soon_threadsafe(q.put_nowait, item)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _sse_line(evt: str, evt_id: int, data: dict) -> str:
    """progress 事件丢弃 bar 字段（纯展示，重放/续传用不上）。"""
    if evt == "progress" and isinstance(data, dict):
        data = {k: v for k, v in data.items() if k != "bar"}
    return f"event: {evt}\nid: {evt_id}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
