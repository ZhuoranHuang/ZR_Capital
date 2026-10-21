"""录制器健康检查（systemd 每 5 分钟运行）。

盘中（美东 9:30–16:00，周一至周五）若 zr-recorder 进程不在，或 5 分钟内没有新落盘文件，
写一行 JSON 到 ZR_DATA_ROOT/alerts/YYYY-MM-DD.jsonl，同时打印到 stdout（journald），退出码 1。
非盘中退出 0，不写记录。

不查交易日历（避免依赖 API）：美股假日会误报，由负责人在 alerts 中识别。
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
from pathlib import Path
from zoneinfo import ZoneInfo

from zr.config import get_settings

ET = ZoneInfo("America/New_York")
SESSION_START = dt.time(9, 30)
SESSION_END = dt.time(16, 0)
MAX_FILE_AGE_S = 300.0
SERVICE = "zr-recorder.service"


def in_session(now_et: dt.datetime) -> bool:
    return now_et.weekday() < 5 and SESSION_START <= now_et.time() < SESSION_END


def newest_file_age_s(recorder_root: Path, now_utc: dt.datetime, date_et: dt.date) -> float | None:
    """当日分区下最新 parquet 的年龄（秒）；没有文件返回 None。"""
    part = recorder_root / f"date={date_et.isoformat()}"
    if not part.is_dir():
        return None
    newest = max((p.stat().st_mtime for p in part.rglob("*.parquet")), default=None)
    if newest is None:
        return None
    return now_utc.timestamp() - newest


def evaluate(
    now_et: dt.datetime,
    recorder_active: bool,
    file_age_s: float | None,
    max_age_s: float = MAX_FILE_AGE_S,
) -> list[str]:
    """返回问题列表；非盘中恒为空。"""
    if not in_session(now_et):
        return []
    problems = []
    if not recorder_active:
        problems.append("recorder_not_running")
    if file_age_s is None:
        problems.append("no_parquet_today")
    elif file_age_s > max_age_s:
        problems.append(f"stale_parquet_{int(file_age_s)}s")
    return problems


def recorder_active() -> bool:
    if shutil.which("systemctl"):
        r = subprocess.run(
            ["systemctl", "is-active", "--quiet", SERVICE], check=False, capture_output=True
        )
        return r.returncode == 0
    r = subprocess.run(["pgrep", "-f", "zr.data.recorder"], check=False, capture_output=True)
    return r.returncode == 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="录制器健康检查")
    parser.add_argument("--force", action="store_true", help="忽略时段判断，立即检查")
    args = parser.parse_args(argv)

    root = get_settings().zr_data_root
    now_utc = dt.datetime.now(dt.UTC)
    now_et = now_utc.astimezone(ET)
    active = recorder_active()
    age = newest_file_age_s(root / "recorder", now_utc, now_et.date())

    if args.force:
        problems = evaluate(now_et.replace(hour=10, minute=0), active, age)
        # 周末 --force 仍会被 weekday 过滤；仅供盘中手动调试
    else:
        problems = evaluate(now_et, active, age)

    record = {
        "ts_utc": now_utc.isoformat(timespec="seconds"),
        "ts_et": now_et.isoformat(timespec="seconds"),
        "in_session": in_session(now_et),
        "recorder_active": active,
        "newest_file_age_s": None if age is None else round(age),
        "problems": problems,
    }
    if problems:
        alerts = root / "alerts"
        alerts.mkdir(parents=True, exist_ok=True)
        with (alerts / f"{now_et.date().isoformat()}.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print("ALERT " + json.dumps(record, ensure_ascii=False))
        return 1
    print("ok " + json.dumps(record, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
