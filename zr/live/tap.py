"""录制器旁路（tap）：把录制器收到的同一批推送，按需写成紧凑 JSONL，供纸面助手实时读取。

不新增订阅、不另开行情连接：录制器回调里多写一行（opt-in：环境变量 ZR_LIVE_TAP_DIR 非空才启用）。
只写助手需要的：
- 正股逐笔成交（标的在 ZR_TAP_UNDERLYINGS 里，默认 MU.US,QQQ.US）：k="t"，
  x=[[时间戳, 价, 量, 类型], ...]
- 这些标的的期权 1 档盘口：k="d"，b/a = 买一 / 卖一价（无报价为 null），bs/as = 挂单量
- 这些标的的期权逐笔成交：k="ot"，x=[[时间戳, 价, 量], ...]
r = 录制器接收时刻（UTC 纳秒）。文件按美东日期：<dir>/YYYY-MM-DD.jsonl，保留最近 3 天。
写失败只计数、不抛出，不能影响录制器本身。
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

log = logging.getLogger("zr.live.tap")
ET = ZoneInfo("America/New_York")
OPTION_RE = re.compile(r"^(?P<und>[A-Z.\-]+?)\d{6}[CP]\d+\.US$")
KEEP_DAYS = 3


def tap_path(tap_dir: Path, day: dt.date) -> Path:
    return tap_dir / f"{day.isoformat()}.jsonl"


def encode(symbol: str, kind: str, payload: dict, recv_ns: int, unds: set[str]) -> dict | None:
    """把一条推送编码成 tap 行；不需要的返回 None。payload = 录制器落盘的 JSON（已解析）。"""
    m = OPTION_RE.match(symbol)
    if m is None:
        if kind != "trade" or symbol not in unds:
            return None
        x = [
            [t.get("timestamp"), t.get("price"), t.get("volume"), t.get("trade_type", "")]
            for t in payload.get("trades") or []
        ]
        return {"r": recv_ns, "s": symbol, "k": "t", "x": x} if x else None
    if f"{m['und']}.US" not in unds:
        return None
    if kind == "depth":
        bids, asks = payload.get("bids") or [], payload.get("asks") or []
        b = bids[0] if bids else {}
        a = asks[0] if asks else {}
        return {
            "r": recv_ns,
            "s": symbol,
            "k": "d",
            "b": b.get("price"),
            "a": a.get("price"),
            "bs": b.get("volume"),
            "as": a.get("volume"),
        }
    if kind == "trade":
        x = [
            [t.get("timestamp"), t.get("price"), t.get("volume")]
            for t in payload.get("trades") or []
        ]
        return {"r": recv_ns, "s": symbol, "k": "ot", "x": x} if x else None
    return None


class TapWriter:
    def __init__(self, tap_dir: Path, unds: set[str], now_fn=lambda: dt.datetime.now(ET)):
        self.dir = Path(tap_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.unds = unds
        self.now_fn = now_fn
        self.errors = 0
        self.lines = 0
        self._lock = threading.Lock()
        self._day: dt.date | None = None
        self._fh = None
        self._cleanup()

    @classmethod
    def from_env(cls) -> TapWriter | None:
        d = os.environ.get("ZR_LIVE_TAP_DIR", "").strip()
        if not d:
            return None
        unds = {
            s.strip()
            for s in os.environ.get("ZR_TAP_UNDERLYINGS", "MU.US,QQQ.US").split(",")
            if s.strip()
        }
        log.info("tap 启用：%s，标的 %s", d, sorted(unds))
        return cls(Path(d), unds)

    def _cleanup(self) -> None:
        today = self.now_fn().date()
        for f in self.dir.glob("*.jsonl"):
            try:
                if (today - dt.date.fromisoformat(f.stem)).days >= KEEP_DAYS:
                    f.unlink()
            except ValueError:
                continue

    def _file(self):
        day = self.now_fn().date()
        if day != self._day:
            if self._fh is not None:
                self._fh.close()
            self._fh = open(tap_path(self.dir, day), "a", buffering=1, encoding="utf-8")  # noqa: SIM115
            self._day = day
        return self._fh

    def wanted(self, symbol: str, kind: str) -> bool:
        m = OPTION_RE.match(symbol)
        if m is None:
            return kind == "trade" and symbol in self.unds
        return kind in ("depth", "trade") and f"{m['und']}.US" in self.unds

    def write(self, symbol: str, kind: str, payload_json: str, recv_ns: int) -> None:
        try:
            if not self.wanted(symbol, kind):  # 先按代码过滤，免得为 SPY 等解析 JSON
                return
            row = encode(symbol, kind, json.loads(payload_json), recv_ns, self.unds)
            if row is None:
                return
            line = json.dumps(row, separators=(",", ":"))
            with self._lock:
                self._file().write(line + "\n")
                self.lines += 1
        except Exception:  # noqa: BLE001
            self.errors += 1
            if self.errors <= 5:
                log.exception("tap 写入失败（不影响录制）")

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None


class TapReader:
    """增量读取当日 tap 文件（tail）：每次 poll() 返回自上次以来新写入的完整行。"""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.pos = 0
        self._buf = ""

    def poll(self) -> list[dict]:
        if not self.path.exists():
            return []
        with open(self.path, encoding="utf-8") as fh:
            fh.seek(self.pos)
            chunk = fh.read()
            self.pos = fh.tell()
        if not chunk:
            return []
        data = self._buf + chunk
        lines = data.split("\n")
        self._buf = lines[-1]  # 最后一段可能写了一半
        out = []
        for ln in lines[:-1]:
            if ln:
                try:
                    out.append(json.loads(ln))
                except json.JSONDecodeError:
                    continue
        return out


def ts_ns(stamp: Any) -> int:
    """tap 里成交时间戳（SDK naive UTC，ISO 字符串）→ UTC 纳秒。"""
    t = dt.datetime.fromisoformat(str(stamp))
    if t.tzinfo is None:
        t = t.replace(tzinfo=dt.UTC)
    return int(t.timestamp()) * 1_000_000_000 + t.microsecond * 1000
