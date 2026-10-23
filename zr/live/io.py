"""纸面助手的 I/O：Telegram、JSONL 日志、state.json、账户只读轮询、手动入场文件、历史 K 灌入。

密钥：Telegram 与长桥 token 只从环境变量 / .env 读取（systemd EnvironmentFile），任何日志、
异常信息都不输出 token（异常只记类型名）。账户只读：today_orders（不调用任何下单接口）。
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import queue
import threading
import urllib.request
from pathlib import Path
from typing import Any

from zr.live.core import Entry
from zr.strategy.incremental import Bar

log = logging.getLogger("zr.live")
OPTION_TAIL = r"\d{6}[CP]\d+\.US$"


class Telegram:
    """后台线程发送；缺 TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID 时只记一次「telegram disabled」。"""

    def __init__(self, token: str | None, chat_id: str | None):
        self.enabled = bool(token and chat_id)
        self._token, self._chat = token, chat_id
        self.sent = self.failed = 0
        self._q: queue.Queue = queue.Queue()
        if self.enabled:
            threading.Thread(target=self._loop, daemon=True).start()
        else:
            log.info("telegram disabled")

    @classmethod
    def from_env(cls) -> Telegram:
        return cls(os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID"))

    def __call__(self, text: str) -> None:
        if self.enabled:
            self._q.put(text[:3900])

    def _loop(self) -> None:
        url = f"https://api.telegram.org/bot{self._token}/sendMessage"
        while True:
            text = self._q.get()
            body = json.dumps({"chat_id": self._chat, "text": text}).encode()
            req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
            try:
                urllib.request.urlopen(req, timeout=10).read()
                self.sent += 1
            except Exception as e:  # noqa: BLE001 - 只记类型，URL 含 token，不能打印
                self.failed += 1
                log.warning("telegram 发送失败：%s", type(e).__name__)

    def drain(self, timeout_s: float = 10.0) -> None:
        import time

        end = time.monotonic() + timeout_s
        while self.enabled and not self._q.empty() and time.monotonic() < end:
            time.sleep(0.2)


class JsonlLog:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._fh = open(path, "a", buffering=1, encoding="utf-8")  # noqa: SIM115

    def __call__(self, row: dict) -> None:
        self._fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    def close(self) -> None:
        self._fh.close()


def write_state(path: Path, state: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, default=str, indent=1), encoding="utf-8")
    tmp.replace(path)


# ---------------- 入场来源 ----------------


def _ns(t: Any) -> int:
    if isinstance(t, dt.datetime):
        if t.tzinfo is None:  # SDK 返回 naive UTC（CLAUDE.md）
            t = t.replace(tzinfo=dt.UTC)
        return int(t.timestamp()) * 1_000_000_000 + t.microsecond * 1000
    return int(t)


class AccountPoller:
    """只读轮询 today_orders：期权订单成交数量增加 → 买入 = 负责人入场，卖出 = 负责人自行平仓。
    已处理的成交数量记在 seen 文件（重启不重复）。未验证：today_orders 在模拟盘返回的字段与实盘
    相同（2026-10-22 VPS 只在实盘上确认了接口可调用、当日 0 单）。"""

    def __init__(self, tctx: Any, seen_path: Path):
        import re

        self.tctx = tctx
        self.seen_path = seen_path
        self.seen: dict[str, float] = (
            json.loads(seen_path.read_text()) if seen_path.exists() else {}
        )
        self.re = re.compile(r"^[A-Z.\-]+" + OPTION_TAIL)

    def poll(self) -> tuple[list[Entry], list[tuple]]:
        from zr.broker.serialize import to_py

        entries, sells = [], []
        for o in self.tctx.today_orders():
            r = to_py(o)
            sym = r.get("symbol", "")
            if not self.re.match(sym):
                continue
            oid = str(r.get("order_id"))
            q = float(r.get("executed_quantity") or 0)
            prev = self.seen.get(oid, 0.0)
            if q <= prev:
                continue
            self.seen[oid] = q
            px = float(r.get("executed_price") or 0)
            ns = _ns(o.updated_at) if getattr(o, "updated_at", None) else 0
            if str(r.get("side")) == "Buy":
                entries.append(Entry(ns, sym, int(q - prev), px, "account", oid))
            else:
                sells.append((ns, sym, int(q - prev), px))
        if entries or sells:
            self.seen_path.write_text(json.dumps(self.seen))
        return entries, sells


class ManualEntries:
    """手动入场文件（模拟盘 token 不可用时的后备；每行 JSON：ns/symbol/qty/price）。
    `uv run python -m zr.live.runner enter SYMBOL QTY PRICE` 追加。"""

    def __init__(self, path: Path):
        self.path = path
        self.pos = 0

    def poll(self) -> list[Entry]:
        if not self.path.exists():
            return []
        with open(self.path, encoding="utf-8") as fh:
            fh.seek(self.pos)
            lines = fh.readlines()
            self.pos = fh.tell()
        out = []
        for ln in lines:
            try:
                d = json.loads(ln)
                out.append(
                    Entry(int(d["ns"]), d["symbol"], int(d["qty"]), float(d["price"]), "manual")
                )
            except (ValueError, KeyError):
                log.warning("手动入场行无法解析，跳过")
        return out


# ---------------- 历史 K 灌入 ----------------


def history_bars(qctx: Any, symbol: str, pages: int, before_ns: int) -> list[Bar]:
    """长桥历史 2 分钟 K（TradeSessions.All、NoAdjust），向前翻 pages 页，只保留 ts < before_ns。
    time 游标按美东时间解释（API_FACTS 补充），返回 timestamp 为 naive UTC。"""
    import pandas as pd
    from longport.openapi import AdjustType, Period, TradeSessions

    from zr.data.backfill_bars import bars_to_frame
    from zr.features.bars import ET

    cursor, frames = None, []
    for _ in range(pages):
        bars = qctx.history_candlesticks_by_offset(
            symbol, Period.Min_2, AdjustType.NoAdjust, False, 1000, cursor, TradeSessions.All
        )
        if not bars:
            break
        df = bars_to_frame(symbol, bars)
        frames.append(df)
        cursor = df["ts"].min().tz_convert(ET).tz_localize(None).to_pydatetime() - dt.timedelta(
            minutes=1
        )
    if not frames:
        return []
    df = pd.concat(frames).drop_duplicates("ts").sort_values("ts")
    out = []
    for r in df.itertuples(index=False):
        ns = int(pd.Timestamp(r.ts).value)
        if ns < before_ns:
            out.append(Bar(ns, r.open, r.high, r.low, r.close, float(r.volume)))
    return out


def official_bars(root: Path, symbol: str, before_ns: int, since_days: int = 10) -> list[Bar]:
    """回放用：diag/official_bars/<U>_2m_all.parquet（scripts/diag_pull_candles.py --all）。"""
    import pandas as pd

    o = pd.read_parquet(root / "diag" / "official_bars" / f"{symbol.split('.')[0]}_2m_all.parquet")
    o["ts"] = pd.to_datetime(o["ts"], utc=True)
    o = o.drop_duplicates("ts").sort_values("ts")
    lo = pd.Timestamp(before_ns, tz="UTC") - pd.Timedelta(days=since_days)
    o = o[(o["ts"] >= lo) & (o["ts"] < pd.Timestamp(before_ns, tz="UTC"))]
    return [
        Bar(int(t.value), a, b, c, d, float(v))
        for t, a, b, c, d, v in zip(
            o["ts"], o["open"], o["high"], o["low"], o["close"], o["volume"], strict=True
        )
    ]
