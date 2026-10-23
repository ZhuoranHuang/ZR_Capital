"""纸面护栏助手：实时运行、录制数据回放、手动入场、状态查看、kill switch。

  uv run python -m zr.live.runner run                 # VPS：09:20–16:05 ET（systemd zr-assistant）
  uv run python -m zr.live.runner replay 2026-10-22 --entry "10:30:05,MU261023C900000.US,2"
  uv run python -m zr.live.runner enter MU261023C900000.US 2 13.25   # 手动登记入场（后备）
  uv run python -m zr.live.runner status              # 打印 state.json
  uv run python -m zr.live.runner kill                # 写 kill switch 文件：全部纸面平仓并退出

数据：录制器旁路 $ZR_DATA_ROOT/live/tap/YYYY-MM-DD.jsonl
（zr/live/tap.py，录制器 ZR_LIVE_TAP_DIR 启用）。
输出：$ZR_DATA_ROOT/live/{state.json, trades_YYYY-MM-DD.jsonl, DAILY_YYYY-MM-DD.md}，
docs/live/DAILY_YYYY-MM-DD.md。只做纸面，不向券商提交任何订单。
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import logging
import time
from pathlib import Path

import pandas as pd
import yaml

from zr.features.bars import ET
from zr.live import report
from zr.live.core import Assistant, Entry, et_of, hm
from zr.live.io import (
    AccountPoller,
    JsonlLog,
    ManualEntries,
    Telegram,
    history_bars,
    official_bars,
    write_state,
)
from zr.live.tap import TapReader, encode, tap_path
from zr.research import v4_decisive as v4d
from zr.strategy.incremental import NS_MIN

log = logging.getLogger("zr.live.runner")
REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "config" / "assistant.yaml"
SEED_DELAY_NS = 3_000_000_000


def load_cfg(path: Path = CONFIG) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def live_dir(root: Path) -> Path:
    d = root / "live"
    d.mkdir(parents=True, exist_ok=True)
    return d


def make_assistant(cfg: dict, day: dt.date, notify, logger, week_before: float = 0.0):
    v4 = v4d.load_v4()
    return Assistant(
        cfg, v4d.state_cfg(v4, cfg["convention"]), v4, day, notify, logger, week_before
    )


def maybe_seed(asst: Assistant, now: int, bars_fn) -> None:
    """某标的收到第一笔逐笔成交后，等它所在分箱结束 + 3 s，用历史 K 灌到该分箱（含），
    之后的分箱由逐笔构建（第一箱可能不完整，不用逐笔）。"""
    p = asst.cfg["period_min"] * NS_MIN
    for u, f in asst.feeds.items():
        if f.live or f.first_trade_ns is None:
            continue
        b0 = f.first_trade_ns - f.first_trade_ns % p
        if now >= b0 + p + SEED_DELAY_NS:
            bars = bars_fn(u, b0 + p)
            f.seed(bars, b0 + p)
            asst.log({"type": "seeded", "und": u, "bars": len(bars), "start_bin": b0 + p})
            asst.notify(
                f"[就绪] {u} 状态层已灌入 {len(bars)} 根历史 K，从 "
                f"{et_of(b0 + p).strftime('%H:%M')} 起实时"
            )


# ---------------- 周累计（风控） ----------------


def week_realized_before(ldir: Path, day: dt.date) -> float:
    """本周（周一起）此前各日的纸面已实现盈亏（来自各日 trades_*.jsonl 的 closed 记录）。"""
    monday = day - dt.timedelta(days=day.weekday())
    tot = 0.0
    for f in glob.glob(str(ldir / "trades_*.jsonl")):
        d = dt.date.fromisoformat(Path(f).stem.split("_")[1])
        if monday <= d < day:
            for ln in open(f, encoding="utf-8"):
                r = json.loads(ln)
                if r.get("type") == "closed":
                    tot += float(r.get("realized") or 0)
    return tot


# ---------------- 实时 ----------------


def run_live(args) -> int:
    from zr.config import get_settings
    from zr.data.calendar import is_trading_day

    cfg = load_cfg()
    root = get_settings().zr_data_root
    now_et = dt.datetime.now(ET)
    day = now_et.date()
    if not args.now and not is_trading_day(day):
        log.info("%s 非交易日，退出", day)
        return 0
    start, end = hm(cfg["session"]["start"]), hm(cfg["session"]["end"])
    if not args.now and now_et.time() < start:
        wait = (dt.datetime.combine(day, start, ET) - now_et).total_seconds()
        log.info("等待到 %s ET（%.0f s）", start, wait)
        time.sleep(max(0.0, wait))
    ldir = live_dir(root)
    kill_file = ldir / "KILL"
    if kill_file.exists():
        kill_file.unlink()  # 上一次的 kill 文件不影响新的一天
    tg = Telegram.from_env()
    tl = ldir / f"trades_{day.isoformat()}.jsonl"
    prior = (
        [json.loads(ln) for ln in tl.read_text(encoding="utf-8").splitlines()]
        if tl.exists()
        else []
    )
    jl = JsonlLog(tl)
    asst = make_assistant(cfg, day, tg, jl, week_realized_before(ldir, day))
    restored = asst.restore(prior)
    if restored:
        tg(f"[重启] 已从当日日志恢复 {restored} 笔持仓记录")

    from zr.broker.longport_client import make_quote_ctx

    qctx = make_quote_ctx()  # 行情只读（历史 K 灌入）

    def bars_fn(u, before):
        return history_bars(qctx, u, int(cfg["seed_pages"]), before)

    poller = None
    try:
        from zr.broker.longport_client import make_trade_ctx

        tctx = make_trade_ctx(cfg["intake"]["account"])  # 只读 today_orders
        poller = AccountPoller(tctx, ldir / f"intake_seen_{day.isoformat()}.json")
    except Exception as e:  # noqa: BLE001 - 缺模拟盘 token 等：只记类型，退回手动入场
        log.warning("账户轮询不可用（%s），只接受手动入场文件", type(e).__name__)
        tg(f"[注意] 账户轮询不可用（{type(e).__name__}）；入场请用 runner enter 手动登记")
    manual = ManualEntries(ldir / f"manual_entries_{day.isoformat()}.jsonl")
    reader = TapReader(tap_path(ldir / "tap", day))
    tg(f"[启动] 纸面助手 {day}，口径 {cfg['convention']}，标的 {', '.join(cfg['underlyings'])}")
    last_state = last_poll = 0.0
    while True:
        now = time.time_ns()
        if et_of(now).time() >= end:
            break
        for row in reader.poll():
            asst.on_tap(row)
        maybe_seed(asst, now, bars_fn)
        asst.on_clock(now)
        mono = time.monotonic()
        if mono - last_poll >= float(cfg["intake"]["poll_s"]):
            last_poll = mono
            for e in manual.poll():
                asst.on_entry(e)
            if poller is not None:
                try:
                    entries, sells = poller.poll()
                    for e in entries:
                        asst.on_entry(e)
                    for s in sells:
                        asst.on_owner_sell(*s)
                except Exception as e:  # noqa: BLE001
                    log.warning("账户轮询失败：%s", type(e).__name__)
        if mono - last_state >= float(cfg["state_refresh_s"]):
            last_state = mono
            write_state(ldir / "state.json", asst.snapshot(now))
        if kill_file.exists():
            asst.kill(now)
            break
        time.sleep(0.2)
    now = time.time_ns()
    write_state(ldir / "state.json", asst.snapshot(now))
    paths = report.write(asst, day, [ldir, REPO / "docs" / "live"])
    tg(
        f"[日终] {day}：入场 {len(asst.positions)} 笔，纸面盈亏 {asst.day_pnl():.2f} USD；"
        f"报告 {paths[-1].name}"
    )
    tg.drain()
    jl.close()
    return 0


# ---------------- 回放 ----------------


def recorder_rows(root: Path, day: dt.date, unds: set[str]) -> list[dict]:
    """录制器落盘的原始推送 → tap 行（与实时同一编码），按接收时刻排序。"""
    rows = []
    for u in unds:
        base = root / "recorder" / f"date={day}" / f"underlying={u.split('.')[0]}"
        for kind in ("trade", "depth"):
            for f in sorted(glob.glob(str(base / f"{kind}_*.parquet"))):
                d = pd.read_parquet(f, columns=["recv_ts_utc_ns", "symbol", "payload"])
                for r, s, p in zip(d["recv_ts_utc_ns"], d["symbol"], d["payload"], strict=True):
                    e = encode(s, kind, json.loads(p), int(r), unds)
                    if e is not None:
                        rows.append(e)
    rows.sort(key=lambda x: x["r"])
    return rows


def replay(
    root: Path,
    day: dt.date,
    entries: list[tuple],
    cfg: dict | None = None,
    out_dir: Path | None = None,
) -> Assistant:
    """entries = [(ET 时刻 "HH:MM:SS", 合约, 张数, 价格或 None=当时卖一)]。"""
    cfg = cfg or load_cfg()
    msgs: list[str] = []
    out_dir = out_dir or (live_dir(root) / "replay")
    jl = JsonlLog(out_dir / f"trades_{day.isoformat()}.jsonl")
    asst = make_assistant(cfg, day, msgs.append, jl)
    rows = recorder_rows(root, day, set(cfg["underlyings"]))
    todo = sorted(
        (int(pd.Timestamp(f"{day} {t}", tz=ET).value), s, int(q), p) for t, s, q, p in entries
    )
    end = int(pd.Timestamp(f"{day} {cfg['session']['end']}", tz=ET).value)

    def bars_fn(u, before):
        return official_bars(root, u, before)

    last_clock = 0
    for row in rows:
        now = row["r"]
        while todo and todo[0][0] <= now:
            t, s, q, p = todo.pop(0)
            qt = asst.quotes.get(s)
            px = p if p is not None else (qt[2] if qt else None)
            if px is None:
                msgs.append(f"[回放] {s} 在 {et_of(t)} 无报价，入场跳过")
                continue
            asst.on_entry(Entry(t, s, q, float(px), "replay"))
        if now >= end:
            break
        asst.on_tap(row)
        if now - last_clock >= 1_000_000_000:
            maybe_seed(asst, now, bars_fn)
            asst.on_clock(now)
            last_clock = now
    asst.on_clock(min(end, rows[-1]["r"]) if rows else end)
    report.write(asst, day, [out_dir])
    write_state(out_dir / f"state_{day.isoformat()}.json", asst.snapshot(last_clock))
    (out_dir / f"notify_{day.isoformat()}.txt").write_text("\n".join(msgs), encoding="utf-8")
    jl.close()
    return asst


# ---------------- CLI ----------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="纸面护栏助手")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--now", action="store_true", help="跳过交易日判断与开盘前等待")
    rp = sub.add_parser("replay")
    rp.add_argument("day")
    rp.add_argument("--entry", action="append", default=[], help="HH:MM:SS,合约,张数[,价格]")
    en = sub.add_parser("enter")
    en.add_argument("symbol")
    en.add_argument("qty", type=int)
    en.add_argument("price", type=float)
    sub.add_parser("status")
    sub.add_parser("kill")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from zr.config import get_settings

    root = get_settings().zr_data_root
    if args.cmd == "run":
        return run_live(args)
    if args.cmd == "replay":
        ents = []
        for e in args.entry:
            parts = e.split(",")
            ents.append(
                (parts[0], parts[1], int(parts[2]), float(parts[3]) if len(parts) > 3 else None)
            )
        a = replay(root, dt.date.fromisoformat(args.day), ents)
        print(json.dumps(a.snapshot(0)["positions"], ensure_ascii=False, indent=1))
        return 0
    ldir = live_dir(root)
    if args.cmd == "enter":
        day = dt.datetime.now(ET).date()
        row = {"ns": time.time_ns(), "symbol": args.symbol, "qty": args.qty, "price": args.price}
        with open(ldir / f"manual_entries_{day.isoformat()}.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        print("已登记", row)
        return 0
    if args.cmd == "status":
        p = ldir / "state.json"
        print(p.read_text(encoding="utf-8") if p.exists() else "无 state.json（助手未运行）")
        return 0
    if args.cmd == "kill":
        (ldir / "KILL").write_text(dt.datetime.now(ET).isoformat(), encoding="utf-8")
        print("已写 kill switch：", ldir / "KILL")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
