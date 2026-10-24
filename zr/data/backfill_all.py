# ruff: noqa: E501  （内嵌在 VPS 上执行的脚本与中文说明为长行）
"""数据补齐调度（负责人 2026-10-25 指令，分支 data/backfill-all）。

由 scripts/backfill_night.sh 在新加坡时间 22:00–次日 08:00 调用（launchd 每 30 分钟触发，单实例锁）；
每次运行到 --until（默认当晚 08:00 SGT）为止，所有任务按交易日断点续传（完成标记
$ZR_DATA_ROOT/backfill_state/<任务>/<日>.done），每完成一个交易日写一行进度
（$ZR_DATA_ROOT/logs/backfill_progress.log）。

顺序：
0. 网络探测（到 ThetaData 服务器的延迟与丢包，丢包 > 2% 时在日志与摘要中提示更换网络，照常运行）；
   每日增量（新加坡时间 06:00 之后、当日尚未做：最近一个已收盘交易日的 MU / QQQ / SPY 期权 1 分钟
   报价，MU / QQQ 平值附近期权逐笔，MU / QQQ 秒级正股价，VPS 上的长桥 1 分钟与日线）。
1. p1：MU 期权逐笔（报价 + OPRA 成交），2026-04-29～2026-10-21，每日最近两个到期日、开盘价 ±4% 内
   全部看涨看跌，09:29–16:01；从近到远。
2. p2：同样范围，2026-04-28 向前至 2025-01-02（已缓存的时间段由 v4d.pull_ticks 自动跳过）。
3. p3a：QQQ 秒级正股价（看跌看涨平价，zr/data/intrabar_pull.py，intrabar/underlying=QQQ/，qa.csv），
   2026-10-21 向前至 2025-01-02；p3b：QQQ 期权逐笔，最近两个到期日、开盘价 ±1.5% 内，同区间。
4. p4：SPY 期权 1 分钟报价（zr/data/thetadata.py 全量拉取的续传），2026-10-21 向前至 2024-07-31，按月。
5. p5：长桥日线（前复权与不复权），MU / QQQ / SPY / SMH，2021-10-01 至今——在 VPS 上用其现有代码只读
   拉取（ssh 管道传入脚本，不改 VPS 仓库），复制到 $ZR_DATA_ROOT/daily/{fwd,none}/。每晚刷新一次。
DATA_INVENTORY：每次运行结束重写 $ZR_DATA_ROOT/reports/DATA_INVENTORY_BACKFILL.md（各数据集起止日期、
天数、缺口、占用空间）。
运行：uv run python -m zr.data.backfill_all [--until HH:MM] [--only p1,p2,...] [--inventory]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from zr.backtest.intrabar import OptionData
from zr.config import get_settings
from zr.data import intrabar_pull as ip
from zr.data.calendar import trading_days
from zr.data.thetadata import ET, ThetaClient, lb_symbol
from zr.features.bars import load_bars
from zr.research import v4_decisive as v4d

log = logging.getLogger("zr.data.backfill_all")
SGT = ZoneInfo("Asia/Singapore")
THETA_IP = (
    "131.226.212.246"  # 2026-10-24 由 Theta Terminal 的上游连接（lsof）确定；探测时优先实时查询
)
REPO = Path(__file__).resolve().parents[2]
D = dt.date
TASKS = {
    "p1": {"und": "MU", "band": 0.04, "from": D(2026, 10, 21), "to": D(2026, 4, 29)},
    "p2": {"und": "MU", "band": 0.04, "from": D(2026, 4, 28), "to": D(2025, 1, 2)},
    "p3a": {"und": "QQQ", "from": D(2026, 10, 21), "to": D(2025, 1, 2)},
    "p3b": {"und": "QQQ", "band": 0.015, "from": D(2026, 10, 21), "to": D(2025, 1, 2)},
    "p4": {"und": "SPY", "from": D(2026, 10, 21), "to": D(2024, 7, 31)},
    "p5": {},
}
DAILY_SYMBOLS = ("MU.US", "QQQ.US", "SPY.US", "SMH.US")


class Deadline(Exception):
    pass


# ---------------- 基础 ----------------


def state_dir(root: Path, task: str) -> Path:
    p = root / "backfill_state" / task
    p.mkdir(parents=True, exist_ok=True)
    return p


def progress(root: Path, msg: str) -> None:
    p = root / "logs" / "backfill_progress.log"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(f"{dt.datetime.now(SGT):%Y-%m-%d %H:%M:%S} SGT  {msg}\n")
    log.info(msg)


def theta_ip() -> str:
    try:
        out = subprocess.run(
            ["lsof", "-nP", "-iTCP:443", "-sTCP:ESTABLISHED"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
        for line in out.splitlines():
            if line.startswith("java"):
                m = re.search(r"->([\d.]+):443", line)
                if m:
                    return m.group(1)
    except Exception:  # noqa: BLE001 - 查不到就用记录的地址
        pass
    return THETA_IP


def probe(root: Path) -> dict:
    ip_ = theta_ip()
    out = subprocess.run(["ping", "-c", "20", "-q", ip_], capture_output=True, text=True).stdout
    loss = re.search(r"([\d.]+)% packet loss", out)
    rtt = re.search(r"= [\d.]+/([\d.]+)/", out)
    r = {
        "time_sgt": f"{dt.datetime.now(SGT):%Y-%m-%d %H:%M}",
        "ip": ip_,
        "loss_pct": float(loss.group(1)) if loss else None,
        "rtt_ms": float(rtt.group(1)) if rtt else None,
    }
    warn = r["loss_pct"] is None or r["loss_pct"] > 2
    progress(
        root,
        f"网络探测 {ip_}：丢包 {r['loss_pct']}%、平均往返 {r['rtt_ms']} ms"
        + ("  ⚠ 丢包 > 2%，建议负责人更换网络（照常运行）" if warn else ""),
    )
    with (root / "logs" / "backfill_network.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return r


def check(deadline: float) -> None:
    if time.time() >= deadline:
        raise Deadline


# ---------------- 期权逐笔 ----------------


def tick_day(root: Path, od: OptionData, und: str, day: dt.date, band: float) -> dict:
    """一个交易日：最近两个到期日、开盘价 ± band 内全部看涨看跌的全天逐笔报价与 OPRA 成交。"""
    bars = load_bars(root, und, day, day)
    if bars.empty:
        return {"status": "nobars"}
    op = float(bars["open"].iloc[0])
    exps = [e for e in od.expirations() if e >= day][:2]
    rows = []
    for e in exps:
        ks = od.strikes(e)
        for k in ks[(ks >= op * (1 - band)) & (ks <= op * (1 + band))]:
            for right in ("call", "put"):
                rows.append(
                    {
                        "symbol": lb_symbol(und, e, right, float(k)),
                        "day": day,
                        "start": pd.Timestamp(f"{day} 09:29", tz=ET).tz_convert("UTC"),
                        "end": pd.Timestamp(f"{day} 16:01", tz=ET).tz_convert("UTC"),
                        "expiry": e,
                        "strike": float(k),
                        "right": right,
                    }
                )
    if not rows:
        return {"status": "nochain"}
    w = pd.DataFrame(rows)
    st = v4d.pull_ticks(root, und, w, 4)
    syms = sorted(w["symbol"].unique())
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda s: _safe_trades(od, s, day), syms))
    tb = root / "diag" / "v4_trades" / f"underlying={und}" / f"date={day}"
    tq = root / "diag" / "v4_ticks" / f"underlying={und}" / f"date={day}"
    missing = [f.name for f in tq.glob("*.parquet") if not (tb / f.name).exists()]
    ok = st["failed"] == 0 and not missing
    return {
        "status": "ok" if ok else "partial",
        "contracts": len(syms),
        **st,
        "missing_trades": len(missing),
    }


def _safe_trades(od: OptionData, sym: str, day: dt.date) -> None:
    try:
        od._trades(sym, day)
    except Exception as e:  # noqa: BLE001 - 单合约失败不中断，下次续传
        log.warning("OPRA 成交拉取失败 %s %s：%s", sym, day, e)


def run_ticks(root: Path, task: str, deadline: float) -> None:
    cfg = TASKS[task]
    od = OptionData(root, cfg["und"])
    sd = state_dir(root, task)
    for day in sorted(trading_days(cfg["to"], cfg["from"]), reverse=True):
        if (sd / f"{day}.done").exists():
            continue
        check(deadline)
        t0 = time.time()
        r = tick_day(root, od, cfg["und"], day, cfg["band"])
        if r["status"] in ("ok", "nobars", "nochain"):
            (sd / f"{day}.done").write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")
        progress(root, f"{task} {cfg['und']} {day} {r} {time.time() - t0:.0f}s")


# ---------------- QQQ 秒级正股价 ----------------


def run_parity(root: Path, task: str, deadline: float) -> None:
    cfg = TASKS[task]
    client = ThetaClient(concurrency=4)
    sd = state_dir(root, task)
    for day in sorted(trading_days(cfg["to"], cfg["from"]), reverse=True):
        if (sd / f"{day}.done").exists():
            continue
        check(deadline)
        t0 = time.time()
        row = ip.build_one(root, client, day, cfg["und"])
        exists = (ip.base_dir(root, cfg["und"]) / f"date={day}" / "S.parquet").exists()
        if exists:
            (sd / f"{day}.done").write_text(json.dumps(row or {}, default=str), encoding="utf-8")
        progress(root, f"{task} {cfg['und']} {day} {row} {time.time() - t0:.0f}s")


# ---------------- SPY 1 分钟 ----------------


def run_spy_1m(root: Path, task: str, deadline: float) -> None:
    cfg = TASKS[task]
    sd = state_dir(root, task)
    m = cfg["from"].replace(day=1)
    while m >= cfg["to"].replace(day=1):
        a, b = max(m, cfg["to"]), min((m + pd.offsets.MonthEnd(0)).date(), cfg["from"])
        key = f"{a:%Y-%m}"
        if not (sd / f"{key}.done").exists():
            check(deadline)
            t0 = time.time()
            rc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "zr.data.thetadata",
                    "--symbols",
                    "SPY.US",
                    "--start",
                    str(a),
                    "--end",
                    str(b),
                    "--concurrency",
                    "4",
                ],
                cwd=REPO,
            ).returncode
            if rc == 0:
                (sd / f"{key}.done").write_text("ok", encoding="utf-8")
            progress(root, f"{task} SPY {a}～{b} rc={rc} {time.time() - t0:.0f}s")
        m = (pd.Timestamp(m) - pd.DateOffset(months=1)).date()


# ---------------- VPS 长桥 ----------------

VPS_DAILY = r"""
import datetime as dt, sys
import pandas as pd
from longport.openapi import AdjustType, Period, TradeSessions
from zr.broker.longport_client import make_quote_ctx
from zr.data.backfill_bars import bars_to_frame
q = make_quote_ctx()
until = dt.datetime(2021, 9, 1)
for adj, at in (("fwd", AdjustType.ForwardAdjust), ("none", AdjustType.NoAdjust)):
    for sym in sys.argv[1].split(","):
        frames, cursor = [], None
        for _ in range(20):
            bars = q.history_candlesticks_by_offset(sym, Period.Day, at, False, 1000, cursor, TradeSessions.Intraday)
            if not bars:
                break
            df = bars_to_frame(sym, bars)
            frames.append(df)
            page_min = df["ts"].min()
            if page_min.tz_localize(None) <= until or (cursor is not None and page_min.tz_localize(None) >= cursor):
                break
            cursor = page_min.tz_localize(None).to_pydatetime() - dt.timedelta(days=1)
        out = pd.concat(frames).drop_duplicates(["ts", "symbol"]).sort_values("ts")
        out.to_parquet(f"/tmp/zr_daily_{adj}_{sym.split('.')[0]}.parquet", index=False)
        print(adj, sym, len(out), out["ts"].min(), out["ts"].max())
"""


def run_vps_daily(root: Path, task: str, deadline: float) -> None:
    sd = state_dir(root, task)
    key = f"{dt.datetime.now(SGT):%Y-%m-%d}"
    if (sd / f"{key}.done").exists():
        return
    check(deadline)
    cmd = [
        "ssh",
        "-o",
        "ConnectTimeout=20",
        "zr",
        f"cd ~/ZR_Capital && ~/.local/bin/uv run --no-sync python - {','.join(DAILY_SYMBOLS)}",
    ]
    r = subprocess.run(cmd, input=VPS_DAILY, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        progress(root, f"{task} VPS 日线失败 rc={r.returncode}：{r.stderr[-300:]}")
        return
    ok = True
    for adj in ("fwd", "none"):
        dst = root / "daily" / adj
        dst.mkdir(parents=True, exist_ok=True)
        for sym in DAILY_SYMBOLS:
            s = sym.split(".")[0]
            rc = subprocess.run(
                ["scp", "-q", f"zr:/tmp/zr_daily_{adj}_{s}.parquet", str(dst / f"{s}.parquet")]
            ).returncode
            ok &= rc == 0
    subprocess.run(["ssh", "zr", "rm -f /tmp/zr_daily_*.parquet"], capture_output=True)
    summ = "; ".join(x for x in r.stdout.splitlines() if x.startswith(("fwd ", "none ")))
    if ok:
        (sd / f"{key}.done").write_text(summ, encoding="utf-8")
    progress(root, f"{task} 长桥日线（前复权 / 不复权）{'完成' if ok else '部分失败'}：{summ}")


# ---------------- 每日增量 ----------------


def last_session(now_et: pd.Timestamp) -> dt.date:
    days = trading_days(now_et.date() - dt.timedelta(days=10), now_et.date())
    if days and days[-1] == now_et.date() and now_et.time() < dt.time(16, 30):
        days = days[:-1]
    return days[-1]


def daily(root: Path, deadline: float) -> None:
    now = dt.datetime.now(SGT)
    if now.hour < 6 or now.hour >= 22:
        return
    day = last_session(pd.Timestamp.now(tz=ET))
    sd = state_dir(root, "daily")
    if (sd / f"{day}.done").exists():
        return
    check(deadline)
    t0 = time.time()
    res = {}
    rc = subprocess.run(
        [
            sys.executable,
            "-m",
            "zr.data.thetadata",
            "--symbols",
            "MU.US,QQQ.US,SPY.US",
            "--start",
            str(day),
            "--end",
            str(day),
            "--concurrency",
            "4",
        ],
        cwd=REPO,
    ).returncode
    res["options_1m"] = rc
    # 长桥当日 1 分钟（VPS 现有代码，只读拉取）→ 同步 bars_1m 到本机，再算 MU / QQQ 逐笔与秒级价
    r = subprocess.run(
        [
            "ssh",
            "-o",
            "ConnectTimeout=20",
            "zr",
            "cd ~/ZR_Capital && ~/.local/bin/uv run --no-sync python -m zr.data.backfill_bars "
            f"--symbols {','.join(DAILY_SYMBOLS)} --until {day - dt.timedelta(days=5)}",
        ],
        capture_output=True,
        text=True,
        timeout=3600,
    )
    res["lb_1m"] = r.returncode
    rs = subprocess.run(
        ["rsync", "-az", "zr:/home/ubuntu/zr_data/bars_1m/", f"{root}/bars_1m/"],
        capture_output=True,
    )
    res["rsync_bars"] = rs.returncode
    for und, band in (("MU", 0.04), ("QQQ", 0.015)):
        res[f"{und}_ticks"] = tick_day(root, OptionData(root, und), und, day, band)["status"]
        res[f"{und}_S"] = (
            "ok"
            if ip.build_one(root, ThetaClient(concurrency=4), day, und) is not None
            or (ip.base_dir(root, und) / f"date={day}" / "S.parquet").exists()
            else "fail"
        )
    sd2 = state_dir(root, "p5")
    for f in sd2.glob("*.done"):  # 日线随每日增量刷新
        f.unlink()
    run_vps_daily(root, "p5", deadline)
    ok = (
        res["options_1m"] == 0
        and res["lb_1m"] == 0
        and all(v == "ok" for k, v in res.items() if k.endswith(("_ticks", "_S")))
    )
    if ok:
        (sd / f"{day}.done").write_text(json.dumps(res), encoding="utf-8")
    progress(root, f"每日增量 {day} {res} {time.time() - t0:.0f}s")


# ---------------- 总表 ----------------


def _size(p: Path) -> int:
    tot = 0
    for dp, _, fs in os.walk(p):
        for f in fs:
            try:
                tot += os.stat(os.path.join(dp, f)).st_size
            except OSError:
                pass
    return tot


def _days_in(p: Path, pat: str = "date=*") -> list[dt.date]:
    out = []
    for d in p.glob(pat):
        try:
            if any(d.iterdir()):
                out.append(dt.date.fromisoformat(d.name.split("=")[1]))
        except (ValueError, OSError):
            continue
    return sorted(out)


def inventory(root: Path) -> str:
    rows = []

    def add(name, p, days, lo, hi):
        want = [x for x in trading_days(lo, hi)]
        have = set(days)
        gap = [x for x in want if x not in have]
        rows.append(
            {
                "数据集": name,
                "起": str(min(days)) if days else "—",
                "止": str(max(days)) if days else "—",
                "天数": len(days),
                "目标区间缺口": f"{len(gap)} / {len(want)}（{lo}～{hi}）",
                "占用 GB": round(_size(p) / 1e9, 2),
            }
        )

    for und, lo in (("MU", D(2025, 1, 2)), ("QQQ", D(2025, 1, 2))):
        p = root / "diag" / "v4_ticks" / f"underlying={und}"
        add(f"{und} 期权逐笔报价", p, _days_in(p), lo, D(2026, 10, 21))
        p = root / "diag" / "v4_trades" / f"underlying={und}"
        add(f"{und} 期权 OPRA 成交", p, _days_in(p), lo, D(2026, 10, 21))
        p = root / "intrabar" / f"underlying={und}"
        add(f"{und} 秒级正股价", p, _days_in(p), lo, D(2026, 10, 21))
    for und, lo in (("MU", D(2023, 10, 2)), ("QQQ", D(2023, 10, 2)), ("SPY", D(2023, 10, 2))):
        p = root / "options_1m" / f"underlying={und}"
        add(f"{und} 期权 1 分钟报价", p, _days_in(p), lo, D(2026, 10, 21))
    for adj in ("fwd", "none"):
        p = root / "daily" / adj
        for f in sorted(p.glob("*.parquet")):
            d = pd.read_parquet(f, columns=["ts"])
            days = sorted({x.date() for x in pd.to_datetime(d["ts"])})
            rows.append(
                {
                    "数据集": f"{f.stem} 长桥日线（{'前复权' if adj == 'fwd' else '不复权'}）",
                    "起": str(days[0]),
                    "止": str(days[-1]),
                    "天数": len(days),
                    "目标区间缺口": "—",
                    "占用 GB": round(f.stat().st_size / 1e9, 3),
                }
            )
    df = pd.DataFrame(rows)
    from zr.ops.markdown import df_to_md

    txt = (
        f"# 数据总表（zr/data/backfill_all.py 自动生成，{dt.datetime.now(SGT):%Y-%m-%d %H:%M} SGT）\n\n"
        + df_to_md(df)
        + "\n\n缺口 = 目标区间内的交易日中没有文件的天数（期权逐笔按「当日有任一合约文件」计，不检查合约是否齐全；"
        "合约级完成情况见 backfill_state/）。\n"
    )
    out = root / "reports" / "DATA_INVENTORY_BACKFILL.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(txt, encoding="utf-8")
    return txt


def status(root: Path) -> str:
    """各优先级完成到哪一天、剩余量。"""
    lines = []
    for task, cfg in TASKS.items():
        if task in ("p5", "p4"):
            continue
        want = sorted(trading_days(cfg["to"], cfg["from"]), reverse=True)
        done = {f.stem for f in state_dir(root, task).glob("*.done")}
        left = [d for d in want if str(d) not in done]
        reached = next((d for d in want if str(d) not in done), None)
        lines.append(
            f"{task}：完成 {len(want) - len(left)} / {len(want)} 个交易日"
            + (f"，下一个待拉 {reached}" if reached else "，全部完成")
        )
    sd = state_dir(root, "p4")
    lines.append(f"p4：完成 {len(list(sd.glob('*.done')))} 个月（目标 2024-07～2026-10，27 个月）")
    lines.append(
        f"p5：最近完成 {max((f.stem for f in state_dir(root, 'p5').glob('*.done')), default='无')}"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--until", default="08:00", help="新加坡时间 HH:MM（跨午夜按次日）")
    ap.add_argument("--only", default=None)
    ap.add_argument("--inventory", action="store_true")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = get_settings().zr_data_root
    if args.inventory:
        print(inventory(root))
        return 0
    if args.status:
        print(status(root))
        return 0
    now = dt.datetime.now(SGT)
    h, m = map(int, args.until.split(":"))
    end = now.replace(hour=h, minute=m, second=0, microsecond=0)
    if end <= now:
        end += dt.timedelta(days=1)
    deadline = end.timestamp()
    probe(root)
    tasks = args.only.split(",") if args.only else ["p5", "p1", "p2", "p3a", "p3b", "p4"]
    runners = {
        "p1": run_ticks,
        "p2": run_ticks,
        "p3a": run_parity,
        "p3b": run_ticks,
        "p4": run_spy_1m,
        "p5": run_vps_daily,
    }
    try:
        daily(root, deadline)
        for t in tasks:
            runners[t](root, t, deadline)
            daily(root, deadline)
        progress(root, "本轮全部任务完成")
    except Deadline:
        progress(root, f"到达截止时间 {end:%H:%M} SGT，暂停（下次自动续传）")
    finally:
        inventory(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
