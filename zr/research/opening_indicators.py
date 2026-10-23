"""开盘时段指标处理（audit/base-final 第 2 节）→ docs/research/OPENING_INDICATORS.md。

确定并锁定每个交易日前 19 根 2 分钟 K（09:30–10:06 标签）的 BOLL / MACD 处理方式：
- 口径 C（主口径，D-028）：含盘前盘后连续、不按日重置 → 09:30 那根的 BOLL 窗口是
  08:52–09:30 的 20 根（19 根盘前 + 自身），MACD 的 EMA 从此前所有 K 连续递推，
  开盘第一根就有完整值，不存在部分窗口或空值问题。
- 口径 A（敏感性，D-027）：RTH-only、按日重置 → BOLL 不足 20 根为空值
  （09:30–10:06 共 19 根；10:08 是第 20 根，首个有效值），%b、带宽、斜率为空，
  状态只可能是 OTHER；MACD 的 EMA 每日以 09:30 收盘价做种，第一根起就有值。
- 口径 B（对照）：同 A，但 BOLL 按已有根数算部分窗口（不入任何主检验）。
口径写在 config/v4_decisive.yaml 的 conventions（warmup: nan），
由 tests/test_opening_session.py 锁定。

App 在开盘 40 分钟内的数值没有截图核实（STATUS 0f）：「与 App 一致」只在
11:26/12:34/14:52 三个时刻核实过（A、C 都 15/15 精确相等）；开盘段 C 的依据是
App 图表显示盘前 K、且只有 C 复现负责人读到的 09:30 后 IGNITION 下行（V4_CHECKPOINTS.md）。

运行：uv run python -m zr.research.opening_indicators
"""

from __future__ import annotations

import copy
import datetime as dt
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from zr.config import get_settings
from zr.features.bars import ET
from zr.features.regime import STATE_NAMES
from zr.ops.markdown import df_to_md
from zr.research import v4_checkpoints as cp
from zr.research import v4_decisive as v4d

log = logging.getLogger("zr.research.opening_indicators")
REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs" / "research" / "OPENING_INDICATORS.md"
DAY = dt.date(2026, 10, 21)
HISTORY_FROM = dt.date(2026, 9, 1)  # 口径 C 连续递推的起点（EMA 早已收敛，与全历史等价，见测试）
WINDOW = ("09:30", "10:30")
WARMUP_BARS = 19  # BOLL(20)：按日重置时第 1–19 根不足 20 根


def conventions(v4: dict) -> dict[str, tuple[dict, dict]]:
    """{键: (口径 dict, 状态层配置)}。B = A 的部分窗口对照。"""
    out = {k: (v4["conventions"][k], v4d.state_cfg(v4, k)) for k in ("C", "A")}
    b = copy.deepcopy(v4["conventions"]["A"])
    b["warmup"] = "partial"
    out["B"] = (b, v4d.state_cfg(v4, "A"))
    return out


def opening_table(root: Path, und: str, day: dt.date, v4: dict) -> pd.DataFrame:
    rows = []
    for key, (conv, cfg) in conventions(v4).items():
        b = cp.bars(root, und, HISTORY_FROM, day, conv)
        x, reg, _ = cp.state_layer(b, cfg, conv)
        et = b.index.tz_convert(ET)
        hm = np.asarray(et.strftime("%H:%M"))
        m = (np.asarray(et.date) == day) & (hm >= WINDOW[0]) & (hm <= WINDOW[1])
        for i in np.flatnonzero(m):
            rows.append(
                {
                    "口径": key,
                    "K 标签": hm[i],
                    "App 时刻": (et[i] - pd.Timedelta(minutes=v4["period"])).strftime("%H:%M"),
                    "close": b["close"].iloc[i],
                    "mid": x["mid"].iloc[i],
                    "upper": x["upper"].iloc[i],
                    "lower": x["lower"].iloc[i],
                    "%b": x["pctb"].iloc[i],
                    "DIF": x["dif"].iloc[i],
                    "DEA": x["dea"].iloc[i],
                    "柱": x["hist"].iloc[i],
                    "状态": STATE_NAMES[int(reg["state"].iloc[i])],
                    "方向": int(reg["dir"].iloc[i]),
                }
            )
    return pd.DataFrame(rows)


def warmup_audit(root: Path, und: str, v4: dict) -> dict:
    """全部交易日：口径 C 首根 RTH K 之前的盘前 2 分钟 K 数（< 19 则 09:30 窗口会跨到前一日盘后）；
    口径 A 每日 BOLL 首个有效值是第几根。"""
    conv_c, conv_a = v4["conventions"]["C"], v4["conventions"]["A"]
    end = dt.date(2026, 10, 21)
    bc = cp.bars(root, und, dt.date(2023, 10, 2), end, conv_c)
    et = bc.index.tz_convert(ET)
    pre = pd.Series(np.asarray(et.strftime("%H:%M")) < "09:30").groupby(np.asarray(et.date)).sum()
    ba = cp.bars(root, und, dt.date(2023, 10, 2), end, conv_a)
    xa, _, _ = cp.state_layer(ba, v4d.state_cfg(v4, "A"), conv_a)
    day_a = np.asarray(ba.index.tz_convert(ET).date)
    first_valid = (
        pd.Series(np.isfinite(xa["mid"].to_numpy()), index=day_a)
        .groupby(level=0)
        .apply(lambda s: int(np.argmax(s.to_numpy())) + 1 if s.any() else -1)
    )
    return {
        "交易日": int(len(pre)),
        "C：盘前 2 分钟 K < 19 根的交易日": int((pre < WARMUP_BARS).sum()),
        "C：盘前 2 分钟 K 中位数": float(pre.median()),
        "A：BOLL 首个有效值 = 第 20 根的交易日": int((first_valid == WARMUP_BARS + 1).sum()),
        "A：其他（全天不足 20 根等）": int((first_valid != WARMUP_BARS + 1).sum()),
    }


def _fmt(t: pd.DataFrame) -> pd.DataFrame:
    t = t.copy()
    for c in ("mid", "upper", "lower", "DIF", "DEA", "柱"):
        t[c] = t[c].map(lambda v: "空" if not np.isfinite(v) else f"{v:.3f}")
    t["%b"] = t["%b"].map(lambda v: "空" if not np.isfinite(v) else f"{v:.3f}")
    return t


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = get_settings().zr_data_root
    v4 = v4d.load_v4()
    t = opening_table(root, "MU", DAY, v4)
    audits = {u: warmup_audit(root, u, v4) for u in ("MU", "QQQ")}
    lines = [
        "# 开盘时段指标处理（audit/base-final 第 2 节）",
        "",
        "`uv run python -m zr.research.opening_indicators` 生成。数据：本地长桥 1 分钟 K",
        f"（口径 C 用含盘前盘后、不清洗的原始 K，从 {HISTORY_FROM} 起连续递推；",
        "口径 A / B 用 RTH-only、按日重置），聚合 2 分钟（`resample_bars`，",
        "与长桥官方 2 分钟 K 的一致性见 BASE_FINAL_AUDIT.md 第 1 节）。",
        "BOLL(20,2) ddof=0、MACD(6,13,6) 标准 EMA；状态 = `zr.features.regime.regimes`，",
        "RANGE 阈值按 config/v4_decisive.yaml（C 0.507、A/B 0.3499）。",
        "「App 时刻」= K 标签 − 2 分钟（D-027 经验规律）。",
        "",
        "## 锁定的处理方式",
        "",
        "| 口径 | 开盘前 19 根（09:30–10:06 标签）BOLL | MACD | 状态判定 | 地位 |",
        "|---|---|---|---|---|",
        "| C | 完整窗口：向前延伸到盘前（09:30 = 08:52–09:30 的 20 根） | 跨日连续 EMA "
        "| 正常 | 主口径（D-028）|",
        "| A | **空值**（第 20 根 10:08 起有值）；%b / 带宽 / 斜率为空 "
        "| 每日首根收盘做种，第 1 根起有值 | 前 19 根只可能是 OTHER | 敏感性 |",
        "| B | 部分窗口（按已有根数） | 同 A | 正常 | 对照，不入检验 |",
        "",
        "由 `tests/test_opening_session.py` 锁定（冻结的 MU 2026-10-21 1 分钟 K）。",
        "",
        "**与 App 的一致性只部分核实**：A、C 在 11:26/12:34/14:52 三个截图时刻都 15/15 精确相等；",
        "开盘 40 分钟内 App 的数值没有截图（STATUS 0f），C 作为主口径的依据是 App 图表显示盘前 K、",
        "且只有 C 复现负责人读到的 09:30 后 IGNITION 下行（V4_CHECKPOINTS.md）。",
        "若负责人截图证明 App 在这段时间显示空值或部分窗口，主口径改为 A 或 B，本表即对照。",
        "",
        "## 全部交易日的开盘段覆盖（2023-10-02 ~ 2026-10-21）",
        "",
        df_to_md(pd.DataFrame([{"标的": u, **a} for u, a in audits.items()])),
        "",
        "## MU 2026-10-21 09:30–10:30 每根 2 分钟 K",
        "",
    ]
    for key in ("C", "A", "B"):
        sub = _fmt(t[t["口径"] == key].drop(columns=["口径"]))
        lines += [f"### 口径 {key}", "", df_to_md(sub), ""]
    lines += [
        "## 边界",
        "",
        "- 只列 MU 2026-10-21 一天；开盘段覆盖统计为 MU、QQQ 2023-10-02 ~ 2026-10-21 全部交易日。",
        "- 未覆盖：App 开盘段的真实数值（待截图）；口径 C 的 EMA 起点取 "
        f"{HISTORY_FROM}，与从 2023-10-02 起递推的差异在测试里核对（< 1e-6）。",
        "",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    log.info("写入 %s", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
