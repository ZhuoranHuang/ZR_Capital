"""标的池：从 config/universe.yaml 加载，ZR_SYMBOLS 作为子集过滤。规范见 docs/UNIVERSE_SPEC.md。"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml
from pydantic import BaseModel, Field

from zr.config import get_settings

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config" / "universe.yaml"


@dataclass(frozen=True)
class ChainRule:
    """录制器订阅集规则：最近 n 个到期日、ATM 两侧各 N 档。"""

    n_expiries: int
    strikes_each_side: int


class UnderlyingSpec(BaseModel):
    n_expiries: int = Field(ge=1)
    strikes_each_side: int = Field(ge=0)
    strike_band_pct: float = Field(gt=0, lt=1)
    max_dte: int = Field(ge=0)

    def chain_rule(self) -> ChainRule:
        return ChainRule(self.n_expiries, self.strikes_each_side)


class Universe(BaseModel):
    subscription_cap: int = Field(ge=1, le=500)
    underlyings: dict[str, UnderlyingSpec]

    @property
    def symbols(self) -> list[str]:
        return list(self.underlyings)

    def chain_rules(self) -> dict[str, ChainRule]:
        return {s: spec.chain_rule() for s, spec in self.underlyings.items()}

    def filtered(self, symbols: list[str]) -> Universe:
        unknown = [s for s in symbols if s not in self.underlyings]
        if unknown:
            raise ValueError(f"ZR_SYMBOLS 含 universe 之外的代码：{unknown}")
        return Universe(
            subscription_cap=self.subscription_cap,
            underlyings={s: self.underlyings[s] for s in symbols},
        )


def load_universe(path: Path | None = None, symbols: list[str] | None = None) -> Universe:
    """读取 yaml 并按 symbols（默认 ZR_SYMBOLS）过滤。"""
    raw = yaml.safe_load((path or DEFAULT_PATH).read_text(encoding="utf-8"))
    uni = Universe.model_validate(raw)
    selected = symbols if symbols is not None else get_settings().symbols
    return uni.filtered(selected)


# ---------- 周度快照（systemd zr-universe 每周五收盘后运行） ----------


def snapshot(uni: Universe, today: dt.date, qctx: Any = None) -> dict[str, Any]:
    """静态规则 + （可选）从期权链解析出的下周订阅集。API 失败只记录错误。"""
    out: dict[str, Any] = {
        "generated_at_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "as_of": today.isoformat(),
        "universe": uni.model_dump(),
        "resolved": {},
        "errors": {},
    }
    if qctx is None:
        return out
    from zr.data.recorder import Recorder, build_subscription_set

    try:
        rec = Recorder(qctx, sink=None, rules=uni.chain_rules(), cap=uni.subscription_cap)
        chains, spots = rec.fetch_chains(today)
        subs = build_subscription_set(chains, spots, today, uni.chain_rules(), uni.subscription_cap)
        for u in uni.symbols:
            out["resolved"][u] = {
                "spot": spots.get(u),
                "expiries": [d.isoformat() for d in sorted(chains.get(u, {}))],
                "n_strikes_per_expiry": {
                    d.isoformat(): len(v) for d, v in sorted(chains.get(u, {}).items())
                },
            }
        out["subscription_set"] = subs
        out["subscription_count"] = len(subs)
    except Exception as exc:  # noqa: BLE001
        out["errors"]["resolve"] = f"{type(exc).__name__}: {exc}"
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="写 universe 快照到 ZR_DATA_ROOT/universe/")
    parser.add_argument("--offline", action="store_true", help="不连接 API")
    args = parser.parse_args(argv)

    uni = load_universe()
    today = dt.datetime.now(ZoneInfo("America/New_York")).date()
    qctx = None
    if not args.offline:
        try:
            from zr.broker.longport_client import make_quote_ctx

            qctx = make_quote_ctx()
        except Exception as exc:  # noqa: BLE001
            print(f"quote ctx 不可用，只写静态快照：{type(exc).__name__}: {exc}")
    snap = snapshot(uni, today, qctx)

    out_dir = get_settings().zr_data_root / "universe"
    out_dir.mkdir(parents=True, exist_ok=True)
    body = json.dumps(snap, ensure_ascii=False, indent=2)
    (out_dir / f"universe_{today.isoformat()}.json").write_text(body, encoding="utf-8")
    (out_dir / "latest.json").write_text(body, encoding="utf-8")
    print(
        f"universe 快照 -> {out_dir}  标的 {len(uni.symbols)}  "
        f"订阅 {snap.get('subscription_count', 'n/a')}  错误 {list(snap['errors'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
