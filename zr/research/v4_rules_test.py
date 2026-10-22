"""R-035：负责人日内单腿样本提取的规则（config/v4_rules.yaml，冻结）在 MU 上回放。

引擎、护栏、合约、tick 成交、统计与 R-033 完全相同（zr/research/v4_decisive.run_conv），
只换信号来源（规则 → 每根 K 多/空两个布尔数组）、离场映射和日期集合。
主检验 = 2024-10-22～2026-10-21 中剔除规则提取期的日子；提取期单列（描述性，不进 FDR）。
运行（需本机 Theta Terminal）：uv run python -m zr.research.v4_rules_test [--no-pull]
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
from pathlib import Path

import pandas as pd
import yaml

from zr.config import get_settings
from zr.research import intraday_single_leg as isl
from zr.research import v4_decisive as v4d

log = logging.getLogger("zr.research.v4_rules_test")
REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "config" / "v4_rules_test.yaml"


def load_spec() -> tuple[dict, list[dict]]:
    spec = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    rules = yaml.safe_load((REPO / spec["rules_file"]).read_text(encoding="utf-8"))["rules"]
    return spec, rules


def make_sig_fn(rules: list[dict], v4: dict):
    def sig_fn(df, x, reg, cfg, conv):
        f = isl.bar_features(df, x, reg, reset=conv["reset"])
        arr = {d: isl.oriented_arrays(f, d) for d in (1, -1)}
        return {r["id"]: (isl.rule_mask(arr[1], r["conditions"]),
                          isl.rule_mask(arr[-1], r["conditions"])) for r in rules}

    return sig_fn


def make_days_fn(spec: dict, v4: dict, part: str):
    w0, w1 = (dt.date.fromisoformat(spec["window"][k]) for k in ("from", "to"))
    e0, e1 = (dt.date.fromisoformat(spec["extraction_window"][k]) for k in ("from", "to"))

    def days_fn(root, und):
        days = [d for d in v4d.sample_days(root, und, v4) if w0 <= d <= w1]
        inside = [d for d in days if e0 <= d <= e1]
        return inside if part == "extract" else [d for d in days if d not in set(inside)]

    return days_fn


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pull", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = get_settings().zr_data_root
    v4 = v4d.load_v4()
    spec, rules = load_spec()
    ids = tuple(r["id"] for r in rules)
    exit_map = {k: dict.fromkeys(ids, rule) for k, rule in spec["exits"].items()}
    out = root / "research" / "v4"
    out.mkdir(parents=True, exist_ok=True)
    for part in ("main", "extract"):
        for conv in (spec["base"], spec["contrast"]):
            res = v4d.run_conv(root, conv, v4, pull=not args.no_pull,
                               sig_fn=make_sig_fn(rules, v4), hyps=ids, exit_map=exit_map,
                               days_fn=make_days_fn(spec, v4, part), unds=spec["underlyings"])
            res["cells"].to_parquet(out / f"rules_cells_{conv}_{part}.parquet", index=False)
            if len(res["trades"]):
                res["trades"].to_parquet(out / f"rules_trades_{conv}_{part}.parquet", index=False)
            pd.Series({"pull": res["pull"], "info": res["info"]}).to_json(
                out / f"rules_run_{conv}_{part}.json", default_handler=str)
            log.info("R-035 %s %s 完成：%d 格", part, conv, len(res["cells"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
