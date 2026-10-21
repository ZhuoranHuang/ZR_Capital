"""标的池：从 config/universe.yaml 加载，ZR_SYMBOLS 作为子集过滤。规范见 docs/UNIVERSE_SPEC.md。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
