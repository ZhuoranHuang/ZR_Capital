"""公开策略调研（zr/research/strategy_survey.py）。

来源表与 survey_*.yaml 的结构校验、oos_pick 判定、报告渲染。
"""

from __future__ import annotations

import pytest
import yaml

from zr.research import strategy_survey as ss


def test_every_survey_yaml_parses_with_required_fields():
    paths = sorted(ss.RULES.glob("survey_*.yaml"))
    assert 5 <= len(paths) <= 8
    for p in paths:
        r = yaml.safe_load(p.read_text(encoding="utf-8"))
        assert r["direction"] in ("long", "short", "both", "trend"), p.name
        assert r["entry"]["all"], p.name
        assert {"tp", "sl", "max_hold_min"} <= set(r["exit"]), p.name
        assert isinstance(r["oos_pick"], bool), p.name


def test_sources_and_rules_consistent():
    rows = ss.load_sources()
    rules = ss.load_rules()
    ss.check_rules(rules, rows)  # 不抛错即通过
    picks = sorted(n for n, r in rules.items() if r["oos_pick"])
    assert picks == ["survey_ema_pullback", "survey_orb_breakout", "survey_rsi2_pullback"]


def test_check_rules_rejects_calibration_window():
    rows = ss.load_sources()
    rules = ss.load_rules()
    bad = {k: dict(v) for k, v in rules.items()}
    bad["survey_rsi2_pullback"]["oos_windows"] = [{"from": "2026-05-01", "to": "2026-07-01"}]
    with pytest.raises(ValueError, match="D-033"):
        ss.check_rules(bad, rows)


def test_check_rules_rejects_wrong_pick():
    rows = ss.load_sources()
    rules = {k: dict(v) for k, v in ss.load_rules().items()}
    rules["survey_orb_breakout"]["oos_pick"] = False
    rules["survey_donchian_breakout"]["oos_pick"] = True
    with pytest.raises(ValueError, match="oos_pick"):
        ss.check_rules(rules, rows)


def test_win_rate_lower_bound():
    assert ss.win_rate_lower(7) == pytest.approx(0.05 ** (1 / 7))
    assert ss.win_rate_lower(5) < ss.win_rate_lower(7) < 1


def test_render_replaces_only_marked_blocks(tmp_path):
    p = tmp_path / "r.md"
    p.write_text("头\n<!-- AUTO:x -->\n旧\n<!-- /AUTO:x -->\n尾\n", encoding="utf-8")
    out = ss.render(p, {"x": "新", "y": "无标记"})
    assert "新" in out and "旧" not in out and out.startswith("头") and out.rstrip().endswith("尾")
    assert "无标记" not in out
