import pytest

from zr.data.universe import DEFAULT_PATH, ChainRule, load_universe


def test_default_universe_loads_and_matches_spec():
    uni = load_universe(DEFAULT_PATH, symbols=["SPY.US", "QQQ.US", "MU.US", "AMD.US"])
    assert uni.subscription_cap == 450
    rules = uni.chain_rules()
    assert rules["SPY.US"] == ChainRule(2, 10)
    assert rules["MU.US"] == ChainRule(2, 8)
    assert uni.underlyings["AMD.US"].strike_band_pct == pytest.approx(0.08)


def test_symbols_filter_is_subset_and_ordered():
    uni = load_universe(DEFAULT_PATH, symbols=["QQQ.US", "SPY.US"])
    assert uni.symbols == ["QQQ.US", "SPY.US"]


def test_unknown_symbol_rejected():
    with pytest.raises(ValueError, match="TSLA.US"):
        load_universe(DEFAULT_PATH, symbols=["SPY.US", "TSLA.US"])


def test_invalid_yaml_values_rejected(tmp_path):
    bad = tmp_path / "u.yaml"
    bad.write_text(
        "subscription_cap: 600\nunderlyings:\n  SPY.US: {n_expiries: 1, strikes_each_side: 1, "
        "strike_band_pct: 0.1, max_dte: 1}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_universe(bad, symbols=["SPY.US"])
