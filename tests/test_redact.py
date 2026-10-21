from zr.ops.redact_api_facts import redact

SAMPLE = """# 长桥 API 核验结果

### quote(正股)

```json
[{"last_done": 512.3, "symbol": "SPY.US"}]
```

### account_balance

```json
[{"currency": "USD", "net_assets": 23456.78, "total_cash": 1200.5,
  "cash_infos": [{"available_cash": 1200.5}]}]
```

### today_orders

```json
[{"order_id": "123456789", "quantity": 1, "price": 2.35, "symbol": "SPY261021C500000.US"}]
```
"""


def test_redact_only_touches_sensitive_sections():
    out, mags = redact(SAMPLE)
    assert '"last_done": 512.3' in out  # 行情不脱敏
    assert "23456.78" not in out and "1200.5" not in out
    assert '"order_id": "[REDACTED]"' in out
    assert '"quantity": [REDACTED]' in out and '"price": [REDACTED]' in out
    assert "SPY261021C500000.US" in out  # 合约代码保留
    assert mags == {"[0].net_assets": "10^4", "[0].total_cash": "10^3", "[0].currency": "USD"}
