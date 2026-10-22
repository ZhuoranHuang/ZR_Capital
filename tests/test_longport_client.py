"""实盘 / 模拟盘 token 分离：字段读取、兼容别名、JWT ac 字段校验。

全部用合成 token，不碰真实密钥。
"""

from __future__ import annotations

import base64
import json
import warnings

import pytest

from zr.broker.longport_client import (
    LongportModeMismatchError,
    MissingLongportKeysError,
    _jwt_claim,
    make_config,
)
from zr.config import Settings


def _jwt(payload: dict) -> str:
    """构造一个结构合法但签名随意的 JWT（测试用，不是真实 token）。"""
    seg = lambda d: base64.urlsafe_b64encode(json.dumps(d).encode()).rstrip(b"=").decode()  # noqa: E731
    return f"{seg({'alg': 'none'})}.{seg(payload)}.fakesig"


LIVE_JWT = _jwt({"ac": "lb_brokeraccount", "sub": "x"})
PAPER_JWT = _jwt({"ac": "lb_papertrading", "sub": "x"})


def _settings(**kw) -> Settings:
    base = {"longport_app_key": "k", "longport_app_secret": "s"}
    return Settings(_env_file=None, **{**base, **kw})


def test_jwt_claim_reads_payload_without_verifying_signature():
    assert _jwt_claim(LIVE_JWT, "ac") == "lb_brokeraccount"
    assert _jwt_claim(PAPER_JWT, "ac") == "lb_papertrading"


def test_jwt_claim_returns_none_for_malformed_token():
    assert _jwt_claim("not-a-jwt", "ac") is None
    assert _jwt_claim("a.b", "ac") is None  # 只有 2 段
    assert _jwt_claim("a.b.c", "ac") is None  # payload 不是合法 base64/json
    assert _jwt_claim(_jwt({"sub": "x"}), "ac") is None  # 没有 ac 字段


def test_access_token_for_paper_uses_paper_field_only():
    s = _settings(longport_paper_access_token=PAPER_JWT)
    assert s.access_token_for("paper").get_secret_value() == PAPER_JWT
    assert s.access_token_for("live") is None


def test_live_access_token_falls_back_to_legacy_field_name():
    s = _settings(longport_access_token=LIVE_JWT)
    assert s.access_token_for("live").get_secret_value() == LIVE_JWT
    assert s.using_legacy_access_token_name is True


def test_new_live_field_name_takes_precedence_and_is_not_legacy():
    s = _settings(longport_live_access_token=LIVE_JWT, longport_access_token="old-ignored")
    assert s.access_token_for("live").get_secret_value() == LIVE_JWT
    assert s.using_legacy_access_token_name is False


def test_access_token_for_rejects_bad_mode():
    s = _settings()
    with pytest.raises(ValueError):
        s.access_token_for("prod")


def test_has_longport_keys_per_mode():
    s = _settings(longport_live_access_token=LIVE_JWT)
    assert s.has_longport_keys("live") is True
    assert s.has_longport_keys("paper") is False
    s2 = _settings(longport_paper_access_token=PAPER_JWT)
    assert s2.has_longport_keys("paper") is True
    assert s2.has_longport_keys("live") is False


def test_make_config_rejects_missing_keys_with_mode_specific_message():
    s = _settings()
    with pytest.raises(MissingLongportKeysError, match="paper"):
        make_config("paper", s)
    with pytest.raises(MissingLongportKeysError, match="live"):
        make_config("live", s)


def test_make_config_rejects_invalid_mode():
    with pytest.raises(ValueError):
        make_config("prod", _settings())  # type: ignore[arg-type]


def test_make_config_paper_mode_requires_papertrading_ac():
    s = _settings(longport_paper_access_token=LIVE_JWT)  # 用了实盘 token 当 paper
    with pytest.raises(LongportModeMismatchError, match="paper"):
        make_config("paper", s)


def test_make_config_live_mode_rejects_papertrading_ac():
    s = _settings(longport_live_access_token=PAPER_JWT)  # 用了模拟盘 token 当 live
    with pytest.raises(LongportModeMismatchError, match="live"):
        make_config("live", s)


def test_make_config_warns_on_legacy_field_name(monkeypatch):
    s = _settings(longport_access_token=LIVE_JWT)
    called = {}

    class _FakeConfig:
        @staticmethod
        def from_apikey(app_key, app_secret, token):
            called["ok"] = (app_key, app_secret, token)
            return "config"

    monkeypatch.setattr("zr.broker.longport_client.Config", _FakeConfig)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        make_config("live", s)
    assert any("LONGPORT_LIVE_ACCESS_TOKEN" in str(x.message) for x in w)
    assert called["ok"] == ("k", "s", LIVE_JWT)


def test_make_config_no_warning_with_new_field_name(monkeypatch):
    s = _settings(longport_live_access_token=LIVE_JWT)

    class _FakeConfig:
        @staticmethod
        def from_apikey(app_key, app_secret, token):
            return "config"

    monkeypatch.setattr("zr.broker.longport_client.Config", _FakeConfig)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        make_config("live", s)
    assert not any("LONGPORT_LIVE_ACCESS_TOKEN" in str(x.message) for x in w)


def test_make_config_defaults_to_live_mode(monkeypatch):
    s = _settings(longport_live_access_token=LIVE_JWT)
    seen = {}

    class _FakeConfig:
        @staticmethod
        def from_apikey(app_key, app_secret, token):
            seen["token"] = token
            return "config"

    monkeypatch.setattr("zr.broker.longport_client.Config", _FakeConfig)
    make_config(settings=s)  # 不传 mode
    assert seen["token"] == LIVE_JWT
