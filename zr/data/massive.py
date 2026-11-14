# ruff: noqa: E501  （中文注释较长）
"""Massive（原 Polygon.io）数据接口：flat files（S3 兼容）与 REST reference 接口（R-057）。

密钥只从 .env 读取（MASSIVE_API_KEY、MASSIVE_S3_ACCESS_KEY、MASSIVE_S3_SECRET_KEY，只在 VPS），
本模块任何日志、异常信息、返回值都不含密钥：REST 用 Authorization 头（不放进 URL），S3 用 SigV4 头签名。

端点（2026-11-14 按官方文档写，首次运行时实测核对，结果见 docs/research/R057_MASSIVE_DATA.md；未核对前视为未验证）：
- flat files：https://files.massive.com ，bucket flatfiles，键 us_stocks_sip/minute_aggs_v1/YYYY/MM/YYYY-MM-DD.csv.gz、
  us_stocks_sip/day_aggs_v1/...；区域 us-east-1，路径式寻址。来源：https://massive.com/docs/flat-files/quickstart（未验证）
- REST：https://api.massive.com/v3/reference/tickers、/v3/reference/splits；分页用响应里的 next_url。来源：
  https://massive.com/docs/rest/stocks/tickers/all-tickers（未验证）
"""

from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from pathlib import Path
from typing import Any

log = logging.getLogger("zr.massive")

S3_ENDPOINT = "https://files.massive.com"
S3_BUCKET = "flatfiles"
S3_REGION = "us-east-1"
REST_BASE = "https://api.massive.com"
MINUTE_PREFIX = "us_stocks_sip/minute_aggs_v1"
DAY_PREFIX = "us_stocks_sip/day_aggs_v1"
UA = "zr-research/1.0"


class MassiveError(Exception):
    """不含密钥的错误：HTTP 状态码 + 截断后的响应体。"""

    def __init__(self, status: int | None, body: str, where: str):
        self.status = status
        super().__init__(f"{where}: HTTP {status}: {body[:300]}")


def _keys() -> tuple[str, str, str]:
    from zr.config import get_settings

    s = get_settings()
    vals = []
    for name in ("massive_api_key", "massive_s3_access_key", "massive_s3_secret_key"):
        v = getattr(s, name)
        vals.append(v.get_secret_value() if v is not None else "")
    return vals[0], vals[1], vals[2]


def day_key(day: dt.date, prefix: str = MINUTE_PREFIX) -> str:
    return f"{prefix}/{day:%Y}/{day:%m}/{day:%Y-%m-%d}.csv.gz"


# ---------------- S3 SigV4（标准库实现，不依赖 boto3） ----------------


def _sign(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


def signing_key(secret_key: str, date: str, region: str, service: str = "s3") -> bytes:
    k = _sign(("AWS4" + secret_key).encode(), date)
    k = _sign(k, region)
    k = _sign(k, service)
    return _sign(k, "aws4_request")


def sigv4_headers(
    method: str,
    url: str,
    access_key: str,
    secret_key: str,
    region: str = S3_REGION,
    now: dt.datetime | None = None,
    payload_hash: str = "UNSIGNED-PAYLOAD",
) -> dict[str, str]:
    """AWS Signature V4（service=s3）请求头。纯函数，便于用 AWS 官方示例做测试。"""
    u = urllib.parse.urlsplit(url)
    now = now or dt.datetime.now(dt.UTC)
    amz_date = now.strftime("%Y%m%dT%H%M%SZ")
    date = now.strftime("%Y%m%d")
    path = urllib.parse.quote(u.path or "/", safe="/~")
    q = urllib.parse.parse_qsl(u.query, keep_blank_values=True)
    cq = "&".join(
        f"{urllib.parse.quote(k, safe='~')}={urllib.parse.quote(v, safe='~')}" for k, v in sorted(q)
    )
    host = u.netloc
    headers = {"host": host, "x-amz-content-sha256": payload_hash, "x-amz-date": amz_date}
    signed = ";".join(sorted(headers))
    canon_headers = "".join(f"{k}:{headers[k]}\n" for k in sorted(headers))
    creq = "\n".join([method, path, cq, canon_headers, signed, payload_hash])
    scope = f"{date}/{region}/s3/aws4_request"
    sts = "\n".join(
        ["AWS4-HMAC-SHA256", amz_date, scope, hashlib.sha256(creq.encode()).hexdigest()]
    )
    sig = hmac.new(signing_key(secret_key, date, region), sts.encode(), hashlib.sha256).hexdigest()
    return {
        "x-amz-content-sha256": payload_hash,
        "x-amz-date": amz_date,
        "Authorization": f"AWS4-HMAC-SHA256 Credential={access_key}/{scope}, SignedHeaders={signed}, Signature={sig}",
    }


class FlatFiles:
    """flat files 只读访问：list（ListObjectsV2）、head、download（流式写盘，原子改名）。"""

    def __init__(self, endpoint: str = S3_ENDPOINT, bucket: str = S3_BUCKET, timeout: float = 120):
        _, self._ak, self._sk = _keys()
        if not (self._ak and self._sk):
            raise RuntimeError("缺少 MASSIVE_S3_ACCESS_KEY / MASSIVE_S3_SECRET_KEY（.env）")
        self.endpoint = endpoint.rstrip("/")
        self.bucket = bucket
        self.timeout = timeout

    def _url(self, key: str = "", query: dict[str, str] | None = None) -> str:
        path = f"/{self.bucket}/{urllib.parse.quote(key, safe='/~')}" if key else f"/{self.bucket}"
        qs = urllib.parse.urlencode(sorted((query or {}).items()), quote_via=urllib.parse.quote)
        return f"{self.endpoint}{path}" + (f"?{qs}" if qs else "")

    def _open(self, method: str, url: str, tries: int = 5):
        for a in range(tries):
            req = urllib.request.Request(url, method=method)
            for k, v in sigv4_headers(method, url, self._ak, self._sk).items():
                req.add_header(k, v)
            req.add_header("User-Agent", UA)
            try:
                return urllib.request.urlopen(req, timeout=self.timeout)  # noqa: S310
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", "replace") if method != "HEAD" else ""
                if e.code in (429, 500, 502, 503, 504) and a < tries - 1:
                    time.sleep(2 ** (a + 1))
                    continue
                raise MassiveError(
                    e.code, body, f"S3 {method} {urllib.parse.urlsplit(url).path}"
                ) from None
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                if a < tries - 1:
                    time.sleep(2 ** (a + 1))
                    continue
                raise MassiveError(
                    None, str(e), f"S3 {method} {urllib.parse.urlsplit(url).path}"
                ) from None
        raise AssertionError("unreachable")

    def list(self, prefix: str, delimiter: str | None = None) -> tuple[list[dict], list[str]]:
        """返回 (对象 [{key,size,last_modified}], 公共前缀)。自动翻页。"""
        objs: list[dict] = []
        prefixes: list[str] = []
        token = None
        while True:
            q = {"list-type": "2", "prefix": prefix}
            if delimiter:
                q["delimiter"] = delimiter
            if token:
                q["continuation-token"] = token
            with self._open("GET", self._url("", q)) as r:
                root = ET.fromstring(r.read())
            ns = root.tag.split("}")[0] + "}" if root.tag.startswith("{") else ""
            for c in root.findall(f"{ns}Contents"):
                objs.append(
                    {
                        "key": c.findtext(f"{ns}Key"),
                        "size": int(c.findtext(f"{ns}Size") or 0),
                        "last_modified": c.findtext(f"{ns}LastModified"),
                    }
                )
            prefixes += [p.findtext(f"{ns}Prefix") for p in root.findall(f"{ns}CommonPrefixes")]
            if (root.findtext(f"{ns}IsTruncated") or "false").lower() != "true":
                return objs, prefixes
            token = root.findtext(f"{ns}NextContinuationToken")

    def exists(self, key: str) -> bool:
        try:
            with self._open("HEAD", self._url(key), tries=3):
                return True
        except MassiveError as e:
            if e.status in (403, 404):
                return False
            raise

    def download(self, key: str, dest: Path, chunk: int = 1 << 20) -> int:
        """流式下载到 dest（先写 .part 再改名），返回字节数。"""
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        n = 0
        with self._open("GET", self._url(key)) as r, open(tmp, "wb") as f:
            while True:
                b = r.read(chunk)
                if not b:
                    break
                f.write(b)
                n += len(b)
        tmp.replace(dest)
        return n


# ---------------- REST ----------------


class Rest:
    """REST 只读：Authorization: Bearer 头（密钥不进 URL、不进日志）；next_url 自动翻页。"""

    def __init__(self, base: str = REST_BASE, timeout: float = 60, min_interval: float = 0.0):
        self._key, _, _ = _keys()
        if not self._key:
            raise RuntimeError("缺少 MASSIVE_API_KEY（.env）")
        self.base = base.rstrip("/")
        self.timeout = timeout
        self.min_interval = min_interval
        self._last = 0.0
        self.calls = 0

    def get(self, path_or_url: str, params: dict[str, Any] | None = None, tries: int = 6) -> dict:
        url = path_or_url if path_or_url.startswith("http") else f"{self.base}{path_or_url}"
        if params:
            url += ("&" if "?" in url else "?") + urllib.parse.urlencode(params)
        for a in range(tries):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            self.calls += 1
            req = urllib.request.Request(url)
            req.add_header("Authorization", f"Bearer {self._key}")
            req.add_header("User-Agent", UA)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:  # noqa: S310
                    return json.loads(r.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", "replace")
                if e.code in (429, 500, 502, 503, 504) and a < tries - 1:
                    time.sleep(min(60, 2 ** (a + 1)))
                    continue
                raise MassiveError(e.code, body, f"GET {urllib.parse.urlsplit(url).path}") from None
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                if a < tries - 1:
                    time.sleep(min(60, 2 ** (a + 1)))
                    continue
                raise MassiveError(None, str(e), f"GET {urllib.parse.urlsplit(url).path}") from None
        raise AssertionError("unreachable")

    def paged(self, path: str, params: dict[str, Any]) -> Iterator[dict]:
        body = self.get(path, params)
        while True:
            yield from body.get("results") or []
            nxt = body.get("next_url")
            if not nxt:
                return
            body = self.get(nxt)

    def tickers(self, **params: Any) -> list[dict]:
        p = {"market": "stocks", "limit": 1000, **params}
        return list(self.paged("/v3/reference/tickers", p))

    def splits(self, **params: Any) -> list[dict]:
        p = {"limit": 1000, **params}
        return list(self.paged("/v3/reference/splits", p))
