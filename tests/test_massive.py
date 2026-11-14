# ruff: noqa: E501
"""zr.data.massive：SigV4 签名用 AWS 官方文档示例做已知答案测试（不联网）。"""

import datetime as dt

from zr.data import massive as m

AK = "AKIAIOSFODNN7EXAMPLE"
SK = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
EMPTY = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_signing_key_aws_example():
    # https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_sigv-create-signed-request.html 派生密钥示例
    k = m.signing_key("wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY", "20120215", "us-east-1", "iam")
    assert k.hex() == "f4780e2d9f65fa895f9c67b32ce1baf0b0d8a43505a000a1a9e090d414db404d"


def test_sigv4_list_objects_aws_example():
    # AWS S3 文档「Signature Calculations for the Authorization Header」GET Bucket (List Objects) 示例
    h = m.sigv4_headers(
        "GET",
        "https://examplebucket.s3.amazonaws.com/?max-keys=2&prefix=J",
        AK,
        SK,
        now=dt.datetime(2013, 5, 24, tzinfo=dt.UTC),
        payload_hash=EMPTY,
    )
    assert h["Authorization"].endswith(
        "Signature=34b48302e7b5fa45bde8084f4b7868a86f0a534bc59db6670ed5711ef69dc6f7"
    )
    assert "SignedHeaders=host;x-amz-content-sha256;x-amz-date" in h["Authorization"]


def test_day_key():
    assert (
        m.day_key(dt.date(2021, 7, 19)) == "us_stocks_sip/minute_aggs_v1/2021/07/2021-07-19.csv.gz"
    )
    assert m.day_key(dt.date(2021, 7, 19), m.DAY_PREFIX).startswith(
        "us_stocks_sip/day_aggs_v1/2021/07/"
    )


def test_errors_carry_no_secret():
    e = m.MassiveError(403, "x" * 1000, "S3 GET /flatfiles/k")
    assert len(str(e)) < 400
