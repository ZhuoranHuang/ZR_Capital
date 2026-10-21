"""把长桥 SDK 返回的 pyo3 对象转成纯 Python 结构，供 JSON / Parquet 落盘。"""

import datetime as dt
import json
from decimal import Decimal
from enum import Enum
from typing import Any


def _enum_name(obj: Any) -> str:
    # longport 枚举是 pyo3 类，repr 形如 "OrderSide.Buy"；标准 Enum 走 .name
    if isinstance(obj, Enum):
        return obj.name
    return repr(obj).split(".")[-1]


def _is_sdk_enum(obj: Any) -> bool:
    t = type(obj)
    return t.__module__.startswith("longport") and "." in repr(obj) and not hasattr(obj, "__dict__")


MAX_DEPTH = 12


def _public_attrs(obj: Any) -> dict[str, Any]:
    out = {}
    for name in dir(obj):
        if name.startswith("_"):
            continue
        try:
            val = getattr(obj, name)
        except Exception:  # noqa: BLE001 - 属性读取失败只跳过
            continue
        if callable(val):
            continue
        out[name] = val
    return out


def to_py(obj: Any, *, datetime_as_str: bool = False, _depth: int = 0) -> Any:
    """递归转换：Decimal→float，枚举→名称，SDK 对象→dict，list/tuple→list。

    pyo3 枚举的成员会作为同类型属性出现在 dir() 里（Market.US 上有 Market.HK …），
    若按数据对象递归会无限循环；凡属性里出现同类型实例即视为枚举，取名称。
    """
    if _depth > MAX_DEPTH:
        return repr(obj)
    kw = {"datetime_as_str": datetime_as_str, "_depth": _depth + 1}
    if obj is None or isinstance(obj, bool | int | float | str):
        return obj
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, Enum):
        return obj.name
    if isinstance(obj, dt.datetime | dt.date | dt.time):
        return obj.isoformat() if datetime_as_str else obj
    if isinstance(obj, list | tuple):
        return [to_py(x, **kw) for x in obj]
    if isinstance(obj, dict):
        return {str(k): to_py(v, **kw) for k, v in obj.items()}
    if _is_sdk_enum(obj):
        return _enum_name(obj)
    attrs = _public_attrs(obj)
    if any(isinstance(v, type(obj)) for v in attrs.values()):
        return _enum_name(obj)
    fields = {name: to_py(val, **kw) for name, val in attrs.items()}
    return fields if fields else repr(obj)


def to_json(obj: Any) -> str:
    return json.dumps(to_py(obj, datetime_as_str=True), ensure_ascii=False, sort_keys=True)
