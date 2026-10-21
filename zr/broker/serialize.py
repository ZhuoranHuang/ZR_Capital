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


def to_py(obj: Any, *, datetime_as_str: bool = False) -> Any:
    """递归转换：Decimal→float，枚举→名称，SDK 对象→dict，list/tuple→list。"""
    if obj is None or isinstance(obj, bool | int | float | str):
        return obj
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, Enum):
        return obj.name
    if isinstance(obj, dt.datetime | dt.date | dt.time):
        return obj.isoformat() if datetime_as_str else obj
    if isinstance(obj, list | tuple):
        return [to_py(x, datetime_as_str=datetime_as_str) for x in obj]
    if isinstance(obj, dict):
        return {str(k): to_py(v, datetime_as_str=datetime_as_str) for k, v in obj.items()}
    if _is_sdk_enum(obj):
        return _enum_name(obj)
    # 一般 SDK 数据对象：取非下划线、非可调用属性
    fields = {}
    for name in dir(obj):
        if name.startswith("_"):
            continue
        try:
            val = getattr(obj, name)
        except Exception:  # noqa: BLE001 - 属性读取失败只跳过
            continue
        if callable(val):
            continue
        fields[name] = to_py(val, datetime_as_str=datetime_as_str)
    return fields if fields else repr(obj)


def to_json(obj: Any) -> str:
    return json.dumps(to_py(obj, datetime_as_str=True), ensure_ascii=False, sort_keys=True)
