# ruff: noqa: E501  （中文日志较长）
"""Massive flat files 原始文件归档（R-057，负责人 2026-11-14 决定 0p）。只在 VPS 上运行。

当前订阅可下载的全部分钟聚合与日聚合日文件，按日期从早到晚下载（最早的日期最先过期），落到
$ZR_DATA_ROOT/massive_raw/<S3 键>（与 S3 键同路径），边下边算 sha256，追加写
$ZR_DATA_ROOT/massive_raw/MANIFEST_vps.tsv（key、bytes、sha256、下载时间 UTC）。已在清单里且大小一致的跳过（断点续传）；
返回 403 的记为过期并继续。由现有同步任务（deploy/sync_from_vps.sh，rsync 不带 --delete）传到 Mac 外接盘，
Mac 上用 --phase verify-mac 核对文件数与校验和，核对通过的文件再用 --phase prune 从 VPS 删除。

运行（VPS，低优先级，不拖慢逐日处理）：
  cd ~/ZR_Capital && nice -n 10 env ZR_DATA_ROOT=/home/ubuntu/zr_data PYTHONPATH=<代码副本> .venv/bin/python -P -m zr.data.massive_archive --phase download
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import logging
import shutil
import time
from pathlib import Path

from zr.data import massive as mv

log = logging.getLogger("zr.massive_archive")
MANIFEST = "MANIFEST_vps.tsv"
MIN_FREE_GB = 5.0


def raw_root(root: Path) -> Path:
    return root / "massive_raw"


def key_day(key: str) -> str:
    return key.split("/")[-1][:10]


def read_manifest(path: Path) -> dict[str, tuple[int, str]]:
    out: dict[str, tuple[int, str]] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) >= 3 and parts[0] != "key":
                out[parts[0]] = (int(parts[1]), parts[2])
    return out


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def plan(ff: mv.FlatFiles, start: str, prefixes=(mv.MINUTE_PREFIX, mv.DAY_PREFIX)) -> list[dict]:
    """全部 ≥ start 的键，按 (日期, 前缀) 排序：同一天的分钟与日文件相邻。"""
    objs = []
    for pre in prefixes:
        o, _ = ff.list(pre + "/")
        objs += [x for x in o if x["key"].endswith(".csv.gz") and key_day(x["key"]) >= start]
    return sorted(objs, key=lambda x: (key_day(x["key"]), x["key"]))


def download(root: Path, start: str) -> dict:
    ff = mv.FlatFiles()
    rr = raw_root(root)
    rr.mkdir(parents=True, exist_ok=True)
    man_path = rr / MANIFEST
    man = read_manifest(man_path)
    if not man_path.exists():
        man_path.write_text("key\tbytes\tsha256\tdownloaded_utc\n", encoding="utf-8")
    todo = plan(ff, start)
    stats = {
        "planned": len(todo),
        "skipped": 0,
        "downloaded": 0,
        "bytes": 0,
        "expired": [],
        "seconds": 0.0,
    }
    t0 = time.monotonic()
    for i, o in enumerate(todo):
        key = o["key"]
        dest = rr / key
        if key in man and dest.exists() and dest.stat().st_size == man[key][0] == o["size"]:
            stats["skipped"] += 1
            continue
        free = shutil.disk_usage(rr).free / 1e9
        if free < MIN_FREE_GB:
            log.error("磁盘剩余 %.1f GB < %.1f GB，停止归档", free, MIN_FREE_GB)
            stats["stopped_low_disk"] = True
            break
        try:
            n, sha = ff.download_hashed(key, dest)
        except mv.MassiveError as e:
            if e.status == 403:
                stats["expired"].append(key)
                log.warning("已过期（403）：%s", key)
                continue
            raise
        if n != o["size"]:
            raise RuntimeError(f"大小不符：{key} 下载 {n} ≠ 列表 {o['size']}")
        with open(man_path, "a", encoding="utf-8") as f:
            f.write(f"{key}\t{n}\t{sha}\t{dt.datetime.now(dt.UTC).isoformat(timespec='seconds')}\n")
        stats["downloaded"] += 1
        stats["bytes"] += n
        if (i + 1) % 50 == 0:
            el = time.monotonic() - t0
            log.info(
                "%d/%d %s，已下 %.2f GB，%.0f 秒",
                i + 1,
                len(todo),
                key_day(key),
                stats["bytes"] / 1e9,
                el,
            )
    stats["seconds"] = round(time.monotonic() - t0, 1)
    log.info("归档结束：%s", {k: (len(v) if isinstance(v, list) else v) for k, v in stats.items()})
    return stats


def verify_local(root: Path, manifest: Path) -> dict:
    """在 Mac 上：按 VPS 清单逐个核对本地文件的大小与 sha256。返回核对通过的键列表与不符项。"""
    man = read_manifest(manifest)
    ok, bad, missing = [], [], []
    for key, (n, sha) in sorted(man.items()):
        p = raw_root(root) / key
        if not p.exists():
            missing.append(key)
        elif p.stat().st_size != n or sha256_file(p) != sha:
            bad.append(key)
        else:
            ok.append(key)
    return {"manifest": len(man), "ok": ok, "bad": bad, "missing": missing}


def prune(root: Path, verified_list: Path, done_list: Path | None) -> dict:
    """在 VPS 上：删除 Mac 核对通过、且逐日处理已完成（done_list 里有该日）的原始文件。"""
    keys = [k for k in verified_list.read_text(encoding="utf-8").split() if k]
    done = (
        set(done_list.read_text(encoding="utf-8").split())
        if done_list and done_list.exists()
        else None
    )
    n = freed = 0
    for k in keys:
        if done is not None and key_day(k) not in done:
            continue
        p = raw_root(root) / k
        if p.exists():
            freed += p.stat().st_size
            p.unlink()
            n += 1
    return {"deleted": n, "freed_gb": round(freed / 1e9, 2)}


def main(argv: list[str] | None = None) -> int:
    import json

    from zr.config import get_settings

    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["download", "verify-mac", "prune"], required=True)
    ap.add_argument("--start", default="2021-11-15")
    ap.add_argument("--manifest", help="verify-mac：从 VPS 取回的 MANIFEST_vps.tsv")
    ap.add_argument("--verified", help="prune：Mac 核对通过的键列表（每行一个）")
    ap.add_argument("--done", help="prune：逐日处理已完成的日期列表（每行一个）")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = get_settings().zr_data_root
    if a.phase == "download":
        print(json.dumps(download(root, a.start), ensure_ascii=False))
    elif a.phase == "verify-mac":
        r = verify_local(root, Path(a.manifest))
        out = raw_root(root) / "VERIFIED_mac.txt"
        out.write_text("\n".join(r["ok"]) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {k: (len(v) if isinstance(v, list) else v) for k, v in r.items()},
                ensure_ascii=False,
            )
        )
        print("bad:", r["bad"][:20], "missing:", r["missing"][:5], "->", out)
    else:
        print(
            json.dumps(
                prune(root, Path(a.verified), Path(a.done) if a.done else None), ensure_ascii=False
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
