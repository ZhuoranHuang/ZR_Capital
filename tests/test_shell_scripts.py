"""Shell 脚本静态检查：变量后紧跟中文等非 ASCII 字符时 bash 会把它并进变量名
（如 `$rc，` 被解析成变量 `rc，`，在 set -u 下直接报 unbound variable）。必须写成 `${rc}，`。"""

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = sorted([*ROOT.glob("scripts/**/*.sh"), *ROOT.glob("deploy/**/*.sh")])
BAD = re.compile(r"\$[A-Za-z_][A-Za-z0-9_]*[^\x00-\x7f]")


def test_found_scripts():
    assert len(SCRIPTS) >= 5


def test_no_unbraced_var_before_non_ascii():
    offenders = []
    for p in SCRIPTS:
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            if BAD.search(line):
                offenders.append(f"{p.relative_to(ROOT)}:{i}: {line.strip()}")
    assert not offenders, "\n".join(offenders)


def test_bash_syntax():
    for p in SCRIPTS:
        r = subprocess.run(["bash", "-n", str(p)], capture_output=True, text=True)
        assert r.returncode == 0, f"{p}: {r.stderr}"
