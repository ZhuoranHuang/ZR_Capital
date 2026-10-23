"""R-043：把 exit_habits 生成的 x4 候选参数补登到 RESEARCH_LOG（运行 v4_exits 之前执行）。"""

from __future__ import annotations

from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
c = {d["id"]: d for d in yaml.safe_load((REPO / "config/v4_exits_x4.yaml").read_text())["x4"]}
tp, n, sl = c["c1_tp"]["tp_pct"], c["c2_adverse"]["n"], c["c3_sl"]["sl_pct"]
note = (
    "x4 补登（运行前，由 `zr/research/exit_habits.py` 从负责人样本算出）："
    f"c1 中间价 ≥ +{tp}% 全卖；c2 正股 2 分钟 K 连续 {n} 根逆向收盘全卖；"
    f"c3 中间价 ≤ {sl}% 全卖；三者任一先到，另加 30 分钟上限与强平。"
)
p = REPO / "docs/RESEARCH_LOG.md"
s = p.read_text(encoding="utf-8")
i = s.index("| R-043 |")
j = s.index("\n", i)
row = s[i:j].replace("| 待运行（x4 规则待补登） | 待运行 |", f"| {note} | 待运行 |")
p.write_text(s[:i] + row + s[j:], encoding="utf-8")
print(note)
