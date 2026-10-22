import numpy as np
import pandas as pd

from zr.research.model_v2 import uniqueness_weights


def test_uniqueness_weights_overlap():
    day = pd.Series(["d1", "d1", "d1", "d2"])
    minute = np.array([10, 10, 100, 10])
    w = uniqueness_weights(day, minute, H=5)
    # 两个完全重叠的事件各 0.5；不重叠的为 1；另一天的不受影响
    assert np.allclose(w, [0.5, 0.5, 1.0, 1.0])
    w2 = uniqueness_weights(pd.Series(["d1", "d1"]), np.array([10, 12]), H=4)
    # 窗口 [11,14] 与 [13,16] 重叠 2 分钟：每个 = (2×1 + 2×0.5)/4 = 0.75
    assert np.allclose(w2, [0.75, 0.75])
