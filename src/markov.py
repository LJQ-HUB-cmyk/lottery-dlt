"""方案 B：用聚合特征定义状态，检验开奖序列的马尔可夫性。

要回答的问题不是"下期开什么"，而是"这条序列到底有没有记忆"。

H₀：零阶（各期独立，转移矩阵每一行都等于稳态分布）
H₁：一阶马尔可夫（下期状态依赖当期状态）

检验统计量用似然比 G²，在 H₀ 下渐近服从 χ²，自由度 (S-1)²，S 为状态数。
G² = 2 Σ n_ij · ln( n_ij / E_ij )，其中 E_ij = n_i· · n_·j / n

为什么选 G² 而不是 Pearson χ²：稀疏列联表下 G² 的渐近性质通常更稳，
而且它直接是似然比，跟"零阶 vs 一阶"的模型比较对得上。
"""

import re

import numpy as np
import pandas as pd
from scipy import stats

FRONT_COLS = ["r1", "r2", "r3", "r4", "r5"]
BACK_COLS = ["b1", "b2"]
FRONT_MAX = 35

# 每格期望频数的经验下限。低于这个值，卡方近似不可靠。
MIN_EXPECTED = 5.0


def load(path, lot=None):
    """读开奖数据。号码列名由彩种配置决定，未指定时按文件实际列自适应。"""
    df = pd.read_csv(path, dtype={"issue": str})
    if lot is not None:
        cols = ([f"r{i}" for i in range(1, lot.front_pick + 1)]
                + [f"b{i}" for i in range(1, lot.back_pick + 1)])
    else:
        cols = [c for c in df.columns if re.fullmatch(r"[rb]\d+", c)]
    for c in cols:
        df[c] = df[c].astype(int)
    return df.sort_values("issue").reset_index(drop=True)


# ---------- 状态定义 ----------
# 每个函数把一期开奖映射成一个整数状态。状态数必须足够小，
# 使得 (状态数)² 个转移格子在 2900 期数据下仍有足够期望频数。


def state_sum(row, n_bins=5):
    """前区和值分箱。和值范围 15..165，理论均值 90。"""
    total = sum(row[c] for c in FRONT_COLS)
    # 用分位点式的固定边界，让各箱频数大致均衡
    edges = [70, 82, 94, 106]
    return int(np.searchsorted(edges, total))


def state_odd(row):
    """前区奇数个数。0 和 5 太罕见（各约 2%），合并成 4 态避免期望频数塌陷。"""
    k = sum(1 for c in FRONT_COLS if row[c] % 2)
    return min(max(k, 1), 4) - 1


def state_big(row):
    """前区大号（>17）个数，同样合并两端成 4 态。"""
    k = sum(1 for c in FRONT_COLS if row[c] > FRONT_MAX / 2)
    return min(max(k, 1), 4) - 1


def state_zone(row):
    """按区间分布 (1-12 / 13-24 / 25-35) 的最大占用区归类，3 个状态。"""
    counts = [0, 0, 0]
    for c in FRONT_COLS:
        v = row[c]
        counts[0 if v <= 12 else 1 if v <= 24 else 2] += 1
    return int(np.argmax(counts))


def state_back_sum(row):
    """后区和值分箱，3 个状态。后区和值范围 3..23。"""
    total = row["b1"] + row["b2"]
    return int(np.searchsorted([10, 16], total))


STATE_DEFS = {
    "前区和值(5态)": (state_sum, 5),
    "奇数个数(4态)": (state_odd, 4),
    "大号个数(4态)": (state_big, 4),
    "主导区间(3态)": (state_zone, 3),
    "后区和值(3态)": (state_back_sum, 3),
}


def encode(df, fn):
    return np.array([fn(row) for _, row in df.iterrows()], dtype=int)


# ---------- 检验 ----------


def transition_counts(states, n_states):
    """n[i,j] = 从状态 i 转移到状态 j 的次数。"""
    n = np.zeros((n_states, n_states), dtype=float)
    np.add.at(n, (states[:-1], states[1:]), 1)
    return n


def g_test(states, n_states):
    """似然比检验 H₀=零阶 vs H₁=一阶。

    返回 (G², 自由度, p 值, 最小期望频数, 有效状态数)。
    """
    n = transition_counts(states, n_states)

    # 丢掉从未出现的状态，否则自由度会虚高
    used = (n.sum(axis=1) + n.sum(axis=0)) > 0
    n = n[np.ix_(used, used)]
    s = n.shape[0]

    total = n.sum()
    row_sums = n.sum(axis=1, keepdims=True)
    col_sums = n.sum(axis=0, keepdims=True)
    expected = row_sums @ col_sums / total

    mask = n > 0
    g2 = 2 * np.sum(n[mask] * np.log(n[mask] / expected[mask]))
    dof = (s - 1) ** 2
    p = stats.chi2.sf(g2, dof)

    return g2, dof, p, expected.min(), s


def benjamini_hochberg(p_values, alpha=0.05):
    """BH 法控制 FDR，返回每个检验是否被拒绝。"""
    p = np.asarray(p_values)
    order = np.argsort(p)
    m = len(p)
    thresholds = alpha * (np.arange(1, m + 1)) / m

    passed = p[order] <= thresholds
    k = np.max(np.flatnonzero(passed)) + 1 if passed.any() else 0

    rejected = np.zeros(m, dtype=bool)
    if k:
        rejected[order[:k]] = True
    return rejected


def run(df, alpha=0.05):
    """跑完所有状态定义，返回结果列表。"""
    results = []
    for name, (fn, n_states) in STATE_DEFS.items():
        states = encode(df, fn)
        g2, dof, p, min_exp, s = g_test(states, n_states)
        results.append(
            {
                "name": name,
                "g2": g2,
                "dof": dof,
                "p": p,
                "min_expected": min_exp,
                "states_used": s,
            }
        )

    rejected = benjamini_hochberg([r["p"] for r in results], alpha)
    for r, rej in zip(results, rejected):
        r["rejected"] = bool(rej)
    return results


def format_report(results, n_periods, alpha=0.05):
    lines = [
        f"马尔可夫性检验（方案 B：聚合特征状态），数据 {n_periods} 期",
        "",
        "H₀：序列无记忆（各期独立）    H₁：一阶马尔可夫（依赖上期状态）",
        "",
        f"{'状态定义':<16}{'G²':>9}{'自由度':>7}{'p 值':>9}{'最小期望频数':>13}",
        "-" * 56,
    ]

    for r in results:
        lines.append(
            f"{r['name']:<16}{r['g2']:>9.3f}{r['dof']:>7}{r['p']:>9.3f}"
            f"{r['min_expected']:>13.1f}"
        )

    lines += ["-" * 56, ""]

    weak = [r["name"] for r in results if r["min_expected"] < MIN_EXPECTED]
    if weak:
        lines.append(f"⚠ 期望频数偏低（<{MIN_EXPECTED}），卡方近似存疑：{', '.join(weak)}")
        lines.append("")

    n_rej = sum(r["rejected"] for r in results)
    lines.append(f"BH 法 FDR 校正（α={alpha}），{len(results)} 个检验：")
    if n_rej:
        for r in results:
            if r["rejected"]:
                lines.append(f"  ✗ {r['name']} 拒绝 H₀ —— 该维度上检测到记忆效应")
        lines.append("")
        lines.append("  注意：拒绝 H₀ 意味着序列在这个维度上不是无记忆的。")
        lines.append("  在下结论前必须先排除状态定义本身引入的伪相关。")
    else:
        lines.append("  全部无法拒绝 H₀ —— 未检测到任何一阶记忆效应")
        lines.append("")
        lines.append("  这不是失败，是一个真结论：在这些聚合维度上，开奖序列")
        lines.append("  的行为与独立随机过程无法区分。冷热号、遗漏值一类基于")
        lines.append("  '上期影响下期'的理论，在这批数据上没有统计支持。")

    return "\n".join(lines)
