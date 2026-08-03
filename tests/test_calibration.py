"""校准检验本身——在用它下任何结论之前，先证明它是可信的工具。

两件事必须验证：

1. 第一类错误率：喂给它确定独立的数据，它误报"有记忆"的比例应该 ≈ α。
   报太多说明检验失控，结论会全是假阳性。

2. 统计功效：喂给它确定有记忆的数据，它必须能检测出来。
   如果检测不到，那"未检测到记忆"这句话就毫无信息量——
   可能只是检验太弱，而不是数据真的无记忆。

没有第 2 项，src/markov.py 的输出不能作为任何结论的依据。
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.markov import g_test  # noqa: E402

FRONT_MAX = 35
N_PERIODS = 2903  # 与真实数据等长，功效结论才可迁移
SUM_EDGES = [70, 82, 94, 106]
NUMBERS = np.arange(1, FRONT_MAX + 1)


def sum_state(draws):
    """(n,5) 号码矩阵 -> 和值分箱状态，与 markov.state_sum 一致。"""
    return np.searchsorted(SUM_EDGES, draws.sum(axis=1))


def _weighted_top5(weights, rng):
    """Gumbel top-k：等价于按 weights 做不放回加权抽样。"""
    g = rng.gumbel(size=FRONT_MAX)
    keys = np.log(weights) + g
    return NUMBERS[np.argpartition(-keys, 5)[:5]]


def simulate_independent(n, rng):
    """确定无记忆：每期均匀不放回抽 5 个。"""
    g = rng.gumbel(size=(n, FRONT_MAX))
    return NUMBERS[np.argpartition(-g, 5, axis=1)[:, :5]]


def simulate_with_memory(n, rng, eps):
    """确定有记忆：上期和值偏高时，下期号码权重偏向大号（正自相关）。

    eps=0 退化为独立；eps 越大记忆越强。
    """
    draws = np.empty((n, 5), dtype=int)
    draws[0] = _weighted_top5(np.ones(FRONT_MAX), rng)

    tilt = (NUMBERS - 18) / 17.0
    for t in range(1, n):
        direction = 1.0 if draws[t - 1].sum() > 90 else -1.0
        draws[t] = _weighted_top5(1.0 + eps * direction * tilt, rng)
    return draws


def reject_rate(sim_fn, n_sims, alpha, seed0, **kw):
    rejects, p_values = 0, []
    for i in range(n_sims):
        rng = np.random.default_rng(seed0 + i)
        states = sum_state(sim_fn(N_PERIODS, rng, **kw))
        _, _, p, _, _ = g_test(states, len(SUM_EDGES) + 1)
        p_values.append(p)
        rejects += p < alpha
    return rejects / n_sims, np.array(p_values)


def main():
    alpha = 0.05

    print(f"每次模拟 {N_PERIODS} 期（与真实数据等长），状态=前区和值 5 态\n")

    print("【1】第一类错误率 —— 喂独立数据，看误报多少")
    n_sims = 200
    rate, p_values = reject_rate(simulate_independent, n_sims, alpha, seed0=1000)
    # p 值在 H₀ 下应服从均匀分布
    from scipy import stats

    ks_p = stats.kstest(p_values, "uniform").pvalue
    print(f"  {n_sims} 次模拟，误报率 = {rate:.3f}（期望 ≈ {alpha}）")
    print(f"  p 值均匀性 KS 检验：p = {ks_p:.3f}"
          f"（>0.05 表示 p 值分布正常）")
    calib_ok = abs(rate - alpha) < 0.04 and ks_p > 0.05
    print(f"  {'通过' if calib_ok else '异常'}：检验{'没有' if calib_ok else ''}失控\n")

    print("【2】统计功效 —— 喂有记忆的数据，看能否检出")
    n_sims = 100
    print(f"  {'记忆强度 eps':<14}{'检出率':>10}")
    print("  " + "-" * 24)
    power = {}
    for eps in (0.05, 0.10, 0.20, 0.30):
        rate, _ = reject_rate(
            simulate_with_memory, n_sims, alpha, seed0=2000, eps=eps
        )
        power[eps] = rate
        print(f"  {eps:<14.2f}{rate:>10.2f}")

    print()
    detectable = [e for e, r in power.items() if r >= 0.80]
    if detectable:
        print(f"  通过：记忆强度 ≥ {min(detectable):.2f} 时检出率达 80% 以上")
        print("  → 该检验确实具备发现一阶记忆的能力")
    else:
        print("  警告：即使最强记忆也检不出，此检验不可用于下结论")

    print("\n" + "=" * 56)
    if calib_ok and detectable:
        print("结论：检验工具可信。")
        print(f"它能稳定检出 eps≥{min(detectable):.2f} 量级的记忆效应，")
        print("因此在真实数据上「未检测到记忆」是有信息量的负结果，")
        print("而不是工具失灵。")
        return 0
    print("结论：检验工具存在问题，真实数据上的结论暂不可采信。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
