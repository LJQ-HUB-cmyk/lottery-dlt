"""显著性判定本身——在用它宣布任何算法"有效/无效"之前，先证明它可信。

这套系统唯一的存在理由是不说谎。而它最容易说谎的地方，就是把一个纯属
运气的好成绩盖章成"显著优于随机"。四件事必须验证：

1. 零分布是精确的。"随便选号该命中多少"是组合学事实，不是估计值。
   若它算错，所有 p 值都错。

2. p 值校准。喂给它确定随机的预测，误报"显著"的比例必须 ≤ α。
   离散统计量上偏保守是正常且正确的方向——宁可漏报，不可谎报。

3. 多重比较受控。同时检验 9 个算法而不校正，族错误率远超名义值。
   这是最隐蔽的一种说谎：每个单项看起来都合规。

4. 堆叠模型的增量特征与原算法逐位一致。_walk_features 为了性能重写了
   遗漏、周期、转移矩阵的算法，一旦与原实现产生分歧，堆叠模型学的就
   不再是"其它算法的输出"，整个对照失去意义。
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import lotteries  # noqa: E402
from src.predict import (_walk_features, score_due_cycle, score_hot,  # noqa: E402
                         score_markov, score_overdue, score_sum_target,
                         to_matrix, zone_spec)
from src.track import _exact_p, _holm, _null_pmf  # noqa: E402

ALPHA = 0.05


def synthetic_df(lot, n, seed):
    """确定随机的开奖记录：每期等概率无放回抽号。"""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        f = np.sort(rng.choice(lot.front_max, lot.front_pick, replace=False) + 1)
        b = np.sort(rng.choice(lot.back_max, lot.back_pick, replace=False) + 1)
        rows.append({**{f"r{i + 1}": int(f[i]) for i in range(lot.front_pick)},
                     **{f"b{i + 1}": int(b[i]) for i in range(lot.back_pick)}})
    return pd.DataFrame(rows)


def check_null_distribution():
    """零分布必须与超几何理论值精确相符，不是近似。"""
    print("\n[1] 零分布精确性")
    print(f"  {'彩种':<10}{'卷积期望':>12}{'理论期望':>12}{'偏差':>12}{'概率和':>10}")
    ok = True
    for lot in lotteries.LOTTERIES.values():
        pmf = _null_pmf(lot)
        got = float(np.dot(np.arange(len(pmf)), pmf))
        want = lot.front_pick ** 2 / lot.front_max
        if lot.back_pick:
            want += lot.back_pick ** 2 / lot.back_max
        diff, total = abs(got - want), pmf.sum()
        bad = diff > 1e-12 or abs(total - 1.0) > 1e-12
        ok &= not bad
        print(f"  {lot.name:<10}{got:>12.6f}{want:>12.6f}{diff:>12.2e}"
              f"{total:>10.6f}{'  ✗' if bad else ''}")
    print("  通过：命中数零分布与组合学理论完全一致" if ok else "  ✗ 零分布算错了")
    return ok


def check_pvalue_calibration(n_pred=8, n_sims=20000, seed=7):
    """喂确定随机的预测，p 值不能过度报小。"""
    print(f"\n[2] p 值校准（{n_sims:,} 次随机预测，每次 {n_pred} 注）")
    lot = lotteries.LOTTERIES["dlt"]
    pmf = _null_pmf(lot)
    rng = np.random.default_rng(seed)
    ps = np.array([_exact_p(lot, rng.choice(len(pmf), size=n_pred, p=pmf))
                   for _ in range(n_sims)])

    ok = True
    print(f"  {'名义水平':>10}{'实测误报率':>12}{'判定':>8}")
    for a in (0.05, 0.10, 0.25):
        got = float((ps < a).mean())
        # 只能偏保守，不能偏激进；离散统计量下界放宽
        bad = got > a
        ok &= not bad
        print(f"  {a:>10.2f}{got:>12.4f}{'  ✗ 超标' if bad else '  ✓':>8}")
    print("  通过：误报率不超过名义水平（偏保守是离散统计量的正确方向）"
          if ok else "  ✗ 检验过于激进，会谎报显著")
    return ok


def check_holm_controls_fwer(n_algos=9, n_pred=30, n_sims=2000, seed=2026):
    """9 个算法全是随机的，系统不能蒙出"显著优于随机"。"""
    print(f"\n[3] 多重比较（{n_algos} 个纯随机算法，各 {n_pred} 期，"
          f"模拟 {n_sims:,} 次）")
    lot = lotteries.LOTTERIES["dlt"]
    pmf = _null_pmf(lot)
    expected = float(np.dot(np.arange(len(pmf)), pmf))
    rng = np.random.default_rng(seed)

    raw_fp = adj_fp = 0
    for _ in range(n_sims):
        hits = [rng.choice(len(pmf), size=n_pred, p=pmf) for _ in range(n_algos)]
        raw = [_exact_p(lot, h) for h in hits]
        if any(p < ALPHA and h.mean() > expected for p, h in zip(raw, hits)):
            raw_fp += 1
        if any(p < ALPHA and h.mean() > expected
               for p, h in zip(_holm(raw), hits)):
            adj_fp += 1

    raw_r, adj_r = raw_fp / n_sims, adj_fp / n_sims
    ok = adj_r <= ALPHA
    print(f"  校正前族错误率  {raw_r:>7.1%}   （名义 {ALPHA:.0%}，未校正必然超标）")
    print(f"  Holm 校正后     {adj_r:>7.1%}   目标 ≤ {ALPHA:.0%}"
          f"{'' if ok else '   ✗ 未达标'}")
    print("  通过：多重比较受控，系统不会自己生产假阳性" if ok
          else "  ✗ 族错误率失控，会谎报显著")
    return ok


def check_incremental_features(n_periods=800, seed=3):
    """增量维护的特征必须与各算法原实现逐位相同。"""
    print(f"\n[4] 堆叠特征一致性（{n_periods} 期合成数据）")
    ok = True
    for key in ("dlt", "ssq"):
        lot = lotteries.LOTTERIES[key]
        df = synthetic_df(lot, n_periods, seed)
        for zone in ("front", "back"):
            if zone == "back" and not lot.back_pick:
                continue
            m = to_matrix(df, lot, zone)
            _, _, n_pick = zone_spec(lot, zone)
            pad = np.zeros((1, m.shape[1]), dtype=m.dtype)
            last = None
            for feat in _walk_features(np.vstack([m, pad]), n_pick):
                last = feat

            cases = {
                "遗漏值": (last[:, 0], score_overdue(df, lot, zone)),
                "近100期": (last[:, 3], score_hot(df, lot, zone, window=100)),
                "回归周期": (last[:, 4], score_due_cycle(df, lot, zone)),
                "马尔可夫": (last[:, 5], score_markov(df, lot, zone)),
                "和值偏离": (last[:, 6], score_sum_target(df, lot, zone)),
            }
            worst = max(float(np.abs(a - b).max()) for a, b in cases.values())
            bad = worst > 1e-9
            ok &= not bad
            print(f"  {lot.name:<10}{zone:<6} 五项特征最大偏差 {worst:.2e}"
                  f"{'  ✗' if bad else '  ✓'}")
    print("  通过：增量实现与原算法完全等价" if ok else "  ✗ 增量实现与原算法分歧")
    return ok


def main():
    t0 = time.time()
    print("=" * 56)
    print("显著性判定与堆叠特征的正确性验证")
    print("=" * 56)

    results = [
        check_null_distribution(),
        check_pvalue_calibration(),
        check_holm_controls_fwer(),
        check_incremental_features(),
    ]

    print("\n" + "=" * 56)
    if all(results):
        print(f"全部通过（{time.time() - t0:.0f} 秒）。")
        print("零分布精确、p 值不激进、族错误率受控、增量特征等价——")
        print("因此「未显著优于随机」这个结论是有信息量的，不是工具失灵。")
        return 0
    print(f"有 {results.count(False)} 项未通过——显著性结论暂不可用。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
