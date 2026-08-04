"""空对照实验：把同一套模型分别喂给真实开奖和确定随机的合成数据。

这是整个项目里最有说服力的一块证据，所以做成常驻命令而不是一次性脚本。

思路很简单。所有"彩票可预测"的主张都建立在一个前提上：模型从历史里
学到的东西是真的。检验它只需要一个对照组——用 numpy 生成一份数学上
保证均匀独立的假开奖，喂给同一个模型。

如果模型在假数据上学出的结构，和在真实开奖上学出的一样，那它在真实
开奖上也什么都没学到。特征重要性的相关系数就是这个"一样"的度量。

固定随机种子，任何人重跑都得到同一组数字。
"""

import numpy as np

from .fetch import data_path
from .markov import load
from .predict import STACK_FEATURES, _walk_features, to_matrix, zone_spec

SEED = 42
# 随机森林超参数与 score_stack 保持一致，否则对照不成立
RF = dict(n_estimators=150, max_depth=8, min_samples_leaf=40,
          n_jobs=-1, random_state=0)


def synthetic(lot, n_periods, zone="front", seed=SEED):
    """生成确定随机的开奖矩阵：每期从号码池里等概率无放回抽取。

    这是"绝对随机"的操作性定义——不是假设某个真实摇奖机随机，
    而是用一个已知随机的过程造出数据，作为比较的基准。
    """
    _, n_max, n_pick = zone_spec(lot, zone)
    rng = np.random.default_rng(seed)
    m = np.zeros((n_periods, n_max), dtype=np.int8)
    for t in range(n_periods):
        m[t, rng.choice(n_max, size=n_pick, replace=False)] = 1
    return m


def profile(m, lot, zone="front"):
    """在给定的开奖矩阵上训练堆叠模型，返回它"学到"了什么。"""
    from sklearn.ensemble import RandomForestClassifier

    T, n = m.shape
    _, _, n_pick = zone_spec(lot, zone)
    min_hist = min(200, T // 2)

    m_ext = np.vstack([m, np.zeros((1, n), dtype=m.dtype)])
    X, y, last = [], [], None
    for t, feat in enumerate(_walk_features(m_ext, n_pick)):
        if min_hist <= t < T:
            X.append(feat)
            y.append(m[t])
        last = feat

    X, y = np.vstack(X), np.concatenate(y)
    clf = RandomForestClassifier(**RF).fit(X, y)

    out = clf.predict_proba(last)[:, 1]
    ins = clf.predict_proba(X)[:, 1]
    return {
        "importance": clf.feature_importances_,
        "base_rate": float(y.mean()),
        "in_lo": float(ins.min()), "in_hi": float(ins.max()),
        "out_lo": float(out.min()), "out_hi": float(out.max()),
        "out_spread": float(out.max() - out.min()),
    }


def compare(lot, zone="front", seed=SEED):
    """真实开奖 vs 合成随机，同一模型跑两遍。"""
    m_real = to_matrix(load(data_path(lot), lot), lot, zone)
    m_fake = synthetic(lot, len(m_real), zone, seed)

    real, fake = profile(m_real, lot, zone), profile(m_fake, lot, zone)
    r = float(np.corrcoef(real["importance"], fake["importance"])[0, 1])
    return {
        "lottery": lot.key, "name": lot.name, "periods": len(m_real),
        "real": real, "fake": fake, "corr": r,
        "max_diff": float(np.abs(real["importance"] - fake["importance"]).max()),
    }


def report(lot, zone="front", seed=SEED):
    """打印对照结果。"""
    c = compare(lot, zone, seed)
    real, fake = c["real"], c["fake"]
    _, n_max, n_pick = zone_spec(lot, zone)

    print(f"\n{c['name']}（前区 {n_pick}/{n_max}，{c['periods']} 期，seed={seed}）")
    print(f"  {'特征':<10}{'真实开奖':>10}{'合成随机':>10}{'差':>9}")
    print("  " + "-" * 39)
    for name, a, b in zip(STACK_FEATURES, real["importance"], fake["importance"]):
        print(f"  {name:<10}{a:>10.3f}{b:>10.3f}{a - b:>+9.3f}")
    print("  " + "-" * 39)

    print(f"  正样本率      {real['base_rate']:>10.4f}{fake['base_rate']:>10.4f}"
          f"   理论 {n_pick / n_max:.4f}")
    print(f"  样本内概率区间 {real['in_lo']:.3f}~{real['in_hi']:.3f}"
          f"    {fake['in_lo']:.3f}~{fake['in_hi']:.3f}")
    print(f"  样本外概率区间 {real['out_lo']:.3f}~{real['out_hi']:.3f}"
          f"    {fake['out_lo']:.3f}~{fake['out_hi']:.3f}")

    print(f"\n  特征重要性相关系数  r = {c['corr']:.4f}   最大差异 {c['max_diff']:.4f}")
    if c["corr"] > 0.9:
        print("  → 模型在真实开奖上学到的结构，与在纯随机数据上学到的几乎一致。")
        print("    它没有从真实开奖里提取到任何随机性之外的信息。")
    elif c["corr"] > 0.7:
        print("  → 两者高度相似，但存在可见差异，值得继续观察。")
    else:
        print("  → 两者出现实质分歧。这可能意味着真实开奖存在可测结构，")
        print("    也可能只是样本波动——需要更多期数和重复种子才能下结论。")
    return c
