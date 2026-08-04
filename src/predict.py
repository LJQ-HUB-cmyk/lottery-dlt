"""预测引擎：多种算法，统一接口，配置驱动。

每个算法接收历史数据和彩种配置，输出每个号码的评分。上层按分数取前 N 个出号。
所有号码范围、选号数量都来自 Lottery 配置，不含任何彩种硬编码。
"""

import numpy as np

from .lotteries import get as get_lottery


def zone_spec(lot, zone):
    """返回该区域的 (列名列表, 号码上限, 选号个数)。"""
    if zone == "front":
        cols = [f"r{i}" for i in range(1, lot.front_pick + 1)]
        return cols, lot.front_max, lot.front_pick
    cols = [f"b{i}" for i in range(1, lot.back_pick + 1)]
    return cols, lot.back_max, lot.back_pick


def to_matrix(df, lot, zone):
    """(期数 × 号码数) 的 0/1 矩阵。"""
    cols, n_max, _ = zone_spec(lot, zone)
    m = np.zeros((len(df), n_max), dtype=np.int8)
    for c in cols:
        m[np.arange(len(df)), df[c].to_numpy() - 1] = 1
    return m


def gaps(matrix):
    """每个号码的当前遗漏值（距离上次出现多少期）。"""
    n_periods, n_nums = matrix.shape
    out = np.empty(n_nums)
    for j in range(n_nums):
        hits = np.flatnonzero(matrix[:, j])
        out[j] = n_periods - hits[-1] - 1 if len(hits) else n_periods
    return out


# ---------- 各算法：返回每个号码的评分 ----------


def score_hot(df, lot, zone, window=100, **kw):
    """热号：近期出现次数越多分越高。"""
    return to_matrix(df, lot, zone)[-window:].sum(axis=0).astype(float)


def score_cold(df, lot, zone, window=100, **kw):
    """冷号：近期出现次数越少分越高。"""
    c = to_matrix(df, lot, zone)[-window:].sum(axis=0).astype(float)
    return c.max() - c


def score_overdue(df, lot, zone, **kw):
    """遗漏值：越久没出分越高。"""
    return gaps(to_matrix(df, lot, zone))


def score_due_cycle(df, lot, zone, **kw):
    """回归周期：遗漏值超过该号码历史平均间隔越多，分越高。"""
    m = to_matrix(df, lot, zone)
    g = gaps(m)
    out = np.empty(m.shape[1])
    for j in range(m.shape[1]):
        hits = np.flatnonzero(m[:, j])
        mean_gap = np.diff(hits).mean() if len(hits) > 1 else len(m)
        out[j] = g[j] / mean_gap if mean_gap > 0 else 0
    return out


def score_markov(df, lot, zone, **kw):
    """马尔可夫转移：用上期实际开出的号码，加权本期各号的转移概率。"""
    m = to_matrix(df, lot, zone)
    n = m.shape[1]

    trans = np.zeros((n, n))
    for t in range(len(m) - 1):
        trans[np.ix_(np.flatnonzero(m[t]), np.flatnonzero(m[t + 1]))] += 1

    row = trans.sum(axis=1, keepdims=True)
    prob = np.divide(trans, row, out=np.zeros_like(trans), where=row > 0)
    return prob[np.flatnonzero(m[-1])].sum(axis=0)


def score_sum_target(df, lot, zone, **kw):
    """和值回归：偏好能把本期和值拉向历史均值的号码。"""
    m = to_matrix(df, lot, zone)
    _, n_max, n_pick = zone_spec(lot, zone)
    nums = np.arange(1, n_max + 1)
    target = (m * nums).sum(axis=1).mean() / n_pick
    return -np.abs(nums - target).astype(float)


def score_ml(df, lot, zone, **kw):
    """机器学习：把"号码 j 下期是否出现"当二分类，随机森林输出概率。"""
    from sklearn.ensemble import RandomForestClassifier

    m = to_matrix(df, lot, zone)
    n_nums = m.shape[1]
    min_hist = min(200, len(m) // 2)

    def feats(hist):
        return np.column_stack([
            gaps(hist),
            hist[-10:].sum(axis=0),
            hist[-30:].sum(axis=0),
            hist[-100:].sum(axis=0),
            np.arange(1, n_nums + 1),
        ])

    X = [feats(m[:t]) for t in range(min_hist, len(m))]
    y = [m[t] for t in range(min_hist, len(m))]

    clf = RandomForestClassifier(
        n_estimators=150, max_depth=8, min_samples_leaf=40, n_jobs=-1, random_state=0
    )
    clf.fit(np.vstack(X), np.concatenate(y))

    p = clf.predict_proba(feats(m))
    return p[:, 1] if p.shape[1] == 2 else np.full(n_nums, p[0, 0])


def _walk_features(m, n_pick):
    """按时间前进，逐期产出「用 m[:t] 能算出的全部特征」。

    天真做法是在每个时点调用一遍各算法，但 score_markov 每次都要重建
    n×n 转移矩阵，总复杂度 O(T²·k²)，2905 期根本跑不完。这里把所有统计量
    改成增量维护，每步降到 O(n)。

    产出的列与 STACK_FEATURES 一一对应。
    """
    T, n = m.shape
    nums = np.arange(1, n + 1, dtype=float)

    # 前缀和：任意窗口的出现次数都能 O(1) 取到
    cum = np.vstack([np.zeros((1, n), dtype=np.int32),
                     np.cumsum(m, axis=0, dtype=np.int32)])

    last_seen = np.full(n, -1)
    first_seen = np.full(n, -1)
    n_hits = np.zeros(n)
    trans = np.zeros((n, n))
    row = np.zeros(n)
    sum_total = 0.0

    for t in range(T):
        gap = np.where(last_seen >= 0, t - 1 - last_seen, t).astype(float)

        # 平均间隔 = (末次 - 首次) / (出现次数 - 1)，与 score_due_cycle 同义
        span = (last_seen - first_seen).astype(float)
        mean_gap = np.where(n_hits > 1, span / np.maximum(n_hits - 1, 1),
                            float(max(t, 1)))
        cycle = gap / np.maximum(mean_gap, 1e-9)

        # 马尔可夫：只取上期开出号码那几行，O(k·n)
        mk = np.zeros(n)
        if t >= 1:
            for j in np.flatnonzero(m[t - 1]):
                if row[j] > 0:
                    mk += trans[j] / row[j]

        target = (sum_total / t / n_pick) if t else 0.0

        yield np.column_stack([
            gap,
            (cum[t] - cum[max(0, t - 10)]).astype(float),
            (cum[t] - cum[max(0, t - 30)]).astype(float),
            (cum[t] - cum[max(0, t - 100)]).astype(float),
            cycle,
            mk,
            -np.abs(nums - target),
            nums,
        ])

        # 用 m[t] 更新状态——必须在 yield 之后，否则特征里混入当期答案
        idx = np.flatnonzero(m[t])
        if t >= 1:
            prev = np.flatnonzero(m[t - 1])
            trans[np.ix_(prev, idx)] += 1
            row[prev] += len(idx)
        new = idx[first_seen[idx] < 0]
        first_seen[new] = t
        last_seen[idx] = t
        n_hits[idx] += 1
        sum_total += nums[idx].sum()


STACK_FEATURES = ["遗漏值", "近10期", "近30期", "近100期",
                  "回归周期", "马尔可夫", "和值偏离", "号码"]


def score_stack(df, lot, zone, **kw):
    """堆叠模型：把其它算法的评分当特征，让随机森林自己学怎么组合。

    混合模型的权重（MIX_WEIGHTS）是手工拍的，没有依据。这里换成让模型
    从历史里学——这是"你们模型太简陋"这个反驳的正面回应。

    注意冷号没有进特征：score_cold = max - score_hot，是热号的单调递减
    变换，而决策树对单调变换不变，加进去提供的信息严格为零。
    """
    from sklearn.ensemble import RandomForestClassifier

    m = to_matrix(df, lot, zone)
    T, n = m.shape
    _, _, n_pick = zone_spec(lot, zone)
    min_hist = min(200, T // 2)

    # 多走一步（补一行空数据）就能在同一次遍历里拿到"下期"的特征：
    # t=T 时用的历史正好是完整的 m，那一行空数据只参与它之后的状态更新
    m_ext = np.vstack([m, np.zeros((1, n), dtype=m.dtype)])

    X, y, last = [], [], None
    for t, feat in enumerate(_walk_features(m_ext, n_pick)):
        if min_hist <= t < T:
            X.append(feat)
            y.append(m[t])
        last = feat
    if not X:
        return np.full(n, 0.5)

    clf = RandomForestClassifier(
        n_estimators=150, max_depth=8, min_samples_leaf=40,
        n_jobs=-1, random_state=0,
    )
    clf.fit(np.vstack(X), np.concatenate(y))

    p = clf.predict_proba(last)
    return p[:, 1] if p.shape[1] == 2 else np.full(n, p[0, 0])


def score_random(df, lot, zone, seed=None, **kw):
    """纯随机：对照基线。"""
    _, n_max, _ = zone_spec(lot, zone)
    return np.random.default_rng(seed).random(n_max)


ALGORITHMS = {
    "hot": ("热号统计", score_hot),
    "cold": ("冷号统计", score_cold),
    "overdue": ("遗漏值", score_overdue),
    "cycle": ("回归周期", score_due_cycle),
    "markov": ("马尔可夫转移", score_markov),
    "sum": ("和值回归", score_sum_target),
    "ml": ("机器学习", score_ml),
    "stack": ("堆叠模型", score_stack),
    "random": ("随机", score_random),
    "mix": ("混合模型", None),
}

MIX_WEIGHTS = {"hot": 1, "overdue": 1, "markov": 1, "cycle": 1, "ml": 2}


def _normalize(s):
    s = np.asarray(s, dtype=float)
    lo, hi = s.min(), s.max()
    return (s - lo) / (hi - lo) if hi > lo else np.full_like(s, 0.5)


def score_mix(df, lot, zone, weights=None, **kw):
    """混合模型：多个算法归一化后加权求和。"""
    weights = weights or MIX_WEIGHTS
    _, n_max, _ = zone_spec(lot, zone)
    total = np.zeros(n_max)
    for name, w in weights.items():
        if w and ALGORITHMS.get(name, (None, None))[1]:
            total += w * _normalize(ALGORITHMS[name][1](df, lot, zone, **kw))
    return total


def _score(df, lot, zone, algo, **kw):
    fn = score_mix if algo == "mix" else ALGORITHMS[algo][1]
    return fn(df, lot, zone, **kw)


def predict(df, lot=None, algo="mix", seed=None, **kw):
    """出号。返回前区/后区号码与各号评分。"""
    lot = lot or get_lottery()

    sf = _score(df, lot, "front", algo, seed=seed, **kw)
    front = sorted((np.argsort(-sf)[: lot.front_pick] + 1).tolist())

    if lot.back_pick:
        sb = _score(df, lot, "back", algo, seed=seed, **kw)
        back = sorted((np.argsort(-sb)[: lot.back_pick] + 1).tolist())
    else:
        sb, back = np.array([]), []

    return {
        "lottery": lot.key,
        "algorithm": algo,
        "name": ALGORITHMS[algo][0],
        "front": front,
        "back": back,
        "front_scores": _normalize(sf).tolist(),
        "back_scores": _normalize(sb).tolist() if lot.back_pick else [],
    }


def explain(df, lot=None, zone="front", algo="mix", **kw):
    """把出号理由摊开：每个号码的原始指标、各算法评分、综合分与排名。"""
    lot = lot or get_lottery()
    m = to_matrix(df, lot, zone)
    _, n_nums, n_pick = zone_spec(lot, zone)
    g = gaps(m)

    comps = {
        name: _normalize(ALGORITHMS[name][1](df, lot, zone, **kw))
        for name in ("hot", "cold", "overdue", "cycle", "markov", "sum", "ml")
    }

    if algo == "mix":
        final = sum(w * comps[n] for n, w in MIX_WEIGHTS.items())
        used = MIX_WEIGHTS
    else:
        final = comps.get(algo)
        if final is None:
            final = _normalize(ALGORITHMS[algo][1](df, lot, zone, **kw))
        used = {algo: 1}

    order = np.argsort(-final)
    rank = np.empty(n_nums, dtype=int)
    rank[order] = np.arange(1, n_nums + 1)

    rows = []
    for j in range(n_nums):
        hits = np.flatnonzero(m[:, j])
        mean_gap = float(np.diff(hits).mean()) if len(hits) > 1 else float(len(m))
        rows.append({
            "number": j + 1,
            "gap": int(g[j]),
            "mean_gap": round(mean_gap, 1),
            "gap_ratio": round(g[j] / mean_gap, 2) if mean_gap else 0.0,
            "freq_10": int(m[-10:, j].sum()),
            "freq_30": int(m[-30:, j].sum()),
            "freq_100": int(m[-100:, j].sum()),
            "freq_all": int(m[:, j].sum()),
            "contributions": {k: round(float(w * comps[k][j]), 3)
                              for k, w in used.items()},
            "score": round(float(final[j]), 3),
            "rank": int(rank[j]),
            "selected": bool(rank[j] <= n_pick),
        })

    return {
        "lottery": lot.key,
        "zone": zone,
        "algorithm": algo,
        "weights": used,
        "expected_freq": round(len(m) * n_pick / n_nums, 1),
        "rows": rows,
    }


def predict_multi(df, lot=None, algo="mix", n_sets=5, seed=None, **kw):
    """出多注：评分最高的一注 + 按评分加权随机的若干注。"""
    lot = lot or get_lottery()
    base = predict(df, lot, algo, seed=seed, **kw)
    rng = np.random.default_rng(seed)

    def weighted(scores, n_max, n_pick):
        s = np.array(scores) if len(scores) else np.ones(n_max)
        p = (s + 0.1) / (s + 0.1).sum()
        return sorted(rng.choice(np.arange(1, n_max + 1), n_pick,
                                 replace=False, p=p).tolist())

    sets = [(base["front"], base["back"])]
    guard = 0
    while len(sets) < n_sets and guard < n_sets * 50:
        guard += 1
        f = weighted(base["front_scores"], lot.front_max, lot.front_pick)
        b = (weighted(base["back_scores"], lot.back_max, lot.back_pick)
             if lot.back_pick else [])
        if (f, b) not in sets:
            sets.append((f, b))

    base["sets"] = sets
    return base
