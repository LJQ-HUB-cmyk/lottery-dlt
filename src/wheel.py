"""Lottery wheel = 覆盖设计 C(v, k, t)。

从 v 个号码中选出最少的 k-号组合，使任意 t 个号码都被至少一注同时包含。
彩票语义：圈定 v 个号，若开奖号里有 t 个落在这 v 个里，保证至少一注中这 t 个。

这个问题是 NP-hard，大量参数的精确最优值至今未知。这里提供四种手段：

  schonheim  理论下界（不是构造，是"不可能比这更少"的证明）
  greedy     贪心，快，通常离下界有明显 gap
  cyclic     循环群构造 —— 代数结构压缩搜索空间，常优于贪心
  anneal     模拟退火，在固定注数下尝试补齐覆盖

wheel 不改变期望回报（N 注的期望恒等于 N × 单注期望），它改变的是概率
分布的形状：把"可能颗粒无收"的质量挪到"条件满足即必中"上。
"""

from itertools import combinations
from math import ceil, comb

import numpy as np


def schonheim(v, k, t):
    """Schönheim 下界，递归定义：L(v,k,t) = ⌈v/k · L(v-1,k-1,t-1)⌉"""
    if t == 1:
        return ceil(v / k)
    return ceil(v / k * schonheim(v - 1, k - 1, t - 1))


def _index(v, t):
    """所有 t-子集的编号表，返回 {子集元组: 下标}。"""
    return {s: i for i, s in enumerate(combinations(range(v), t))}


def _coverage_lists(v, k, t):
    """每个 k-子集覆盖哪些 t-子集（下标数组）。"""
    tidx = _index(v, t)
    blocks = list(combinations(range(v), k))
    cover = [np.array([tidx[s] for s in combinations(b, t)], dtype=np.int32)
             for b in blocks]
    return blocks, cover, len(tidx)


def verify(blocks, v, t):
    """暴力验证：所有 C(v,t) 个 t-子集是否都被覆盖。返回未覆盖的个数。"""
    covered = set()
    for b in blocks:
        covered.update(combinations(sorted(b), t))
    return comb(v, t) - len(covered)


def greedy(v, k, t):
    """经典贪心集合覆盖：每次取覆盖最多未覆盖 t-子集的 block。"""
    blocks, cover, n_t = _coverage_lists(v, k, t)
    done = np.zeros(n_t, dtype=bool)
    chosen = []

    while not done.all():
        best_i, best_gain = -1, -1
        for i, c in enumerate(cover):
            gain = int((~done[c]).sum())
            if gain > best_gain:
                best_i, best_gain = i, gain
        done[cover[best_i]] = True
        chosen.append(blocks[best_i])

    return chosen


def _canonical(block, v):
    """循环轨道的标准代表：所有循环移位里字典序最小的那个。"""
    return min(
        tuple(sorted((x + s) % v for x in block)) for s in range(v)
    )


def cyclic(v, k, t):
    """循环群构造：在 Z_v 的移位作用下，按整条轨道来选，而不是单个 block。

    这正是代数结构起作用的地方——搜索空间从 C(v,k) 个 block 降到约
    C(v,k)/v 条轨道，且轨道自带对称性，往往比贪心更接近下界。
    """
    tidx = _index(v, t)

    orbits = {}
    for b in combinations(range(v), k):
        rep = _canonical(b, v)
        if rep not in orbits:
            members = {tuple(sorted((x + s) % v for x in rep)) for s in range(v)}
            cov = set()
            for m in members:
                cov.update(tidx[s] for s in combinations(sorted(m), t))
            orbits[rep] = (sorted(members), np.fromiter(cov, dtype=np.int32))

    done = np.zeros(len(tidx), dtype=bool)
    chosen = []
    for _ in range(len(orbits)):
        if done.all():
            break
        best, best_gain = None, -1
        for rep, (members, cov) in orbits.items():
            gain = int((~done[cov]).sum())
            # 同等收益下优先选轨道小的，避免为覆盖少量子集拖入整条大轨道
            if gain > best_gain or (gain == best_gain and best
                                    and len(members) < len(orbits[best][0])):
                best, best_gain = rep, gain
        if best_gain <= 0:
            break
        members, cov = orbits[best]
        done[cov] = True
        chosen.extend(members)
        del orbits[best]

    # 轨道整条加入会有冗余，逐个尝试剔除
    return _prune(chosen, v, t)


def _prune(blocks, v, t):
    """去掉移除后仍保持完全覆盖的多余 block。"""
    tidx = _index(v, t)
    count = np.zeros(len(tidx), dtype=np.int32)
    covs = []
    for b in blocks:
        c = np.array([tidx[s] for s in combinations(sorted(b), t)], dtype=np.int32)
        covs.append(c)
        count[c] += 1

    keep = [True] * len(blocks)
    # 从后往前删，倾向保留先被选中的（贡献通常更大）
    for i in range(len(blocks) - 1, -1, -1):
        if (count[covs[i]] >= 2).all():
            count[covs[i]] -= 1
            keep[i] = False

    return [b for b, k_ in zip(blocks, keep) if k_]


def anneal(v, k, t, n_blocks, iters=200_000, seed=0, init=None):
    """固定注数 n_blocks，用模拟退火尝试把未覆盖数压到 0。

    返回 (blocks, 未覆盖数)。未覆盖数为 0 表示找到了这个规模的可行解。
    """
    blocks, cover, n_t = _coverage_lists(v, k, t)
    rng = np.random.default_rng(seed)
    n_blocks = min(n_blocks, len(blocks))

    if init is not None:
        lookup = {b: i for i, b in enumerate(blocks)}
        cur = [lookup[tuple(sorted(b))] for b in init][:n_blocks]
        pool = [i for i in range(len(blocks)) if i not in set(cur)]
        cur += list(rng.choice(pool, n_blocks - len(cur), replace=False))
    else:
        cur = list(rng.choice(len(blocks), n_blocks, replace=False))

    count = np.zeros(n_t, dtype=np.int32)
    for i in cur:
        count[cover[i]] += 1
    uncovered = int((count == 0).sum())

    best, best_unc = list(cur), uncovered
    t0, t1 = 2.0, 0.01

    for step in range(iters):
        if best_unc == 0:
            break
        temp = t0 * (t1 / t0) ** (step / iters)

        slot = rng.integers(n_blocks)
        old, new = cur[slot], int(rng.integers(len(blocks)))
        if new in cur:
            continue

        count[cover[old]] -= 1
        count[cover[new]] += 1
        cand = int((count == 0).sum())

        if cand <= uncovered or rng.random() < np.exp((uncovered - cand) / temp):
            cur[slot] = new
            uncovered = cand
            if cand < best_unc:
                best, best_unc = list(cur), cand
        else:
            count[cover[new]] -= 1
            count[cover[old]] += 1

    return [blocks[i] for i in best], best_unc


def exact(v, k, t, time_limit=300.0):
    """整数规划精确求解，用 HiGHS。

    这是集合覆盖的标准 ILP 形式：
        min  Σ xᵢ
        s.t. Σ_{i: block i 覆盖 t-子集 j} xᵢ ≥ 1   (∀ j)
             xᵢ ∈ {0,1}

    与启发式的本质区别：求解器返回 optimal 时，给出的是**被证明的最优值**，
    没有更好的解存在。这能把"构造不够好"和"下界不够紧"这两种 gap 区分开。

    返回 dict：size、blocks、是否证明最优、对偶界。
    """
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import csc_matrix

    blocks, cover, n_t = _coverage_lists(v, k, t)
    n_b = len(blocks)

    rows = np.concatenate(cover)
    cols = np.concatenate([np.full(len(c), i) for i, c in enumerate(cover)])
    A = csc_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_t, n_b))

    res = milp(
        c=np.ones(n_b),
        constraints=LinearConstraint(A, lb=1, ub=np.inf),
        integrality=np.ones(n_b),
        bounds=Bounds(0, 1),
        options={"time_limit": time_limit, "presolve": True},
    )

    if res.x is None:
        return {"size": None, "blocks": [], "proven": False,
                "status": res.message, "dual_bound": None}

    chosen = [blocks[i] for i in np.flatnonzero(res.x > 0.5)]
    return {
        "size": len(chosen),
        "blocks": chosen,
        "proven": res.status == 0,
        "status": res.message,
        # 求解器给出的下界：真实最优不可能低于它
        "dual_bound": ceil(res.mip_dual_bound - 1e-6)
        if res.mip_dual_bound is not None else None,
    }


def solve(v, k, t, anneal_iters=400_000, seeds=4, exact_time=0.0):
    """综合求解：启发式打底，可选 ILP 精确求解收尾。

    每个规模用多个随机种子重试——退火会卡在局部极小，换种子常能突破。
    exact_time > 0 时再跑 ILP：它可能给出更好的解，更重要的是能**证明**
    当前解已是最优，从而把"构造不够好"和"Schönheim 下界不紧"区分开。

    返回 dict：下界、各方法结果、最终构造、是否证明最优。
    """
    lb = schonheim(v, k, t)
    g = greedy(v, k, t)
    c = cyclic(v, k, t)

    best = min([g, c], key=len)
    method = "greedy" if len(g) <= len(c) else "cyclic"

    size = len(best) - 1
    while size >= lb:
        found = None
        for seed in range(seeds):
            cand, unc = anneal(v, k, t, size, iters=anneal_iters, seed=seed, init=best)
            if unc == 0:
                found = cand
                break
        if found is None:
            break
        best, method, size = found, f"anneal({size})", size - 1

    out = {
        "lower_bound": lb,
        "greedy": len(g),
        "cyclic": len(c),
        "best": best,
        "size": len(best),
        "method": method,
        "uncovered": verify(best, v, t),
        "proven": False,
        "dual_bound": None,
    }

    if exact_time:
        e = exact(v, k, t, time_limit=exact_time)
        if e["size"] is not None:
            out["dual_bound"] = e["dual_bound"]
            out["proven"] = e["proven"]
            if e["size"] < out["size"]:
                out.update(best=e["blocks"], size=e["size"], method="ILP",
                           uncovered=verify(e["blocks"], v, t))
            elif e["proven"] and e["size"] == out["size"]:
                out["method"] += " (ILP 证明最优)"

    return out
