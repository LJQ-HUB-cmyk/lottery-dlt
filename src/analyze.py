"""对一套 wheel 注单做实际回报分析。

回答"这个保证值多少钱"：保证条件多久触发一次、平均能拿回多少、
以及最关键的——它跟花同样的钱随便买相比，差别在哪。

因为马尔可夫检验已证明序列无可利用的记忆，选哪 v 个号在统计上完全等价，
所以用号码池 1..v 模拟的结果对任何号码池都成立。
"""

from itertools import combinations
from math import comb

import numpy as np

from .prize import LEVEL_AMOUNT, LEVEL_NAMES, PRIZE_TABLE, TICKET_PRICE

FRONT_MAX = 35
BACK_MAX = 12


def build_tickets(front_blocks, back_pool_size):
    """wheel 的前区 block × 后区全组合 = 完整注单。"""
    backs = list(combinations(range(back_pool_size), 2))
    return [(list(b), list(bk)) for b in front_blocks for bk in backs]


def _masks(tickets):
    f = np.zeros((len(tickets), FRONT_MAX), dtype=np.int8)
    b = np.zeros((len(tickets), BACK_MAX), dtype=np.int8)
    for i, (front, back) in enumerate(tickets):
        f[i, front] = 1
        b[i, back] = 1
    return f, b


def trigger_probability(v, t, n_back):
    """保证条件触发的精确概率（超几何分布，无需模拟）。

    前区：5 个开奖号中至少 t 个落在 v 个号码池内
    后区：2 个开奖号都落在 n_back 个号码池内
    """
    p_front = sum(
        comb(v, i) * comb(FRONT_MAX - v, 5 - i)
        for i in range(t, 6)
        if 5 - i <= FRONT_MAX - v
    ) / comb(FRONT_MAX, 5)

    p_back = comb(n_back, 2) / comb(BACK_MAX, 2) if n_back else 1.0
    return p_front, p_back, p_front * p_back


def exact_expected_value():
    """单注期望回报——精确计算，不用模拟。

    P(前区中 a 个) = C(5,a)·C(30,5-a)/C(35,5)   （超几何）
    P(后区中 c 个) = C(2,c)·C(10,2-c)/C(12,2)
    E = Σ P(a)·P(c)·奖金(a,c)

    关键：期望是线性的，所以 N 注注单的总期望恒等于 N × 单注期望，
    与这 N 注怎么选、怎么构造完全无关。这就是"wheel 不改变期望"的
    严格证明——不是经验观察，是恒等式。
    """
    total = 0.0
    breakdown = {}
    for (a, c), (lv, amt) in PRIZE_TABLE.items():
        pa = comb(5, a) * comb(FRONT_MAX - 5, 5 - a) / comb(FRONT_MAX, 5)
        pc = comb(2, c) * comb(BACK_MAX - 2, 2 - c) / comb(BACK_MAX, 2)
        ev = pa * pc * amt
        total += ev
        breakdown[(a, c)] = (lv, pa * pc, ev)
    return total, breakdown


def simulate(tickets, n_trials=200_000, seed=0, batch=20_000):
    """蒙特卡洛：模拟 n_trials 次开奖，统计整套注单的回报。"""
    fmask, bmask = _masks(tickets)
    rng = np.random.default_rng(seed)

    levels = {lv: 0 for lv in LEVEL_NAMES}
    payouts = np.empty(n_trials, dtype=np.int64)
    done = 0

    while done < n_trials:
        n = min(batch, n_trials - done)

        gf = rng.gumbel(size=(n, FRONT_MAX))
        draw_f = np.zeros((n, FRONT_MAX), dtype=np.int8)
        np.put_along_axis(draw_f, np.argpartition(-gf, 5, axis=1)[:, :5], 1, axis=1)

        gb = rng.gumbel(size=(n, BACK_MAX))
        draw_b = np.zeros((n, BACK_MAX), dtype=np.int8)
        np.put_along_axis(draw_b, np.argpartition(-gb, 2, axis=1)[:, :2], 1, axis=1)

        hf = fmask @ draw_f.T  # (注数, n)
        hb = bmask @ draw_b.T

        pay = np.zeros((len(tickets), n), dtype=np.int64)
        for (a, c), (lv, amt) in PRIZE_TABLE.items():
            hit = (hf == a) & (hb == c)
            pay += hit * amt
            levels[lv] += int(hit.sum())

        payouts[done : done + n] = pay.sum(axis=0)
        done += n

    return payouts, levels


def report(v, t, n_back, front_blocks, n_trials=200_000, seed=0):
    tickets = build_tickets(front_blocks, n_back) if n_back else [
        (list(b), []) for b in front_blocks
    ]
    cost = len(tickets) * TICKET_PRICE

    p_front, p_back, p_joint = trigger_probability(v, t, n_back)
    payouts, levels = simulate(tickets, n_trials, seed)

    mean = payouts.mean()
    lines = [
        f"注单：{len(tickets):,} 注，成本 {cost:,} 元",
        "",
        "保证条件触发概率（精确计算）：",
        f"  5 个前区号中 ≥{t} 个落在你圈的 {v} 个里      {p_front:>8.4%}",
    ]
    if n_back:
        lines.append(f"  2 个后区号都落在你圈的 {n_back} 个里          {p_back:>8.4%}")
        lines.append(f"  两者同时满足（保证生效）                {p_joint:>8.4%}")
        lines.append(f"  → 平均每 {1 / p_joint:,.0f} 期触发一次")

    ev_per_ticket, _ = exact_expected_value()
    ev_total = ev_per_ticket * len(tickets)

    lines += [
        "",
        "期望回报（精确计算，非模拟）：",
        f"  单注期望        {ev_per_ticket:>10.4f} 元   （成本 {TICKET_PRICE} 元）",
        f"  整套注单期望    {ev_total:>10.2f} 元   （成本 {cost:,} 元）",
        f"  真实 ROI        {(ev_total - cost) / cost:>10.1%}   ← 这个数字不会变",
        "",
        f"模拟 {n_trials:,} 期的回报分布：",
        f"  中位数          {np.median(payouts):>10.0f} 元",
        f"  75% 分位        {np.percentile(payouts, 75):>10.0f} 元",
        f"  95% 分位        {np.percentile(payouts, 95):>10.0f} 元",
        f"  颗粒无收的期数  {(payouts == 0).mean():>10.1%}",
        f"  单期最高回报    {payouts.max():>10,} 元",
        f"  模拟均值        {mean:>10.2f} 元   "
        f"（对比精确期望 {ev_total:.2f}）",
    ]

    drift = abs(mean - ev_total) / ev_total
    if drift > 0.15:
        lines += [
            "",
            f"  ⚠ 模拟均值偏离精确期望 {drift:.0%}。奖金分布是重尾的——一次头奖就能",
            "    把 20 万期的均值整个抬起来。**均值必须以精确计算为准，模拟只能**",
            "    **用来看分布形状**。任何用蒙特卡洛均值声称的 ROI 都不可信。",
        ]

    lines += ["", "中奖等级分布（按注计）："]

    total_hits = sum(levels.values())
    for lv in sorted(levels):
        if levels[lv]:
            per = levels[lv] / n_trials
            lines.append(
                f"  {LEVEL_NAMES[lv]}({LEVEL_AMOUNT[lv]:>10,}元)  "
                f"{levels[lv]:>8,} 注   平均每期 {per:>6.3f} 注"
            )
    if not total_hits:
        lines.append("  （模拟期间未中任何奖）")

    lines += [
        "",
        f"对照：随便买 {len(tickets):,} 注，期望回报**一模一样**是 {ev_total:.2f} 元。",
        "因为期望是线性的，N 注的总期望恒等于 N × 单注期望，与注单怎么构造无关。",
        "wheel 改变的只是回报的分布形状——把概率质量集中到保证条件上，",
        "代价是放弃随机注单那种分散命中的可能。期望值一分钱都没有增加。",
    ]
    return "\n".join(lines)
