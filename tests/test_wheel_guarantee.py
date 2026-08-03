"""端到端验证 wheel 的保证在真实开奖语义下确实成立。

src.wheel.verify 只证明了组合学性质（所有 t-子集被覆盖）。这里验证的是
彩票语义：模拟大量开奖，凡是"开奖号中有 ≥t 个落在号码池内"的场次，
必须存在至少一注命中了 t 个。一次反例都不允许。

同时确认反面：条件不满足时保证不适用（否则说明保证被高估了）。
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.wheel import solve, verify  # noqa: E402

FRONT_MAX = 35
N_TRIALS = 200_000


def check(v, t, seed=0):
    r = solve(v, 5, t, seeds=3)
    blocks = r["best"]

    # 号码池取 1..v，wheel 的下标 0..v-1 映射成号码 1..v
    tickets = [set(x + 1 for x in b) for b in blocks]
    pool = set(range(1, v + 1))

    rng = np.random.default_rng(seed)
    numbers = np.arange(1, FRONT_MAX + 1)

    triggered = violations = 0
    best_hit_when_short = 0

    for _ in range(N_TRIALS):
        g = rng.gumbel(size=FRONT_MAX)
        draw = set(numbers[np.argpartition(-g, 5)[:5]].tolist())

        in_pool = draw & pool
        best_hit = max(len(tk & draw) for tk in tickets)

        if len(in_pool) >= t:
            triggered += 1
            if best_hit < t:
                violations += 1
        else:
            best_hit_when_short = max(best_hit_when_short, best_hit)

    return {
        "v": v, "t": t, "size": r["size"], "lb": r["lower_bound"],
        "combinatorial_gap": verify(blocks, v, t),
        "triggered": triggered, "violations": violations,
        "max_hit_when_short": best_hit_when_short,
    }


def main():
    print(f"每个配置模拟 {N_TRIALS:,} 次开奖（前区 35 选 5）\n")
    print(f"{'配置':<14}{'注数':>5}{'下界':>5}{'组合缺口':>9}"
          f"{'触发次数':>9}{'违约':>7}")
    print("-" * 52)

    all_ok = True
    for v, t in [(8, 3), (9, 3), (10, 3), (9, 4)]:
        r = check(v, t)
        ok = r["violations"] == 0 and r["combinatorial_gap"] == 0
        all_ok &= ok
        print(f"C({r['v']},5,{r['t']}){'':<6}{r['size']:>5}{r['lb']:>5}"
              f"{r['combinatorial_gap']:>9}{r['triggered']:>9}"
              f"{r['violations']:>7}{'' if ok else '  ✗'}")

    print("-" * 52)
    if all_ok:
        print("\n通过：所有触发场次的保证全部兑现，零违约。")
        print("wheel 声称的「保证」在开奖语义下成立。")
        return 0
    print("\n失败：存在违约场次，保证不成立。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
