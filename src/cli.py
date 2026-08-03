"""命令行入口。"""

import argparse
from itertools import combinations
from pathlib import Path

from . import analyze, fetch, markov, wheel
from .prize import LEVEL_NAMES, TICKET_PRICE, guaranteed_level

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "dlt_history.csv"

DISCLAIMER = """
────────────────────────────────────────────────────────────
本工具不预测开奖结果，也无法提高中奖概率。
马尔可夫检验回答的是"序列有没有记忆"，wheel 优化的是"注数
怎么分配才能兑现某个保证"——两者都不改变期望回报。
理性对待，勿投入超出承受能力的资金。
────────────────────────────────────────────────────────────"""


def _require_data():
    if not DATA_PATH.exists():
        raise SystemExit("未找到数据，请先运行：python -m src.cli fetch")
    return markov.load(DATA_PATH)


def cmd_fetch(args):
    total, added = fetch.update()
    print(f"完成：共 {total} 期，新增 {added} 期 → {DATA_PATH}")


def cmd_test(args):
    df = _require_data()
    print(markov.format_report(markov.run(df, args.alpha), len(df), args.alpha))
    print(DISCLAIMER)


def cmd_wheel(args):
    v, t = args.pool, args.guarantee
    if not 5 <= v <= 35:
        raise SystemExit("前区号码池须在 5..35")
    if not 1 <= t <= 5:
        raise SystemExit("保证命中数须在 1..5")

    print(f"求解 C({v}, 5, {t}) —— 圈 {v} 个前区号，保证命中其中任意 {t} 个")
    if args.exact:
        print(f"（含 ILP 精确求解，上限 {args.exact:.0f} 秒）")
    print()
    r = wheel.solve(v, 5, t, seeds=args.seeds, exact_time=args.exact)

    print(f"  Schönheim 下界   {r['lower_bound']:>6} 注")
    if r["dual_bound"] is not None and r["dual_bound"] > r["lower_bound"]:
        print(f"  ILP 对偶界       {r['dual_bound']:>6} 注   （比 Schönheim 更紧）")
    print(f"  贪心构造         {r['greedy']:>6} 注")
    print(f"  循环群构造       {r['cyclic']:>6} 注")
    print(f"  最终采用         {r['size']:>6} 注   [{r['method']}]")
    print(f"  暴力验证缺口     {r['uncovered']:>6}   "
          f"{'完全覆盖 ✓' if r['uncovered'] == 0 else '未完全覆盖 ✗'}")

    if r["proven"]:
        print(f"\n  ★ {r['size']} 注已被 ILP 证明是最优解，不存在更少的构造。")
        if r["size"] > r["lower_bound"]:
            print(f"    与 Schönheim 下界 {r['lower_bound']} 的差距来自下界不紧，"
                  "而非构造不力。")
    elif r["dual_bound"] is not None:
        print(f"\n  真实最优被夹在 [{r['dual_bound']}, {r['size']}] 之间"
              "（ILP 未在时限内证完）。")

    n_back = comb_back = None
    if args.back:
        n_back = args.back
        comb_back = len(list(combinations(range(n_back), 2)))

    tickets = r["size"] * (comb_back or 1)
    print(f"\n  总注数 {tickets:,}   成本 {tickets * TICKET_PRICE:,} 元"
          + (f"（前区 {r['size']} × 后区 C({n_back},2)={comb_back}）" if comb_back else ""))

    tb = 2 if args.back else 0
    level, amount = guaranteed_level(t, tb)
    print(f"\n  保证条件：开奖 5 个前区号中有 {t} 个落在你圈的 {v} 个里"
          + (f"，且 2 个后区号都落在你圈的 {n_back} 个里" if args.back else ""))
    if level:
        print(f"  → 必中至少 {LEVEL_NAMES[level]}（约 {amount:,} 元）")
    else:
        print(f"  → 命中 {t}+{tb} 不构成奖级")

    if args.show:
        print("\n  号码组合（下标从 1 开始，对应你圈的第几个号）：")
        for i, b in enumerate(r["best"], 1):
            print(f"    {i:>3}. {' '.join(f'{x + 1:>2}' for x in b)}")

    if args.analyze:
        print("\n" + "─" * 60)
        print(analyze.report(v, t, args.back or 0, r["best"], n_trials=args.analyze))

    print(DISCLAIMER)



def cmd_pick(args):
    """直接出一套可以拿去买的号码。"""
    import numpy as np

    rng = np.random.default_rng(args.seed)
    front_pool = sorted(rng.choice(np.arange(1, 36), args.pool, replace=False).tolist())
    back_pool = sorted(rng.choice(np.arange(1, 13), args.back, replace=False).tolist())

    print(f"前区号码池（{args.pool} 个）：{' '.join(f'{n:02d}' for n in front_pool)}")
    print(f"后区号码池（{args.back} 个）：{' '.join(f'{n:02d}' for n in back_pool)}")
    print()

    r = wheel.solve(args.pool, 5, args.guarantee, seeds=args.seeds)
    backs = list(combinations(back_pool, 2))
    total = len(r["best"]) * len(backs)

    print(f"共 {total} 注，{total * TICKET_PRICE} 元"
          f"（前区 {len(r['best'])} 组 × 后区 {len(backs)} 组）\n")

    n = 0
    for blk in r["best"]:
        f = " ".join(f"{front_pool[i]:02d}" for i in blk)
        for bk in backs:
            n += 1
            print(f"  {n:>3}.  {f}  +  {bk[0]:02d} {bk[1]:02d}")

    level, amount = guaranteed_level(args.guarantee, 2)
    _, _, p_joint = analyze.trigger_probability(args.pool, args.guarantee, args.back)
    ev, _ = analyze.exact_expected_value()

    print(f"\n  保证：开奖 5 个前区号有 {args.guarantee} 个落在上面的池子里、"
          f"且 2 个后区号都在池内时，")
    print(f"        必中至少{LEVEL_NAMES[level]}（约 {amount:,} 元）"
          f"—— 触发概率 {p_joint:.4%}，平均每 {1/p_joint:,.0f} 期一次")
    print(f"  期望：{ev * total:.2f} 元回报 / {total * TICKET_PRICE} 元成本"
          f"  =  ROI {(ev * total - total * TICKET_PRICE) / (total * TICKET_PRICE):.1%}")
    print("\n  这套号码的中奖概率与任何其他组合完全相同。")
    print(DISCLAIMER)



def cmd_bet(args):
    """记录一次真实投注。"""
    from . import lotteries, mybets, track
    from .fetch import data_path
    from .markov import load

    lot = lotteries.get(args.lottery)
    dp = data_path(lot)
    if not dp.exists():
        raise SystemExit(f"未找到 {lot.name} 数据，请先 fetch")
    df = load(dp, lot)

    issue = args.issue or track.next_issue(df, lot)
    tickets = []

    if args.from_algo:
        # 照抄某算法对该期的预测——必须是开奖前已锁定的那条
        recs = [r for r in track.load_records(lot)
                if r["issue"] == issue and r["algo"] == args.from_algo]
        if not recs:
            raise SystemExit(
                f"{issue} 期没有 {args.from_algo} 的已锁定预测，先跑 cron lock")
        tickets = [(recs[0]["front"], recs[0]["back"])]
        source = args.from_algo
    else:
        if not args.numbers:
            raise SystemExit("请用 --numbers 给号码，或用 --from 照抄某算法预测")
        for txt in args.numbers:
            tickets.append(mybets.parse_ticket(txt, lot))
        source = "manual"

    rec = mybets.add_bet(lot, issue, tickets, cost=args.cost,
                         note=args.note or "", source=source)

    print(f"已记录：{lot.name} {issue} 期 {len(tickets)} 注，"
          f"成本 {rec['cost']:g} {lot.currency}")
    for t in rec["tickets"]:
        f = " ".join(f"{n:02d}" for n in t["front"])
        b = " ".join(f"{n:02d}" for n in t["back"])
        print(f"    {f}" + (f"  +  {b}" if b else ""))
    print(f"  记录哈希 {rec['hash'][:16]}…")

    ok, bad = mybets.verify_chain(lot=lot)
    print(f"  投注链{'完整 ✓' if ok else f'断裂 @{bad} ✗'}")


def cmd_mybets(args):
    """查看真实战绩。"""
    from . import lotteries, mybets
    from .fetch import data_path
    from .markov import load

    lot = lotteries.get(args.lottery)
    df = load(data_path(lot), lot)
    st = mybets.status(lot, df)
    s = st["summary"]

    print(f"{lot.name} 真实投注战绩　　链{'完整 ✓' if st['chain_valid'] else '断裂 ✗'}\n")
    if not s["periods"]:
        print("  还没有已结算的投注")
        if s["pending"]:
            print(f"  {s['pending']} 笔待开奖")
        return

    cur = lot.currency
    print(f"  已结算 {s['periods']} 期，共 {s['tickets']} 注")
    print(f"  投入 {s['cost']:g} {cur}    回报 {s['payout']:g} {cur}    "
          f"净 {s['net']:+g} {cur}")
    print(f"  实际 ROI {s['roi']:>7.1%}     理论期望 ROI {s['expected_roi']:>7.1%}")
    print(f"  同注数的理论期望回报 {s['expected_payout']:g} {cur}"
          f"（实际 {s['payout']:g}）")

    if s["levels"]:
        print("\n  中奖情况：")
        for k, v in s["levels"].items():
            print(f"    {k} × {v}")

    diff = s["payout"] - s["expected_payout"]
    print(f"\n  {'高于' if diff >= 0 else '低于'}理论期望 {abs(diff):g} {cur}")
    print("  注意：样本量小的时候，实际值围绕期望大幅波动是正常的。")

    print("\n  逐期记录：")
    for b in st["bets"]:
        if not b["settled"]:
            print(f"    {b['issue']}  {len(b['tickets'])}注  待开奖")
            continue
        hits = " ".join(f"{d['hit_front']}+{d['hit_back']}" for d in b["detail"])
        print(f"    {b['issue']}  {len(b['tickets'])}注  成本{b['cost']:g}  "
              f"回报{b['payout']:g}  净{b['net']:+g}  命中[{hits}]")


def main():
    p = argparse.ArgumentParser(prog="python -m src.cli", description="大乐透组合分析工具")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("fetch", help="抓取/更新历史开奖数据").set_defaults(func=cmd_fetch)

    pt = sub.add_parser("test", help="马尔可夫性检验：序列有没有记忆")
    pt.add_argument("--alpha", type=float, default=0.05, help="显著性水平")
    pt.set_defaults(func=cmd_test)

    pw = sub.add_parser("wheel", help="求解覆盖设计 wheel")
    pw.add_argument("--pool", type=int, required=True, help="前区号码池大小 v")
    pw.add_argument("--guarantee", type=int, required=True, help="保证命中数 t")
    pw.add_argument("--back", type=int, help="后区号码池大小（可选，默认只算前区）")
    pw.add_argument("--seeds", type=int, default=4, help="退火重试种子数")
    pw.add_argument("--exact", type=float, default=0.0, metavar="秒",
                    help="ILP 精确求解的时限，>0 时启用（可证明最优性）")
    pw.add_argument("--show", action="store_true", help="打印具体号码组合")
    pw.add_argument("--analyze", type=int, default=0, metavar="期数",
                    help="模拟指定期数，分析实际回报与触发概率")
    pw.set_defaults(func=cmd_wheel)

    pp = sub.add_parser("pick", help="直接出一套可以买的号码")
    pp.add_argument("--pool", type=int, default=9, help="前区号码池大小")
    pp.add_argument("--back", type=int, default=3, help="后区号码池大小")
    pp.add_argument("--guarantee", type=int, default=3, help="保证命中数")
    pp.add_argument("--seed", type=int, default=None, help="随机种子，不给则每次不同")
    pp.add_argument("--seeds", type=int, default=4)
    pp.set_defaults(func=cmd_pick)

    pb = sub.add_parser("bet", help="记录一次真实投注")
    pb.add_argument("--lottery", default="dlt", help="彩种")
    pb.add_argument("--issue", help="期号，默认下一期")
    pb.add_argument("--numbers", nargs="+", metavar="号码",
                    help='每注一个字符串，如 "7,9,29,33,35+1,11"')
    pb.add_argument("--from", dest="from_algo", metavar="算法",
                    help="照抄该算法对本期的已锁定预测，如 mix")
    pb.add_argument("--cost", type=float, help="实际花费，默认按注数×单价")
    pb.add_argument("--note", help="备注")
    pb.set_defaults(func=cmd_bet)

    pm = sub.add_parser("mybets", help="查看真实投注战绩")
    pm.add_argument("--lottery", default="dlt")
    pm.set_defaults(func=cmd_mybets)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
