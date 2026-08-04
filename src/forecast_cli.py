"""判断校准的命令行入口。

    python -m src.forecast_cli add "比特币年内破 15 万" 0.35 2026-12-31 \\
        "以 CoinGecko 日收盘价为准，任意一天 ≥150000 USD 即为真"
    python -m src.forecast_cli list
    python -m src.forecast_cli settle f0001 yes "11-02 收盘 151203"
    python -m src.forecast_cli score
    python -m src.forecast_cli verify
    python -m src.forecast_cli anchor
"""

import sys
from datetime import date

from . import forecast as fc

TRUE = {"yes", "y", "true", "1", "对", "是", "真"}
FALSE = {"no", "n", "false", "0", "错", "否", "假"}


def cmd_add(args):
    if len(args) < 4:
        print('用法: add "判断内容" 信心度 揭晓日 "判定标准"', file=sys.stderr)
        print('例:   add "比特币年内破 15 万" 0.35 2026-12-31 '
              '"以 CoinGecko 日收盘价为准，任意一天 ≥150000 USD 即为真"',
              file=sys.stderr)
        return 2
    claim, prob, by, criteria = args[0], args[1], args[2], " ".join(args[3:])
    r = fc.add(claim, prob, by, criteria)
    print(f"已锁定 {r['id']}　信心度 {r['probability']:.0%}　揭晓日 {r['resolve_by']}")
    print(f"  {r['claim']}")
    print(f"  判定标准：{r['criteria']}")
    print(f"  hash {r['hash'][:16]}…")
    print("\n建议接着跑 anchor，让这条判断进入比特币时间戳。")
    return 0


def cmd_list(args):
    records = fc.load()
    pend = fc.pending(records)
    done = fc.resolved(records)

    if pend:
        print(f"\n待结算（{len(pend)} 条）")
        for r, due in sorted(pend, key=lambda x: x[0]["resolve_by"]):
            mark = "  ← 已过揭晓日，可以结算了" if due else ""
            print(f"  {r['id']}  {r['probability']:>4.0%}  "
                  f"{r['resolve_by']}  {r['claim']}{mark}")
    if done:
        print(f"\n已结算（{len(done)} 条）")
        for f, s in done:
            hit = "✓ 发生" if s["outcome"] else "✗ 未发生"
            print(f"  {f['id']}  {f['probability']:>4.0%}  {hit}  {f['claim']}")
    if not records:
        print("还没有任何判断。用 add 记第一条。")
    return 0


def cmd_settle(args):
    if len(args) < 2:
        print("用法: settle <id> <yes|no> [备注]", file=sys.stderr)
        return 2
    v = args[1].strip().lower()
    if v not in TRUE and v not in FALSE:
        print(f"结果只能是 yes 或 no，给的是 {args[1]!r}", file=sys.stderr)
        return 2
    r = fc.settle(args[0], v in TRUE, " ".join(args[2:]))
    print(f"{r['ref']} 已结算：{'发生' if r['outcome'] else '未发生'}"
          f"{'　' + r['note'] if r['note'] else ''}")
    print(f"  hash {r['hash'][:16]}…　结算记录本身也在链上，不可改判。")
    return 0


def cmd_score(args):
    s = fc.score()
    if not s["n"]:
        print("还没有已结算的判断。等揭晓日到了先 settle。")
        return 0

    print(f"\n已结算 {s['n']} 条　实际发生率 {s['base_rate']:.0%}")
    print(f"  你的 Brier 分数    {s['brier']:.4f}   （越低越好）")
    print(f"  参考模型 Brier     {s['ref_brier']:.4f}   "
          f"（只知道发生率、对每件事报同一个数）")
    if s["bss"] is not None:
        v = s["bss"]
        verdict = "你的判断带来了信息" if v > 0 else "还不如只报基础发生率"
        print(f"  技巧分 BSS        {v:>+.4f}   {verdict}")

    print(f"\n校准表　你说的 vs 实际发生的")
    print(f"  {'信心区间':<12}{'条数':>5}{'你说':>8}{'实际':>8}{'偏差':>9}")
    for row in s["table"]:
        d = row["said"] - row["actual"]
        print(f"  {row['range']:<12}{row['n']:>5}{row['said']:>8.0%}"
              f"{row['actual']:>8.0%}{d:>+9.0%}")

    if s["n"] < 20:
        print(f"\n样本只有 {s['n']} 条，现在别下结论。"
              "校准要看出系统性偏差，一般需要几十条以上。")
    return 0


def cmd_verify(args):
    records = fc.load()
    ok, bad = fc.verify_chain(records)
    anchors = fc._anchor_log()
    print(f"  判断链　{'完整' if ok else f'断裂@{bad}'}　"
          f"共 {len(records)} 条记录　已锚定 {len(anchors)} 次")
    return 0 if ok else 1


def cmd_anchor(args):
    r = fc.stamp()
    if r.get("skipped"):
        print("当前链头已锚定，跳过")
    elif r["ok"]:
        print(f"✓ 已提交　链头 {r['head'][:16]}…")
    else:
        print(f"✗ 失败：{r.get('error') or r.get('output', '')[:120]}")
    up = fc.upgrade()
    if up:
        print(f"升级待确认凭证：{sum(1 for x in up if x['upgraded'])}/{len(up)} 份已完整")
    return 0 if r.get("ok") else 1


COMMANDS = {"add": cmd_add, "list": cmd_list, "settle": cmd_settle,
            "score": cmd_score, "verify": cmd_verify, "anchor": cmd_anchor}


def main(argv):
    if len(argv) < 2 or argv[1] not in COMMANDS:
        print(__doc__.strip())
        return 2
    try:
        return COMMANDS[argv[1]](argv[2:])
    except fc.ForecastError as e:
        print(f"拒绝：{e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
