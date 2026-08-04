"""定时任务：多彩种数据更新、结算、锁定预测、上链。

各彩种开奖时间（当地时区）：
    大乐透        周一/三/六 20:30 CST
    双色球        周二/四/日 21:15 CST
    Powerball    周一/三/六 22:59 ET
    EuroMillions 周二/五     20:45 CET

建议 crontab（UTC，覆盖所有时区，多跑几次无副作用——有新数据才动作）：
    0 */3 * * *   cd /app && .venv/bin/python -m src.cron settle
    30 */6 * * *  cd /app && .venv/bin/python -m src.cron lock
    0 4 * * *     cd /app && .venv/bin/python -m src.cron anchor

settle：抓最新开奖 → 已锁定的预测自动结算 → 清缓存
lock  ：为下一期生成并锁定预测（必须在开奖前跑）
anchor：把各彩种链头锚定到比特币
"""

import sys
import traceback
from datetime import datetime

from . import anchor as anchor_mod
from . import cache, fetch, lotteries, track
from .fetch import data_path
from .markov import load


def _each(fn):
    """对所有启用的彩种执行，单个失败不影响其他。"""
    rc = 0
    for lot in lotteries.enabled():
        try:
            fn(lot)
        except Exception as e:
            rc = 1
            print(f"  ✗ {lot.name}: {type(e).__name__}: {e}", file=sys.stderr)
            if "-v" in sys.argv:
                traceback.print_exc()
    return rc


def cmd_settle():
    changed = []

    def one(lot):
        total, added = fetch.update(lot)
        mark = f"+{added}" if added else "无新增"
        print(f"  {lot.name:<14} {total:>5} 期  {mark}")
        if added:
            changed.append(lot.key)

    print(f"[{datetime.now():%F %T}] 抓取开奖数据")
    rc = _each(one)

    if changed:
        n = cache.clear()
        print(f"  {len(changed)} 个彩种有新数据，清理缓存 {n} 项")

    print("\n战绩：")

    def report(lot):
        p = data_path(lot)
        if not p.exists():
            return
        st = track.status(load(p, lot), lot)
        flag = "✓" if st["chain_valid"] else "✗ 链断裂"
        print(f"  {lot.name:<14} {flag}  预测 {st['total_predictions']:>4}  "
              f"已结算 {st['settled']:>4}  待开奖 {len(st['pending']):>2}")
        for s in st["summary"][:3]:
            beat = "  ← 显著优于随机" if s["beats_random"] else ""
            pa = "—" if s["p_adjusted"] is None else f"{s['p_adjusted']:.3f}"
            print(f"      {s['name']:<10} n={s['n']:<4} "
                  f"平均命中={s['mean_hits']:<6} 期望={s['expected_hits']:<6} "
                  f"校正p={pa:<6} ROI={s['roi']:>7.1%}{beat}")

    _each(report)
    return rc


def cmd_lock():
    print(f"[{datetime.now():%F %T}] 锁定下期预测")

    def one(lot):
        p = data_path(lot)
        if not p.exists():
            print(f"  {lot.name:<14} 跳过（无数据）")
            return
        df = load(p, lot)
        issue, added = track.make_predictions(df, lot)
        ok, bad = track.verify_chain(lot=lot)
        if not ok:
            raise RuntimeError(f"哈希链在第 {bad} 条断裂")
        status = f"新增 {len(added)} 条" if added else "已存在，跳过"
        print(f"  {lot.name:<14} {issue} 期  {status}  链✓")

    return _each(one)


def cmd_anchor():
    print(f"[{datetime.now():%F %T}] 锚定链头到比特币")

    def one(lot):
        st = anchor_mod.status(lot)
        if st["current_head"] is None:
            print(f"  {lot.name:<14} 跳过（无预测记录）")
            return
        if st["head_anchored"]:
            print(f"  {lot.name:<14} 当前链头已锚定，跳过")
            return
        r = anchor_mod.stamp(lot)
        print(f"  {lot.name:<14} {'✓ 已提交' if r.get('ok') else '✗ 失败'}  "
              f"链头 {(st['current_head'] or '')[:16]}…")

    rc = _each(one)

    print("\n升级待确认凭证（比特币确认后才能完成）：")

    def up(lot):
        res = anchor_mod.upgrade(lot)
        done = sum(1 for x in res if x["upgraded"])
        if res:
            print(f"  {lot.name:<14} {done}/{len(res)} 份证明已完整")

    _each(up)
    return rc


def cmd_verify():
    rc = 0
    for lot in lotteries.enabled():
        ok, bad = track.verify_chain(lot=lot)
        st = anchor_mod.status(lot)
        print(f"  {lot.name:<14} 链{'完整' if ok else f'断裂@{bad}'}  "
              f"锚定 {st['total']} 次  未锚定 {st['unanchored']} 条")
        if not ok:
            rc = 1
    return rc


def cmd_null():
    """空对照：同一个模型分别喂真实开奖和确定随机数据，比较它学到了什么。"""
    from . import null as null_mod

    print(f"[{datetime.now():%F %T}] 空对照实验")
    print("  用 numpy 生成等概率无放回的合成开奖，与真实开奖跑同一个堆叠模型。")
    print("  若模型在两者上学出的结构一致，说明它在真实开奖里也什么都没学到。")
    return _each(lambda lot: null_mod.report(lot))


COMMANDS = {"settle": cmd_settle, "lock": cmd_lock,
            "anchor": cmd_anchor, "verify": cmd_verify, "null": cmd_null}


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd not in COMMANDS:
        raise SystemExit(f"用法：python -m src.cron [{' | '.join(COMMANDS)}]")
    raise SystemExit(COMMANDS[cmd]())
