"""判断校准链：写入守卫、防篡改、以及 Brier／校准的算法正确性。

这套东西和彩票链共享同一个失败模式：**哈希链完好，记录的却是一句
可以随便解释的话**。密码学防不住这个，只有写入时的守卫能。所以守卫
本身必须被测——它们是这里唯一的防线。

计分部分单独验证，因为「我判断挺准」这句话的全部重量都压在它上面：
Brier 分数算错、或者拿错的基线去比，得出的结论会和没有基线时一样空。
"""

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import forecast as fc  # noqa: E402

FUTURE = "2099-12-31"


def fresh_chain():
    """把模块指向一个空的临时链，不碰真实数据。"""
    d = Path(tempfile.mkdtemp())
    fc.DATA = d
    fc.CHAIN = d / "forecasts.jsonl"
    fc.ANCHOR_DIR = d / "anchors-forecast"
    fc.ANCHOR_LOG = d / "anchors-forecast.jsonl"
    return d


def expect_reject(fn, label):
    try:
        fn()
        print(f"    ✗ {label}：本该拒绝却写进去了")
        return False
    except fc.ForecastError:
        print(f"    ✓ {label}")
        return True


def check_guards():
    """守卫是这里唯一的防线，逐条撞一遍。"""
    print("\n[1] 写入守卫")
    fresh_chain()
    ok = True
    good = "以官方公布的数据为准，超过阈值即为真"

    ok &= expect_reject(lambda: fc.add("x", 1.0, FUTURE, good), "信心度 1.0 被拒")
    ok &= expect_reject(lambda: fc.add("x", 0.0, FUTURE, good), "信心度 0.0 被拒")
    ok &= expect_reject(lambda: fc.add("x", 0.5, FUTURE, "含糊"), "判定标准过短被拒")
    ok &= expect_reject(lambda: fc.add("", 0.5, FUTURE, good), "空判断被拒")
    ok &= expect_reject(lambda: fc.add("x", 0.5, "2020-01-01", good),
                        "揭晓日已过被拒")
    ok &= expect_reject(lambda: fc.add("x", 0.5, "2099/12/31", good),
                        "日期格式错误被拒")

    r = fc.add("正常的判断", 0.35, FUTURE, good)
    print(f"    ✓ 合法输入写入成功（{r['id']}）")

    fc.settle(r["id"], True, "备注")
    ok &= expect_reject(lambda: fc.settle(r["id"], False), "二次结算被拒")
    ok &= expect_reject(lambda: fc.settle("f9999", True), "结算不存在的判断被拒")

    print("  通过：所有守卫生效" if ok else "  ✗ 有守卫失效")
    return ok


def check_tamper():
    """改动任何一条记录，链必须断，且能指出位置。"""
    print("\n[2] 防篡改")
    fresh_chain()
    good = "以官方公布的数据为准，超过阈值即为真"
    for i in range(3):
        fc.add(f"判断 {i}", 0.3 + 0.1 * i, FUTURE, good)
    fc.settle("f0002", True)

    ok, bad = fc.verify_chain()
    print(f"    原始链完整：{ok}（{len(fc.load())} 条记录）")
    if not ok:
        print("  ✗ 干净的链就校验失败了")
        return False

    good_all = True
    for field, value, label in [("probability", 0.99, "改信心度"),
                                ("claim", "改过的说法", "改判断内容"),
                                ("outcome", False, "改结算结果"),
                                ("created_at", "2020-01-01T00:00:00+00:00", "改时间")]:
        records = fc.load()
        idx = next(i for i, r in enumerate(records) if field in r)
        records[idx][field] = value
        broke, at = fc.verify_chain(records)
        hit = (not broke) and at == idx
        good_all &= hit
        print(f"    {'✓' if hit else '✗'} {label} → "
              f"{'断裂@' + str(at) if not broke else '未检出'}")

    print("  通过：任何字段被改都会断链" if good_all else "  ✗ 有改动未被检出")
    return good_all


def check_scoring():
    """Brier 与技巧分必须与手算一致，且方向正确。"""
    print("\n[3] 计分正确性")
    good = "以官方公布的数据为准，超过阈值即为真"

    # 手算校验：信心度 0.9/0.8/0.3/0.1，结果 真/真/假/假
    fresh_chain()
    ps, ys = [0.9, 0.8, 0.3, 0.1], [True, True, False, False]
    for i, (p, y) in enumerate(zip(ps, ys)):
        fc.add(f"判断 {i}", p, FUTURE, good)
        fc.settle(f"f{i + 1:04d}", y)
    s = fc.score()

    want_brier = sum((p - (1 if y else 0)) ** 2 for p, y in zip(ps, ys)) / 4
    want_base = 0.5
    want_ref = sum((want_base - (1 if y else 0)) ** 2 for y in ys) / 4
    ok = (abs(s["brier"] - want_brier) < 1e-12
          and abs(s["base_rate"] - want_base) < 1e-12
          and abs(s["ref_brier"] - want_ref) < 1e-12)
    print(f"    Brier      实得 {s['brier']:.6f}  手算 {want_brier:.6f}")
    print(f"    发生率     实得 {s['base_rate']:.6f}  手算 {want_base:.6f}")
    print(f"    参考 Brier 实得 {s['ref_brier']:.6f}  手算 {want_ref:.6f}")
    print(f"    技巧分 BSS {s['bss']:>+.4f}（判断有信息量，应 > 0）")
    ok &= s["bss"] > 0

    # 反向：故意把信心度全押反，技巧分必须为负
    fresh_chain()
    for i, (p, y) in enumerate(zip(ps, ys)):
        fc.add(f"判断 {i}", 1 - p, FUTURE, good)
        fc.settle(f"f{i + 1:04d}", y)
    bad = fc.score()
    ok &= bad["bss"] < 0
    print(f"    押反后 BSS {bad['bss']:>+.4f}（应 < 0）"
          f"  {'✓' if bad['bss'] < 0 else '✗'}")

    # 校准表：全部报 70% 的十条里恰好 7 条为真 → said≈actual
    fresh_chain()
    for i in range(10):
        fc.add(f"判断 {i}", 0.7, FUTURE, good)
        fc.settle(f"f{i + 1:04d}", i < 7)
    cal = fc.score()
    row = next(r for r in cal["table"] if r["n"] == 10)
    hit = abs(row["said"] - 0.7) < 1e-12 and abs(row["actual"] - 0.7) < 1e-12
    ok &= hit
    print(f"    校准表 你说 {row['said']:.0%} 实际 {row['actual']:.0%}"
          f"  {'✓ 完美校准' if hit else '✗'}")

    print("  通过：计分与手算一致，方向正确" if ok else "  ✗ 计分有误")
    return ok


def check_digest_frozen():
    """摘要文件必须写一次即冻结——彩票那边真实踩过的坑。"""
    print("\n[4] 摘要文件不可重写")
    fresh_chain()
    good = "以官方公布的数据为准，超过阈值即为真"
    fc.add("判断", 0.5, FUTURE, good)

    p1, d1 = fc.digest_file()
    raw1 = p1.read_bytes()
    p2, d2 = fc.digest_file()
    raw2 = p2.read_bytes()

    ok = p1 == p2 and raw1 == raw2 and d1["created_at"] == d2["created_at"]
    print(f"    第二次调用 {'字节完全一致' if ok else '内容变了'}"
          f"  {'✓' if ok else '✗ 之前盖的凭证会静默失配'}")
    print("  通过：摘要冻结" if ok else "  ✗ 摘要会被重写")
    return ok


def main():
    print("=" * 56)
    print("判断校准链的守卫、防篡改与计分验证")
    print("=" * 56)
    results = [check_guards(), check_tamper(), check_scoring(),
               check_digest_frozen()]
    print("\n" + "=" * 56)
    if all(results):
        print("全部通过。")
        print("守卫拦得住含糊的判断，链拦得住事后改动，计分有正确的基线——")
        print("因此「我判断挺准」这句话在这里是可结算的，不是形容词。")
        return 0
    print(f"有 {results.count(False)} 项未通过。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
