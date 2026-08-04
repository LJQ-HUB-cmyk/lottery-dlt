"""摘要文件只能写一次——这是比特币时间戳成立的前提。

锚定的做法是：把链头摘要写成 json，再用 OpenTimestamps 对**这个文件的
字节**盖章。凭证承诺的是文件的 SHA-256，所以文件内容一旦变动，凭证立刻
失配，而且失配是静默的——不看 `ots verify` 根本不会发现。

危险在于文件名只由记录数和链头决定，内容里却有 created_at。同一个链头
第二次生成摘要时若重新写入，时间戳变了，字节就变了，之前盖的章全部作废。

这个 bug 真实发生过：dlt 最早那份 chain-000008 的凭证一度失配，靠锚定
日志里留存的原始 created_at 才逐字节复原（重建文件的 SHA-256 与凭证
承诺值相符，属自证）。这里把它钉死。
"""

import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import anchor, lotteries  # noqa: E402


def check_digest_is_write_once():
    """同一条链上重复调用 digest_file，文件字节必须一模一样。"""
    print("\n[1] 摘要文件写一次即冻结")
    lot = lotteries.LOTTERIES["dlt"]
    records = [
        {"hash": "a" * 64, "issue": "26001"},
        {"hash": "b" * 64, "issue": "26002"},
    ]

    tmp = Path(tempfile.mkdtemp())
    orig_base = anchor.BASE
    anchor.BASE = tmp
    try:
        p1, d1 = anchor.digest_file(lot, records)
        b1 = p1.read_bytes()
        h1 = hashlib.sha256(b1).hexdigest()

        p2, d2 = anchor.digest_file(lot, records)
        b2 = p2.read_bytes()
        h2 = hashlib.sha256(b2).hexdigest()
    finally:
        anchor.BASE = orig_base
        shutil.rmtree(tmp, ignore_errors=True)

    same_path = p1 == p2
    same_bytes = b1 == b2
    same_time = d1["created_at"] == d2["created_at"]
    ok = same_path and same_bytes and same_time

    print(f"  路径一致      {same_path}")
    print(f"  字节一致      {same_bytes}")
    print(f"  created_at 一致 {same_time}")
    print(f"  首次 SHA-256  {h1[:32]}…")
    print(f"  再次 SHA-256  {h2[:32]}…")
    print("  通过：重复调用不会改写文件，已盖的章不会失效" if ok
          else "  ✗ 文件被改写——所有已有凭证会静默作废")
    return ok


def check_existing_proofs_match():
    """仓库里每份 .ots 凭证，都应与它的摘要文件对得上。

    只做本地哈希比对，不需要比特币节点：凭证内部记录了被盖章文件的
    SHA-256，拿它和磁盘上的文件比即可。
    """
    print("\n[2] 现有凭证与摘要文件的对应关系")
    try:
        import opentimestamps.core.serialize as ser
        from opentimestamps.core.timestamp import DetachedTimestampFile
    except ImportError:
        print("  跳过：未安装 opentimestamps 客户端")
        return True

    ok, n = True, 0
    for ots in sorted((ROOT / "data").glob("anchors-*/*.json.ots")):
        target = ots.with_suffix("")
        if not target.exists():
            print(f"  {ots.name:<44} ✗ 缺少原文件")
            ok = False
            continue
        with ots.open("rb") as f:
            det = DetachedTimestampFile.deserialize(ser.StreamDeserializationContext(f))
        want = det.file_digest.hex()
        got = hashlib.sha256(target.read_bytes()).hexdigest()
        match = want == got
        ok &= match
        n += 1
        print(f"  {ots.name:<44} {'✓' if match else '✗ 失配'}")

    print(f"  通过：{n} 份凭证全部与原文件相符" if ok else "  ✗ 有凭证已失配")
    return ok


def main():
    print("=" * 56)
    print("锚定摘要文件的不可变性")
    print("=" * 56)
    results = [check_digest_is_write_once(), check_existing_proofs_match()]
    print("\n" + "=" * 56)
    if all(results):
        print("全部通过。摘要文件写一次即冻结，比特币时间戳的承诺成立。")
        return 0
    print(f"有 {results.count(False)} 项未通过——时间戳可信度受影响。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
