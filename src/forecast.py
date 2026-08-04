"""个人判断校准：把「我觉得会怎样」变成可结算、可验证的记录。

和彩票那套是同一台机器，换了个靶子。彩票的价值在于答案已知（随机），
用来证明机器是准的；这里的价值在于答案未知——你自己的判断，到底在
哪些领域真的比瞎猜强。

三条性质与彩票链完全一致：

1. **写在前面**。判断和信心度在事件揭晓前写入 append-only 哈希链，
   链头锚定进比特币。事后改一个数字，链立刻断。
2. **有基线**。Brier 分数对照「永远报基础概率」的参考模型。
   没有基线，「我判断挺准」就是个没有分母的形容词。
3. **输赢全留**。没有删除接口，没有重新结算接口。

**这里最脆弱的一环不是密码学，是结算。**
彩票的「什么算对」是免费的——号码对上就是对上。换成现实判断，
只要判定标准留一条缝，人就会在事后往对自己有利的方向解释，而且
是无意识的。所以 criteria 是必填字段，和判断一起锁进链；结算本身
也是一条链上记录，带自己的时间戳，同一条判断拒绝二次结算。

哈希链完好、却记录着一句可以随便解释的话——那才是这套东西真正的
失败模式。
"""

import hashlib
import json
import re
from datetime import date, datetime, timezone
from pathlib import Path

from .anchor import _ots

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
CHAIN = DATA / "forecasts.jsonl"
ANCHOR_DIR = DATA / "anchors-forecast"
ANCHOR_LOG = DATA / "anchors-forecast.jsonl"
GENESIS = "0" * 64

# 校准分桶。10 个桶在样本少时太碎，5 个足够看出系统性偏差
BUCKETS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]


class ForecastError(Exception):
    """拒绝写入，并说明为什么。"""


def _hash(rec, prev):
    """只把「不可事后改动」的字段计入哈希。

    与 track.py 同构：结算记录的 outcome 也在里面——结算一旦上链，
    改判就等于断链。
    """
    payload = json.dumps({
        "id": rec["id"], "type": rec["type"],
        "claim": rec.get("claim"), "criteria": rec.get("criteria"),
        "probability": rec.get("probability"),
        "resolve_by": rec.get("resolve_by"),
        "ref": rec.get("ref"), "outcome": rec.get("outcome"),
        "created_at": rec["created_at"],
        "prev_hash": prev,
    }, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def load():
    if not CHAIN.exists():
        return []
    with CHAIN.open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def verify_chain(records=None):
    """校验哈希链完整性。返回 (是否完整, 出问题的位置)。"""
    records = load() if records is None else records
    prev = GENESIS
    for i, r in enumerate(records):
        if r.get("prev_hash") != prev or _hash(r, prev) != r.get("hash"):
            return False, i
        prev = r["hash"]
    return True, None


def _append(rec):
    records = load()
    ok, bad = verify_chain(records)
    if not ok:
        raise ForecastError(f"哈希链在第 {bad} 条断裂，拒绝继续写入")
    prev = records[-1]["hash"] if records else GENESIS
    rec["prev_hash"] = prev
    rec["hash"] = _hash(rec, prev)
    DATA.mkdir(parents=True, exist_ok=True)
    with CHAIN.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def _next_id(records, kind):
    n = sum(1 for r in records if r["type"] == kind)
    return f"{'f' if kind == 'forecast' else 's'}{n + 1:04d}"


def add(claim, probability, resolve_by, criteria, now=None):
    """记一条判断。信心度必须给数字，判定标准必须写清楚。"""
    now = now or datetime.now(timezone.utc).astimezone()

    p = float(probability)
    if not 0.0 < p < 1.0:
        raise ForecastError(
            f"信心度要在 0 和 1 之间（不含）。给的是 {p}。"
            "0 或 1 表示绝对确定，那不是预测，而且会让 Brier 分数无法惩罚"
        )
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(resolve_by)):
        raise ForecastError("resolve_by 要写成 YYYY-MM-DD")
    if not str(claim).strip():
        raise ForecastError("判断内容不能为空")
    if len(str(criteria).strip()) < 8:
        raise ForecastError(
            "判定标准必须写清楚（至少 8 个字）。这不是形式要求——"
            "标准留一条缝，事后就会被无意识地往有利方向解释，"
            "那时哈希链完好，记录的却是一句可以随便解释的话"
        )

    y, m, d = map(int, str(resolve_by).split("-"))
    if date(y, m, d) < now.date():
        raise ForecastError(
            f"揭晓日 {resolve_by} 已经过去了，今天是 {now.date()}。"
            "对已经发生的事记「预测」没有意义，也无法自证清白"
        )

    records = load()
    return _append({
        "id": _next_id(records, "forecast"), "type": "forecast",
        "claim": str(claim).strip(), "criteria": str(criteria).strip(),
        "probability": p, "resolve_by": str(resolve_by),
        "created_at": now.isoformat(),
    })


def settle(ref, outcome, note="", now=None):
    """结算一条判断。同一条只能结算一次。"""
    now = now or datetime.now(timezone.utc).astimezone()
    records = load()

    target = next((r for r in records
                   if r["type"] == "forecast" and r["id"] == ref), None)
    if target is None:
        raise ForecastError(f"找不到判断 {ref}")
    if any(r["type"] == "settle" and r["ref"] == ref for r in records):
        raise ForecastError(
            f"{ref} 已经结算过了。链是 append-only，改判等于断链——"
            "如果当初的判定标准写错了，正确做法是记一条新判断，"
            "而不是把旧的改掉"
        )

    return _append({
        "id": _next_id(records, "settle"), "type": "settle",
        "ref": ref, "outcome": bool(outcome), "note": str(note).strip(),
        "created_at": now.isoformat(),
    })


def resolved(records=None):
    """已结算的判断，配上结果。"""
    records = load() if records is None else records
    outcomes = {r["ref"]: r for r in records if r["type"] == "settle"}
    return [(r, outcomes[r["id"]]) for r in records
            if r["type"] == "forecast" and r["id"] in outcomes]


def pending(records=None, today=None):
    """还没结算的判断，标出是否已过揭晓日。"""
    records = load() if records is None else records
    done = {r["ref"] for r in records if r["type"] == "settle"}
    today = today or date.today()
    out = []
    for r in records:
        if r["type"] == "forecast" and r["id"] not in done:
            y, m, d = map(int, r["resolve_by"].split("-"))
            out.append((r, date(y, m, d) <= today))
    return out


def score(records=None):
    """Brier 分数 + 校准表 + 对照基线的技巧分。

    Brier = 平均((信心度 - 实际结果)²)，越低越好。
    单看它没有意义——0.20 是好是坏取决于你预测的事有多难。所以必须和
    参考模型比：一个只知道基础发生率、对每件事都报同一个数的模型。
    技巧分 BSS = 1 - BS/BS_ref，大于 0 才说明你的判断带来了信息。
    """
    pairs = resolved(records)
    n = len(pairs)
    if n == 0:
        return {"n": 0}

    p = [f["probability"] for f, _ in pairs]
    y = [1.0 if s["outcome"] else 0.0 for _, s in pairs]
    brier = sum((pi - yi) ** 2 for pi, yi in zip(p, y)) / n

    base = sum(y) / n                       # 实际发生率
    ref = sum((base - yi) ** 2 for yi in y) / n
    bss = (1 - brier / ref) if ref > 0 else None

    table = []
    for lo, hi in BUCKETS:
        idx = [i for i in range(n) if lo <= p[i] < hi or (hi == 1.0 and p[i] == 1.0)]
        if idx:
            table.append({
                "range": f"{lo:.0%}–{hi:.0%}", "n": len(idx),
                "said": sum(p[i] for i in idx) / len(idx),
                "actual": sum(y[i] for i in idx) / len(idx),
            })

    return {"n": n, "brier": brier, "base_rate": base,
            "ref_brier": ref, "bss": bss, "table": table}


# ---------- 锚定：与彩票链同样的「摘要写一次即冻结」 ----------

def digest_file(records=None):
    """生成待锚定的摘要文件。已存在就原样复用，绝不重写。

    文件名只由记录数和链头决定，内容里却有 created_at。同一链头第二次
    写入会改变文件字节，之前盖的 .ots 凭证当场失配且不报错——彩票那边
    真实踩过这个坑，这里从一开始就照修好的方式写。
    """
    records = load() if records is None else records
    if not records:
        return None, None

    ANCHOR_DIR.mkdir(parents=True, exist_ok=True)
    path = ANCHOR_DIR / f"chain-{len(records):06d}-{records[-1]['hash'][:12]}.json"
    if path.exists():
        return path, json.loads(path.read_text(encoding="utf-8"))

    doc = {
        "kind": "forecast",
        "head": records[-1]["hash"],
        "count": len(records),
        "chain_digest": hashlib.sha256(
            "\n".join(r["hash"] for r in records).encode()).hexdigest(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")
    return path, doc


def _anchor_log():
    if not ANCHOR_LOG.exists():
        return []
    with ANCHOR_LOG.open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def stamp():
    """把当前链头提交到比特币时间戳服务。链头没变就跳过。"""
    path, doc = digest_file()
    if not path:
        return {"ok": False, "error": "还没有任何判断记录"}
    if any(e["head"] == doc["head"] for e in _anchor_log()):
        return {"ok": True, "skipped": True, "head": doc["head"]}

    code, out = _ots(["stamp", str(path)])
    ots_path = path.with_suffix(path.suffix + ".ots")
    ok = ots_path.exists()
    entry = {
        "created_at": doc["created_at"], "head": doc["head"],
        "count": doc["count"], "chain_digest": doc["chain_digest"],
        "digest_file": path.name, "ots_file": ots_path.name if ok else None,
        "status": "pending" if ok else "failed", "output": out.strip()[:500],
    }
    if ok:
        with ANCHOR_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return {"ok": ok, "code": code, **entry}


def upgrade():
    """把待确认凭证升级为完整的比特币证明（需等区块确认，通常几小时）。"""
    out = []
    for p in sorted(ANCHOR_DIR.glob("*.json.ots")):
        before = p.stat().st_size
        _ots(["upgrade", str(p)])
        out.append({"file": p.name, "upgraded": p.stat().st_size > before})
    return out
