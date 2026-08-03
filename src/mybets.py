"""真实投注记录与结算。

和 track.py 的区别：那边记录的是算法的虚拟预测，这边记录**真金白银买了什么**。
两者用同一套哈希链规则，所以可以并排比较——你的实际战绩 vs 各算法 vs 随机基线。

为什么真实投注也要上链：篡改动机在这里更强（事后美化战绩）。
记录一律 append-only，写错了用 void 作废，而不是删除——
"我曾经记错过"本身也是记录的一部分。
"""

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from .lotteries import get as get_lottery
from .lotteries import judge

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data"
GENESIS = "0" * 64


def bets_path(lot):
    return DATA_DIR / f"mybets-{lot.key}.jsonl"


def parse_ticket(text, lot):
    """解析一注号码。

    接受多种写法：
        "7,9,29,33,35+1,11"
        "07 09 29 33 35 + 01 11"
        "7 9 29 33 35 1 11"      （无分隔符，按个数切分）
    """
    text = text.strip()
    if "+" in text:
        fpart, bpart = text.split("+", 1)
        front = [int(x) for x in re.findall(r"\d+", fpart)]
        back = [int(x) for x in re.findall(r"\d+", bpart)]
    else:
        nums = [int(x) for x in re.findall(r"\d+", text)]
        front, back = nums[: lot.front_pick], nums[lot.front_pick:]

    if len(front) != lot.front_pick:
        raise ValueError(f"前区应有 {lot.front_pick} 个号，得到 {len(front)}：{front}")
    if lot.back_pick and len(back) != lot.back_pick:
        raise ValueError(f"后区应有 {lot.back_pick} 个号，得到 {len(back)}：{back}")
    if len(set(front)) != len(front):
        raise ValueError(f"前区号码重复：{front}")
    if not all(1 <= n <= lot.front_max for n in front):
        raise ValueError(f"前区号码须在 1..{lot.front_max}：{front}")
    if not all(1 <= n <= lot.back_max for n in back):
        raise ValueError(f"后区号码须在 1..{lot.back_max}：{back}")

    return sorted(front), sorted(back)


def _hash(rec, prev):
    payload = json.dumps({
        "lottery": rec["lottery"], "issue": rec["issue"],
        "tickets": rec["tickets"], "cost": rec["cost"],
        "created_at": rec["created_at"], "kind": rec.get("kind", "bet"),
        "voids": rec.get("voids"), "prev_hash": prev,
    }, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def load_bets(lot=None):
    lot = lot or get_lottery()
    p = bets_path(lot)
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def verify_chain(records=None, lot=None):
    records = load_bets(lot) if records is None else records
    prev = GENESIS
    for i, r in enumerate(records):
        if r.get("prev_hash") != prev or _hash(r, prev) != r.get("hash"):
            return False, i
        prev = r["hash"]
    return True, None


def _append(lot, rec):
    records = load_bets(lot)
    rec["prev_hash"] = records[-1]["hash"] if records else GENESIS
    rec["hash"] = _hash(rec, rec["prev_hash"])
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with bets_path(lot).open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def add_bet(lot, issue, tickets, cost=None, note="", source=None, now=None):
    """记一次投注。tickets 是 [(front, back), ...]。"""
    lot = lot or get_lottery()
    now = now or datetime.now(timezone.utc).astimezone()

    rec = {
        "kind": "bet",
        "lottery": lot.key,
        "issue": str(issue),
        "tickets": [{"front": list(f), "back": list(b),
                     "source": source or "manual"} for f, b in tickets],
        # 成本默认按注数×单价，允许手动覆盖（打折、复式等情况）
        "cost": float(cost) if cost is not None else len(tickets) * lot.price,
        "note": note,
        "created_at": now.isoformat(),
    }
    return _append(lot, rec)


def void_bet(lot, target_hash, reason="", now=None):
    """作废一条记录——不删除，追加一条作废声明。"""
    lot = lot or get_lottery()
    now = now or datetime.now(timezone.utc).astimezone()
    rec = {
        "kind": "void", "lottery": lot.key, "issue": "",
        "tickets": [], "cost": 0.0, "voids": target_hash,
        "note": reason, "created_at": now.isoformat(),
    }
    return _append(lot, rec)


def evaluate(lot, df, records=None):
    """结算已开奖的投注。"""
    lot = lot or get_lottery()
    records = load_bets(lot) if records is None else records
    voided = {r["voids"] for r in records if r.get("kind") == "void"}

    fc = [f"r{i}" for i in range(1, lot.front_pick + 1)]
    bc = [f"b{i}" for i in range(1, lot.back_pick + 1)]
    actual = {
        str(row["issue"]): ([int(row[c]) for c in fc], [int(row[c]) for c in bc])
        for _, row in df.iterrows()
    }

    out = []
    for r in records:
        if r.get("kind") != "bet" or r["hash"] in voided:
            continue
        if r["issue"] not in actual:
            out.append({**r, "settled": False, "payout": 0.0, "detail": []})
            continue

        af, ab = actual[r["issue"]]
        detail, payout = [], 0.0
        for t in r["tickets"]:
            hf, hb, lv, amt = judge(lot, t["front"], t["back"], af, ab)
            payout += amt
            detail.append({
                **t, "hit_front": hf, "hit_back": hb,
                "level": lv, "level_name": lot.level_names.get(lv) if lv else None,
                "payout": amt,
            })
        out.append({
            **r, "settled": True, "actual_front": sorted(af),
            "actual_back": sorted(ab), "payout": payout,
            "net": round(payout - r["cost"], 2), "detail": detail,
        })
    return out


def summarize(evaluated, lot=None):
    """真实战绩汇总。"""
    lot = lot or get_lottery()
    settled = [e for e in evaluated if e["settled"]]

    cost = sum(e["cost"] for e in settled)
    payout = sum(e["payout"] for e in settled)
    n_tickets = sum(len(e["tickets"]) for e in settled)
    levels = {}
    for e in settled:
        for d in e["detail"]:
            if d["level"]:
                key = d["level_name"] or str(d["level"])
                levels[key] = levels.get(key, 0) + 1

    from .lotteries import expected_value
    ev_per = expected_value(lot)

    return {
        "periods": len(settled),
        "tickets": n_tickets,
        "cost": round(cost, 2),
        "payout": round(payout, 2),
        "net": round(payout - cost, 2),
        "roi": round((payout - cost) / cost, 4) if cost else 0.0,
        # 同样注数下的理论期望——用来判断实际结果是运气好还是运气差
        "expected_payout": round(ev_per * n_tickets, 2),
        "expected_roi": round((ev_per - lot.price) / lot.price, 4),
        "levels": levels,
        "pending": len([e for e in evaluated if not e["settled"]]),
    }


def status(lot, df):
    lot = lot or get_lottery()
    records = load_bets(lot)
    ok, bad = verify_chain(records, lot)
    ev = evaluate(lot, df, records)
    return {
        "lottery": lot.key,
        "chain_valid": ok,
        "chain_broken_at": bad,
        "total_records": len(records),
        "summary": summarize(ev, lot),
        "bets": sorted(ev, key=lambda x: x["issue"], reverse=True)[:50],
    }
