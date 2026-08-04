"""公开预测记录与自动验证，按彩种独立成链。

这个模块的全部意义在于**不可事后修改**。预测在开奖前写入，开奖后自动比对，
输赢全部保留。任何一条记录被改动，哈希链立刻断裂，公开可查。

每个彩种一条独立的哈希链（data/predictions-{key}.jsonl），互不干扰。

注意本地哈希链的能力边界：它能证明"单条记录没被改过"，但防不住
服务器所有者重写整条链。要解决这个，链头必须锚定到外部——见 anchor.py。
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .lotteries import get as get_lottery
from .lotteries import judge
from .predict import ALGORITHMS, predict

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data"
GENESIS = "0" * 64

# 参与公开验证的算法。random 是基线，必须在场——没有基线就无从判断"有效"。
TRACKED = ["mix", "stack", "ml", "markov", "overdue", "hot", "cold", "cycle",
           "random"]


def pred_path(lot):
    return DATA_DIR / f"predictions-{lot.key}.jsonl"


def _hash(rec, prev):
    payload = json.dumps({
        "lottery": rec["lottery"], "issue": rec["issue"], "algo": rec["algo"],
        "front": rec["front"], "back": rec["back"],
        "created_at": rec["created_at"], "based_on": rec["based_on"],
        "prev_hash": prev,
    }, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def load_records(lot=None):
    lot = lot or get_lottery()
    p = pred_path(lot)
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def verify_chain(records=None, lot=None):
    """校验哈希链完整性。返回 (是否完整, 出问题的位置)。"""
    records = load_records(lot) if records is None else records
    prev = GENESIS
    for i, r in enumerate(records):
        if r.get("prev_hash") != prev or _hash(r, prev) != r.get("hash"):
            return False, i
        prev = r["hash"]
    return True, None


def next_issue(df, lot=None):
    """推算下一期期号。

    两类期号要分开处理：
    - 5 位流水号（中国彩种 26085）→ 序号 +1，跨年回到 001
    - 8 位日期（国外彩种 20260729）→ 必须按开奖日程推算下一个开奖日，
      简单 +1 会算出 20260732 这种不存在的日期
    """
    last = str(df.iloc[-1]["issue"])

    if len(last) == 8 and last.isdigit():
        from datetime import date, timedelta

        d = date(int(last[:4]), int(last[4:6]), int(last[6:]))
        days = (lot.draw_days if lot and lot.draw_days else list(range(7)))
        for i in range(1, 15):
            nd = d + timedelta(days=i)
            if nd.weekday() in days:
                return nd.strftime("%Y%m%d")
        return (d + timedelta(days=1)).strftime("%Y%m%d")

    if len(last) == 5 and last.isdigit():
        year, seq = int(last[:2]), int(last[2:])
        return f"{year:02d}{seq + 1:03d}" if seq < 160 else f"{year + 1:02d}001"

    try:
        return str(int(last) + 1).zfill(len(last))
    except ValueError:
        return f"{last}+1"


class DrawAlreadyHeld(Exception):
    """下一期已经开奖，但结果还没抓到——此刻绝不能写预测。"""


def make_predictions(df, lot=None, algos=None, now=None):
    """为下一期生成预测并追加写入。同一期同一算法不重复写。

    开奖时刻已过就拒绝写入，见下方 DrawAlreadyHeld。
    """
    lot = lot or get_lottery()
    algos = algos or TRACKED
    issue = next_issue(df, lot)
    based_on = str(df.iloc[-1]["issue"])
    now = now or datetime.now(timezone.utc).astimezone()

    # 预测必须写在开奖之前，否则这条记录作为证据是零价值的：外部验证者
    # 只看得到"时间戳晚于开奖"，无从区分诚实的抓取延迟和看着结果补写。
    # 而链上记录不可撤销，一条脏记录会永久留在公开仓库里。
    # 所以宁可漏掉一期，也不写一条无法自证清白的记录。
    last_date = df.iloc[-1].get("date")
    if last_date is not None and lot.draw_days:
        draw_at = lot.draw_at(lot.next_draw_date(str(last_date)[:10]))
        if now >= draw_at:
            raise DrawAlreadyHeld(
                f"{issue} 期已于 {draw_at:%F %H:%M %Z} 开奖，"
                f"现在是 {now:%F %H:%M %Z}，晚了 "
                f"{(now - draw_at).total_seconds() / 60:.0f} 分钟。"
                f"拒绝写入——开奖后生成的预测无法自证清白。"
                f"等结果发布后 settle 会补上，下一期照常锁定。"
            )

    records = load_records(lot)
    existing = {(r["issue"], r["algo"]) for r in records}
    prev = records[-1]["hash"] if records else GENESIS

    added = []
    for algo in algos:
        if (issue, algo) in existing:
            continue
        # random 基线用期号做种子：可复现，且无法事后挑一个好看的
        seed = int(issue) if algo == "random" and issue.isdigit() else None
        p = predict(df, lot, algo, seed=seed)
        rec = {
            "lottery": lot.key, "issue": issue, "algo": algo,
            "algo_name": ALGORITHMS[algo][0],
            "front": p["front"], "back": p["back"],
            "created_at": now.isoformat(), "based_on": based_on,
            "prev_hash": prev,
        }
        rec["hash"] = _hash(rec, prev)
        prev = rec["hash"]
        added.append(rec)

    if added:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with pred_path(lot).open("a", encoding="utf-8") as f:
            for r in added:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return issue, added


def evaluate(records, df, lot=None):
    """把已开奖期次的预测与实际结果比对。"""
    lot = lot or get_lottery()
    fc = [f"r{i}" for i in range(1, lot.front_pick + 1)]
    bc = [f"b{i}" for i in range(1, lot.back_pick + 1)]

    actual = {
        str(row["issue"]): ([int(row[c]) for c in fc], [int(row[c]) for c in bc])
        for _, row in df.iterrows()
    }

    out = []
    for r in records:
        if r["issue"] not in actual:
            continue
        af, ab = actual[r["issue"]]
        hf, hb, lv, amt = judge(lot, r["front"], r["back"], af, ab)
        out.append({
            **r, "actual_front": sorted(af), "actual_back": sorted(ab),
            "hit_front": hf, "hit_back": hb, "level": lv,
            "level_name": lot.level_names.get(lv) if lv else None,
            "payout": amt,
        })
    return out


def summarize(evaluated, lot=None):
    """按算法汇总战绩，并与 random 基线做统计比较。"""
    lot = lot or get_lottery()
    by_algo = {}
    for e in evaluated:
        s = by_algo.setdefault(e["algo"], {
            "algo": e["algo"], "name": e.get("algo_name", e["algo"]),
            "n": 0, "cost": 0.0, "payout": 0.0, "hits": [], "levels": {},
        })
        s["n"] += 1
        s["cost"] += lot.price
        s["payout"] += e["payout"]
        s["hits"].append(e["hit_front"] + e["hit_back"])
        if e["level"]:
            key = e.get("level_name") or str(e["level"])
            s["levels"][key] = s["levels"].get(key, 0) + 1

    base = by_algo.get("random", {}).get("hits", [])
    for s in by_algo.values():
        h = np.array(s["hits"], dtype=float)
        s["mean_hits"] = round(float(h.mean()), 4) if len(h) else 0.0
        s["roi"] = round((s["payout"] - s["cost"]) / s["cost"], 4) if s["cost"] else 0.0
        s["net"] = round(s["payout"] - s["cost"], 2)
        s.pop("hits")
        s["vs_random_p"], s["beats_random"] = None, False

        if base and s["algo"] != "random" and len(h) > 1 and len(base) > 1:
            b = np.array(base, dtype=float)
            if h.std() > 0 or b.std() > 0:
                from scipy import stats as st
                _t, p = st.ttest_ind(h, b, equal_var=False)
                s["vs_random_p"] = round(float(p), 4)
                s["beats_random"] = bool(p < 0.05 and h.mean() > b.mean())

    return sorted(by_algo.values(), key=lambda x: -x["mean_hits"])


def status(df, lot=None):
    """网站要的全部数据。"""
    lot = lot or get_lottery()
    records = load_records(lot)
    ok, bad = verify_chain(records, lot)
    evaluated = evaluate(records, df, lot)
    settled_issues = set(df["issue"].astype(str))

    return {
        "lottery": lot.key,
        "lottery_name": lot.name,
        "chain_valid": ok,
        "chain_broken_at": bad,
        "total_predictions": len(records),
        "settled": len(evaluated),
        "pending": [r for r in records if r["issue"] not in settled_issues],
        "summary": summarize(evaluated, lot) if evaluated else [],
        "recent": sorted(evaluated, key=lambda x: x["issue"], reverse=True)[:60],
    }
