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
from math import comb
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


def _null_pmf(lot):
    """单注预测的命中数分布，在"算法毫无技巧"假设下精确成立。

    随便选 n 个号，开奖开出 n 个号，命中数服从超几何分布——这是组合学
    事实，不需要估计。前区后区独立，卷积得到总命中数的分布。

    用精确分布而不是拿一个随机基线做 t 检验：基线自己只有几期样本，
    噪声比信号大得多，等于拿一把抖动的尺子去量另一把。
    """
    def hyper(n_max, n_pick):
        tot = comb(n_max, n_pick)
        return np.array([comb(n_pick, k) * comb(n_max - n_pick, n_pick - k) / tot
                         for k in range(n_pick + 1)])

    pmf = hyper(lot.front_max, lot.front_pick)
    if lot.back_pick:
        pmf = np.convolve(pmf, hyper(lot.back_max, lot.back_pick))
    return pmf


def _conv_power(pmf, n):
    """n 注独立预测的总命中数分布。二分幂，O(log n) 次卷积。"""
    out, base = np.array([1.0]), pmf
    while n:
        if n & 1:
            out = np.convolve(out, base)
        n >>= 1
        if n:
            base = np.convolve(base, base)
    return out


def _exact_p(lot, hits):
    """单侧精确检验：这么好的成绩，纯靠运气能有多大概率达到？"""
    total = _conv_power(_null_pmf(lot), len(hits))
    obs = int(round(sum(hits)))
    return float(total[min(obs, len(total) - 1):].sum())


def _holm(pvals):
    """Holm–Bonferroni 校正。

    同时检验 m 个算法，每个都用 p<0.05，那么"至少一个纯靠运气蒙到显著"
    的概率是 1-0.95^m——9 个算法就是 37%。这台机器存在的唯一理由是不
    说谎，所以必须控制族错误率，而不是单次错误率。

    Holm 是逐步下降版的 Bonferroni：一致地更强，且同样严格控制 FWER。
    """
    m = len(pvals)
    adj, running = [1.0] * m, 0.0
    for rank, i in enumerate(sorted(range(m), key=lambda i: pvals[i])):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def summarize(evaluated, lot=None):
    """按算法汇总战绩，并与"无技巧"的精确零分布做统计比较。"""
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

    pmf = _null_pmf(lot)
    expected = float(np.dot(np.arange(len(pmf)), pmf))

    # random 也一起检验。它是对照组，不是豁免对象——如果连它都被标成
    # "显著优于随机"，那说明检验本身坏了。
    tested = [s for s in by_algo.values() if s["n"] > 0]
    raw = [_exact_p(lot, s["hits"]) for s in tested]
    adj = _holm(raw) if raw else []

    for s, p, pa in zip(tested, raw, adj):
        h = np.array(s["hits"], dtype=float)
        s["mean_hits"] = round(float(h.mean()), 4)
        s["expected_hits"] = round(expected, 4)
        s["p_value"] = round(p, 4)
        s["p_adjusted"] = round(pa, 4)
        s["beats_random"] = bool(pa < 0.05 and h.mean() > expected)

    for s in by_algo.values():
        s["roi"] = round((s["payout"] - s["cost"]) / s["cost"], 4) if s["cost"] else 0.0
        s["net"] = round(s["payout"] - s["cost"], 2)
        s.pop("hits")
        s.setdefault("mean_hits", 0.0)
        s.setdefault("expected_hits", round(expected, 4))
        s.setdefault("p_value", None)
        s.setdefault("p_adjusted", None)
        s.setdefault("beats_random", False)

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
