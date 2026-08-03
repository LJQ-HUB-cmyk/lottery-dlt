"""数据抓取：每个彩种注册一个抓取器，统一输出格式。

统一 schema：issue, date, r1..rN, b1..bM, sales, pool

加新彩种 = 写一个 @register("key") 的函数，返回上述格式的记录列表。
"""

import csv
import re
import sys
import time
from pathlib import Path

import requests
from lxml import html

from .lotteries import get as get_lottery

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
}

FETCHERS = {}


class ParseError(Exception):
    pass


def register(key):
    def deco(fn):
        FETCHERS[key] = fn
        return fn
    return deco


def data_path(lot):
    return DATA_DIR / f"{lot.key}_history.csv"


def fields(lot):
    return (["issue", "date"]
            + [f"r{i}" for i in range(1, lot.front_pick + 1)]
            + [f"b{i}" for i in range(1, lot.back_pick + 1)]
            + ["sales", "pool"])


def _to_int(text):
    digits = re.sub(r"[^\d]", "", text or "")
    return int(digits) if digits else 0


def _get(url, params, referer, retries=4):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params,
                             headers={**HEADERS, "Referer": referer}, timeout=30)
            r.raise_for_status()
            return r
        except (requests.exceptions.SSLError, requests.exceptions.Timeout):
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))


# ---------- 500.com 系列（大乐透 / 双色球）----------


def _parse_500(resp, lot):
    """500.com 的历史开奖表。列数会随查询范围浮动，
    因此以期号列为前锚、日期列为后锚做相对定位。"""
    tree = html.fromstring(resp.content.decode("gb18030", errors="replace"))
    tbody = tree.xpath('//tbody[@id="tdata"]')
    if not tbody:
        raise ParseError("页面结构变化：未找到 tbody#tdata，可能被反爬拦截")

    nf, nb = lot.front_pick, lot.back_pick
    records = []
    for tr in tbody[0].xpath("./tr"):
        cells = ["".join(td.itertext()).strip() for td in tr.xpath("./td")]
        if len(cells) < nf + nb + 3:
            continue

        idx = next((i for i, c in enumerate(cells[:2]) if re.fullmatch(r"\d{5}", c)), None)
        if idx is None:
            raise ParseError(f"未能定位期号列: {cells[:2]!r}")

        issue = cells[idx]
        front = [int(c) for c in cells[idx + 1: idx + 1 + nf]]
        back = [int(c) for c in cells[idx + 1 + nf: idx + 1 + nf + nb]]

        if len(set(front)) != nf or not all(1 <= n <= lot.front_max for n in front):
            raise ParseError(f"期 {issue}: 前区号码非法 {front}")
        if len(set(back)) != nb or not all(1 <= n <= lot.back_max for n in back):
            raise ParseError(f"期 {issue}: 后区号码非法 {back}")

        date = cells[-1]
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
            raise ParseError(f"期 {issue}: 日期格式非法 {date!r}")

        rec = {"issue": issue, "date": date}
        rec.update({f"r{i + 1}": n for i, n in enumerate(sorted(front))})
        rec.update({f"b{i + 1}": n for i, n in enumerate(sorted(back))})
        rec["sales"] = _to_int(cells[-2])
        rec["pool"] = _to_int(cells[-7]) if len(cells) >= 7 else 0
        records.append(rec)

    records.sort(key=lambda r: r["issue"])
    return records


@register("dlt")
def fetch_dlt(lot, start="07001", end="99999"):
    resp = _get("https://datachart.500.com/dlt/history/newinc/history.php",
                {"start": start, "end": end},
                "https://datachart.500.com/dlt/history/history.shtml")
    return _parse_500(resp, lot)


@register("ssq")
def fetch_ssq(lot, start="03001", end="99999"):
    resp = _get("https://datachart.500.com/ssq/history/newinc/history.php",
                {"start": start, "end": end},
                "https://datachart.500.com/ssq/history/history.shtml")
    return _parse_500(resp, lot)


# ---------- Powerball（纽约州开放数据，官方）----------


@register("powerball")
def fetch_powerball(lot, start=None, **kw):
    """data.ny.gov 的官方开放数据集，JSON。

    规则变更：2015-10-07 起由 5/59+1/35 改为 5/69+1/26。
    早于该日期的记录号码范围不同，直接丢弃——混用会产生超范围号码。
    """
    url = "https://data.ny.gov/resource/d6yy-54nr.json"
    records, offset = [], 0

    while True:
        r = requests.get(url, params={"$limit": 1000, "$offset": offset,
                                      "$order": "draw_date ASC"},
                         headers=HEADERS, timeout=45)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break

        for x in batch:
            date = x["draw_date"][:10]
            if date < lot.rules_from:
                continue
            nums = [int(n) for n in x["winning_numbers"].split()]
            if len(nums) != lot.front_pick + lot.back_pick:
                continue
            front, back = sorted(nums[:lot.front_pick]), nums[lot.front_pick:]
            if not all(1 <= n <= lot.front_max for n in front):
                continue
            if not all(1 <= n <= lot.back_max for n in back):
                continue

            rec = {"issue": date.replace("-", ""), "date": date}
            rec.update({f"r{i + 1}": n for i, n in enumerate(front)})
            rec.update({f"b{i + 1}": n for i, n in enumerate(sorted(back))})
            rec["sales"] = 0
            rec["pool"] = 0
            records.append(rec)

        offset += len(batch)
        if len(batch) < 1000:
            break

    records.sort(key=lambda r: r["issue"])
    return records


# ---------- EuroMillions（euro-millions.com 按年历史）----------


@register("euromillions")
def fetch_euromillions(lot, start=None, **kw):
    """按年抓取历史页。

    规则变更：幸运星 2011-05 由 1..9 增至 1..11，2016-09 增至 1..12。
    早于 rules_from 的记录直接丢弃。
    """
    from datetime import date as _date

    start_year = int(lot.rules_from[:4])
    this_year = _date.today().year
    records = []

    for year in range(start_year, this_year + 1):
        resp = _get(f"https://www.euro-millions.com/results-history-{year}",
                    None, "https://www.euro-millions.com/")
        tree = html.fromstring(resp.content)

        for tr in tree.xpath("//table//tr"):
            tds = tr.xpath("./td")
            if len(tds) < 2:
                continue

            date_txt = " ".join("".join(tds[0].itertext()).split())
            m = re.search(r"(\d{1,2})\w*\s+(\w+)\s+(\d{4})", date_txt)
            if not m:
                continue
            try:
                from datetime import datetime as _dt
                d = _dt.strptime(f"{m.group(1)} {m.group(2)} {m.group(3)}",
                                 "%d %B %Y").date()
            except ValueError:
                continue
            if d.isoformat() < lot.rules_from:
                continue

            nums = [int(x) for x in
                    "".join(tds[1].itertext()).split() if x.isdigit()]
            need = lot.front_pick + lot.back_pick
            if len(nums) < need:
                continue
            front = sorted(nums[:lot.front_pick])
            back = sorted(nums[lot.front_pick:need])

            if not all(1 <= n <= lot.front_max for n in front):
                continue
            if not all(1 <= n <= lot.back_max for n in back):
                continue

            rec = {"issue": d.isoformat().replace("-", ""), "date": d.isoformat()}
            rec.update({f"r{i + 1}": n for i, n in enumerate(front)})
            rec.update({f"b{i + 1}": n for i, n in enumerate(back)})
            rec["sales"] = 0
            rec["pool"] = _to_int(
                "".join(tds[3].itertext()) if len(tds) > 3 else "")
            records.append(rec)

        time.sleep(0.6)  # 对站点友好

    records.sort(key=lambda r: r["issue"])
    return records


# ---------- 通用读写 ----------


def load_local(lot):
    p = data_path(lot)
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save(lot, records):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with data_path(lot).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields(lot))
        w.writeheader()
        w.writerows(records)


def update(lot=None):
    lot = lot or get_lottery()
    if lot.key not in FETCHERS:
        raise SystemExit(f"{lot.name} 尚未接入数据源")

    existing = load_local(lot)
    kwargs = {"start": existing[-1]["issue"]} if existing else {}

    print(f"抓取 {lot.name} {kwargs.get('start', '起始')} 至今...", file=sys.stderr)
    fetched = FETCHERS[lot.key](lot, **kwargs)

    by_issue = {r["issue"]: r for r in existing}
    added = sum(1 for r in fetched if r["issue"] not in by_issue)
    by_issue.update({r["issue"]: r for r in fetched})

    merged = [by_issue[k] for k in sorted(by_issue)]
    save(lot, merged)
    return len(merged), added


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--lottery", default="dlt")
    a = ap.parse_args()

    lot = get_lottery(a.lottery)
    total, added = update(lot)
    print(f"完成：{lot.name} 共 {total} 期，新增 {added} 期 → {data_path(lot)}")
