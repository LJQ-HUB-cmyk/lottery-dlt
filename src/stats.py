"""走势与统计数据，配置驱动，供网页展示。"""

import numpy as np

from .lotteries import get as get_lottery
from .predict import gaps, to_matrix, zone_spec


def _front_cols(lot):
    return [f"r{i}" for i in range(1, lot.front_pick + 1)]


def _back_cols(lot):
    return [f"b{i}" for i in range(1, lot.back_pick + 1)]


def frequency(df, lot=None, zone="front", window=None):
    lot = lot or get_lottery()
    m = to_matrix(df, lot, zone)
    if window:
        m = m[-window:]
    return [{"number": i + 1, "count": int(c)} for i, c in enumerate(m.sum(axis=0))]


def missing(df, lot=None, zone="front"):
    lot = lot or get_lottery()
    m = to_matrix(df, lot, zone)
    g = gaps(m)
    out = []
    for j in range(m.shape[1]):
        hits = np.flatnonzero(m[:, j])
        mean_gap = float(np.diff(hits).mean()) if len(hits) > 1 else float(len(m))
        out.append({
            "number": j + 1,
            "gap": int(g[j]),
            "mean_gap": round(mean_gap, 1),
            "ratio": round(g[j] / mean_gap, 2) if mean_gap else 0.0,
        })
    return out


def sum_trend(df, lot=None, periods=100):
    lot = lot or get_lottery()
    cols = _front_cols(lot)
    return [
        {"issue": r["issue"], "date": r["date"],
         "sum": int(sum(r[c] for c in cols))}
        for _, r in df.tail(periods).iterrows()
    ]


def ratio_distribution(df, lot=None, periods=None):
    lot = lot or get_lottery()
    cols = _front_cols(lot)
    d = df.tail(periods) if periods else df
    k = lot.front_pick
    odd, big = {}, {}
    for _, r in d.iterrows():
        front = [r[c] for c in cols]
        o = sum(1 for x in front if x % 2)
        b = sum(1 for x in front if x > lot.front_max / 2)
        odd[o] = odd.get(o, 0) + 1
        big[b] = big.get(b, 0) + 1
    return {
        "odd": [{"label": f"{v}奇{k - v}偶", "count": c} for v, c in sorted(odd.items())],
        "big": [{"label": f"{v}大{k - v}小", "count": c} for v, c in sorted(big.items())],
    }


def zone_distribution(df, lot=None, periods=None):
    lot = lot or get_lottery()
    cols = _front_cols(lot)
    d = df.tail(periods) if periods else df
    counts = [0] * (len(lot.zones) + 1)
    for _, r in d.iterrows():
        for c in cols:
            counts[lot.zone_of(r[c])] += 1

    labels, lo = [], 1
    for edge in lot.zones:
        labels.append(f"{lo}-{edge}")
        lo = edge + 1
    labels.append(f"{lo}-{lot.front_max}")
    return [{"label": lb, "count": c} for lb, c in zip(labels, counts)]


def recent_draws(df, lot=None, n=30):
    lot = lot or get_lottery()
    fc, bc = _front_cols(lot), _back_cols(lot)
    return [
        {
            "issue": r["issue"], "date": r["date"],
            "front": [int(r[c]) for c in fc],
            "back": [int(r[c]) for c in bc],
            "sum": int(sum(r[c] for c in fc)),
            "pool": int(r["pool"]) if "pool" in r and r["pool"] else 0,
        }
        for _, r in df.tail(n).iloc[::-1].iterrows()
    ]


def overview(df, lot=None):
    lot = lot or get_lottery()
    last = df.iloc[-1]
    from .lotteries import expected_value
    ev = expected_value(lot)
    return {
        "lottery": lot.key,
        "lottery_name": lot.name,
        "country": lot.country,
        "rule": f"{lot.front_pick}/{lot.front_max}"
                + (f"+{lot.back_pick}/{lot.back_max}" if lot.back_pick else ""),
        "total_periods": len(df),
        "latest_issue": last["issue"],
        "latest_date": last["date"],
        "latest_front": [int(last[c]) for c in _front_cols(lot)],
        "latest_back": [int(last[c]) for c in _back_cols(lot)],
        "first_date": df.iloc[0]["date"],
        "jackpot_odds": lot.total_combinations,
        "expected_value": round(ev, 4),
        "roi": round((ev - lot.price) / lot.price, 4),
        "currency": lot.currency,
        "price": lot.price,
    }
