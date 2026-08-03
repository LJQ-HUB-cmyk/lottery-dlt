"""Flask 网页界面，多彩种。

启动：python -m src.web    浏览器打开 http://127.0.0.1:5001
"""

import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from . import anchor, cache, lotteries, stats, track, wheel
from .analyze import exact_expected_value, trigger_probability
from .fetch import data_path
from .markov import load
from .predict import ALGORITHMS, explain, predict_multi
from .prize import guaranteed_level

BASE = Path(__file__).resolve().parent.parent
app = Flask(__name__, template_folder=str(BASE / "templates"),
            static_folder=str(BASE / "static"))

_cache = {}


def get_lot():
    key = request.args.get("lottery") or lotteries.DEFAULT
    if key not in lotteries.LOTTERIES:
        key = lotteries.DEFAULT
    return lotteries.get(key)


def get_df(lot):
    p = data_path(lot)
    if not p.exists():
        return None
    stamp = p.stat().st_mtime
    if _cache.get(lot.key, (None, None))[0] != stamp:
        _cache[lot.key] = (stamp, load(p, lot))
    return _cache[lot.key][1]


def latest_issue(df):
    """最新期号——作为缓存键的一部分，开奖后自动让旧缓存失效。"""
    return str(df.iloc[-1]["issue"])


def need_df(lot):
    df = get_df(lot)
    if df is None:
        return None, (jsonify({"error": f"{lot.name} 数据未抓取"}), 404)
    return df, None


@app.route("/")
def index():
    lot = get_lot()
    df = get_df(lot)
    return render_template(
        "index.html",
        overview=stats.overview(df, lot) if df is not None else None,
        algorithms=[{"key": k, "name": v[0]} for k, v in ALGORITHMS.items()],
        lotteries=[{"key": x.key, "name": x.name, "country": x.country}
                   for x in lotteries.enabled()],
        current=lot.key,
    )


@app.route("/health")
def health():
    """健康检查：确认数据可读、哈希链完整。"""
    from . import track
    out = {"status": "ok", "lotteries": {}}
    for x in lotteries.enabled():
        p = data_path(x)
        ok, bad = track.verify_chain(lot=x)
        out["lotteries"][x.key] = {"data": p.exists(), "chain_valid": ok}
        if not ok:
            out["status"] = "degraded"
            out["lotteries"][x.key]["chain_broken_at"] = bad
    return jsonify(out), (200 if out["status"] == "ok" else 503)


@app.route("/api/lotteries")
def api_lotteries():
    out = []
    for x in lotteries.LOTTERIES.values():
        ev = lotteries.expected_value(x)
        out.append({
            "key": x.key, "name": x.name, "country": x.country,
            "rule": f"{x.front_pick}/{x.front_max}"
                    + (f"+{x.back_pick}/{x.back_max}" if x.back_pick else ""),
            "odds": x.total_combinations, "price": x.price,
            "currency": x.currency, "expected": round(ev, 4),
            "roi": round((ev - x.price) / x.price, 4),
            "enabled": x.enabled, "has_data": data_path(x).exists(),
        })
    return jsonify(out)


@app.route("/api/predict")
def api_predict():
    lot = get_lot()
    df, err = need_df(lot)
    if err:
        return err
    algo = request.args.get("algo", "mix")
    if algo not in ALGORITHMS:
        return jsonify({"error": "未知算法"}), 400

    sets = int(request.args.get("sets", 5))
    seed = request.args.get("seed", type=int)

    def compute():
        r = predict_multi(df, lot, algo, n_sets=sets, seed=seed)
        r["sets"] = [{"front": f, "back": b} for f, b in r["sets"]]
        return r

    # 期号进缓存键：开奖后自动失效，同一期内所有人拿到同一结果
    return jsonify(cache.get_or_compute(
        "predict", [lot.key, algo, sets, seed, latest_issue(df)], compute))


@app.route("/api/explain")
def api_explain():
    lot = get_lot()
    df, err = need_df(lot)
    if err:
        return err
    algo = request.args.get("algo", "mix")
    if algo not in ALGORITHMS:
        return jsonify({"error": "未知算法"}), 400
    zone = request.args.get("zone", "front")

    return jsonify(cache.get_or_compute(
        "explain", [lot.key, algo, zone, latest_issue(df)],
        lambda: explain(df, lot, zone, algo)))


@app.route("/api/track")
def api_track():
    lot = get_lot()
    df, err = need_df(lot)
    if err:
        return err
    st = track.status(df, lot)
    st["anchor"] = anchor.status(lot)
    return jsonify(st)


@app.route("/api/stats/<kind>")
def api_stats(kind):
    lot = get_lot()
    df, err = need_df(lot)
    if err:
        return err
    window = request.args.get("window", type=int)
    zone = request.args.get("zone", "front")

    handlers = {
        "frequency": lambda: stats.frequency(df, lot, zone, window),
        "missing": lambda: stats.missing(df, lot, zone),
        "sum": lambda: stats.sum_trend(df, lot, window or 100),
        "ratio": lambda: stats.ratio_distribution(df, lot, window),
        "zone": lambda: stats.zone_distribution(df, lot, window),
        "recent": lambda: stats.recent_draws(df, lot, window or 30),
        "overview": lambda: stats.overview(df, lot),
    }
    if kind not in handlers:
        return jsonify({"error": "未知类型"}), 400
    return jsonify(cache.get_or_compute(
        "stats", [lot.key, kind, zone, window, latest_issue(df)], handlers[kind]))


@app.route("/api/wheel")
def api_wheel():
    lot = get_lot()
    v = request.args.get("pool", 9, type=int)
    t = request.args.get("guarantee", 3, type=int)
    nb = request.args.get("back", 3, type=int)
    k = lot.front_pick

    if not k <= v <= 14:
        return jsonify({"error": f"号码池请设在 {k}..14（更大规模求解很慢）"}), 400
    if not 1 <= t <= k:
        return jsonify({"error": f"保证命中数须在 1..{k}"}), 400
    if lot.back_pick and not lot.back_pick <= nb <= lot.back_max:
        return jsonify({"error": f"后区号码池须在 {lot.back_pick}..{lot.back_max}"}), 400

    # wheel 只依赖 (v,k,t)，与开奖数据无关，可以长期缓存
    r = cache.get_or_compute("wheel", [v, k, t],
                             lambda: wheel.solve(v, k, t, seeds=2), ttl=0)
    from math import comb

    n_back = comb(nb, lot.back_pick) if lot.back_pick else 1
    total = r["size"] * n_back
    _, _, p_joint = trigger_probability(v, t, nb)
    ev, _ = exact_expected_value()
    level, amount = guaranteed_level(t, lot.back_pick)

    return jsonify({
        "lottery": lot.key, "pool": v, "guarantee": t, "back_pool": nb,
        "lower_bound": r["lower_bound"],
        "front_blocks": [[int(x) + 1 for x in b] for b in r["best"]],
        "front_count": r["size"], "back_combos": n_back,
        "total_tickets": total, "cost": round(total * lot.price, 2),
        "trigger_prob": p_joint,
        "trigger_every": round(1 / p_joint) if p_joint else None,
        "level": lot.level_names.get(level), "level_amount": amount,
        "expected": round(ev * total, 2),
        "roi": round((ev * total - total * lot.price) / (total * lot.price), 4),
    })


def main():
    port = int(os.environ.get("PORT", 5001))  # macOS 的 AirPlay 占用 5000
    print(f"彩票预测公开验证 → http://127.0.0.1:{port}")
    app.run(debug=False, port=port, host="127.0.0.1")


if __name__ == "__main__":
    main()
