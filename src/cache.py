"""磁盘缓存：把重算成本高、结果又不常变的东西存下来。

关键洞察：预测结果只在**开奖后**才会变。同一期内，无论多少人访问，
答案完全一样。所以用"彩种 + 最新期号 + 参数"做键，一期只算一次。

不用 Redis 是为了少一个部署依赖；数据量小（每期几十 KB），
磁盘 JSON 完全够用。真到需要横向扩容时再换。
"""

import hashlib
import json
import os
import threading
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
CACHE_DIR = Path(os.environ.get("CACHE_DIR", BASE / "data" / "cache"))
DEFAULT_TTL = int(os.environ.get("CACHE_TTL", 86400 * 7))

_lock = threading.Lock()
_mem = {}
_MEM_MAX = 256


def _key(namespace, parts):
    raw = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return f"{namespace}-{hashlib.sha1(raw.encode()).hexdigest()[:20]}"


def get_or_compute(namespace, parts, fn, ttl=DEFAULT_TTL):
    """取缓存，没有就算并写入。

    parts 必须包含所有影响结果的因素——尤其是最新期号，
    否则开奖后会返回过期结果。
    """
    k = _key(namespace, parts)

    with _lock:
        hit = _mem.get(k)
    if hit and (ttl <= 0 or time.time() - hit[0] < ttl):
        return hit[1]

    path = CACHE_DIR / f"{k}.json"
    if path.exists():
        age = time.time() - path.stat().st_mtime
        if ttl <= 0 or age < ttl:
            try:
                val = json.loads(path.read_text(encoding="utf-8"))
                _remember(k, val)
                return val
            except (json.JSONDecodeError, OSError):
                pass  # 缓存损坏，重算

    val = fn()

    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(val, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)  # 原子写，避免并发读到半截文件
    except OSError:
        pass  # 磁盘不可写不应该让请求失败

    _remember(k, val)
    return val


def _remember(k, val):
    with _lock:
        if len(_mem) >= _MEM_MAX:
            oldest = min(_mem, key=lambda x: _mem[x][0])
            _mem.pop(oldest, None)
        _mem[k] = (time.time(), val)


def clear(namespace=None):
    """清缓存。开奖数据更新后应当调用。"""
    with _lock:
        if namespace:
            for k in [k for k in _mem if k.startswith(f"{namespace}-")]:
                _mem.pop(k, None)
        else:
            _mem.clear()

    if not CACHE_DIR.exists():
        return 0
    pattern = f"{namespace}-*.json" if namespace else "*.json"
    n = 0
    for f in CACHE_DIR.glob(pattern):
        try:
            f.unlink()
            n += 1
        except OSError:
            pass
    return n


def stats():
    files = list(CACHE_DIR.glob("*.json")) if CACHE_DIR.exists() else []
    return {
        "entries": len(files),
        "memory_entries": len(_mem),
        "size_kb": round(sum(f.stat().st_size for f in files) / 1024, 1),
        "dir": str(CACHE_DIR),
    }
