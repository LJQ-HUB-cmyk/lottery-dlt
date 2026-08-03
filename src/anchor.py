"""把哈希链锚定到比特币区块链，取得第三方时间戳证明。

为什么必须上链：
本地哈希链只能证明"单条记录没被改过"，防不住服务器所有者**重写整条链**。
数据全在自己手里的系统无法自证清白——开奖后重新生成整条链，看起来一样完整。

OpenTimestamps 把链头哈希提交到比特币区块链，由全网算力为
"这个哈希在这个时刻已经存在"作证。这份证明任何人都能独立验证，
不需要信任本站。

流程：
  1. stamp  —— 提交链头哈希，得到 .ots 待确认凭证（立即）
  2. upgrade—— 比特币确认后（几小时），凭证升级为完整证明
  3. verify —— 任何人可独立验证时间戳，无需信任本站
"""

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from . import track

BASE = Path(__file__).resolve().parent.parent
def anchor_dir(lot):
    return BASE / "data" / f"anchors-{lot.key}"


def anchor_log(lot):
    return BASE / "data" / f"anchors-{lot.key}.jsonl"


def chain_head(lot, records=None):
    """当前链头：最后一条记录的哈希。它唯一确定了整条链的全部内容。"""
    records = track.load_records(lot) if records is None else records
    return records[-1]["hash"] if records else None


def digest_file(lot, records=None):
    """生成待锚定的摘要文件：链头 + 记录数 + 全链内容哈希。"""
    records = track.load_records(lot) if records is None else records
    if not records:
        return None, None

    full = hashlib.sha256(
        "\n".join(r["hash"] for r in records).encode()
    ).hexdigest()

    doc = {
        "lottery": lot.key,
        "head": records[-1]["hash"],
        "count": len(records),
        "chain_digest": full,
        "last_issue": records[-1]["issue"],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    d = anchor_dir(lot)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"chain-{len(records):06d}-{records[-1]['hash'][:12]}.json"
    path.write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")
    return path, doc


def _ots_bin():
    """找到 ots 可执行文件：优先当前解释器同目录，其次 PATH。"""
    import shutil
    import sys

    local = Path(sys.executable).parent / "ots"
    if local.exists():
        return str(local)
    return shutil.which("ots")


def _ots(args, cwd=None):
    """调用 opentimestamps 客户端。"""
    exe = _ots_bin()
    if not exe:
        return 127, "opentimestamps 客户端未安装（pip install opentimestamps-client）"
    try:
        r = subprocess.run(
            [exe, *args], capture_output=True, text=True, timeout=180, cwd=cwd
        )
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, "调用超时"


def stamp(lot):
    """把当前链头提交到比特币时间戳服务。"""
    path, doc = digest_file(lot)
    if not path:
        return {"ok": False, "error": "还没有任何预测记录"}

    code, out = _ots(["stamp", str(path)])
    ots_path = path.with_suffix(path.suffix + ".ots")
    ok = ots_path.exists()

    entry = {
        "lottery": lot.key,
        "created_at": doc["created_at"],
        "head": doc["head"],
        "count": doc["count"],
        "chain_digest": doc["chain_digest"],
        "digest_file": path.name,
        "ots_file": ots_path.name if ok else None,
        "status": "pending" if ok else "failed",
        "output": out.strip()[:500],
    }
    if ok:
        log = anchor_log(lot)
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return {"ok": ok, "code": code, **entry}


def upgrade(lot):
    """把待确认凭证升级为完整的比特币证明（需等区块确认，通常几小时）。"""
    results = []
    for ots in sorted(anchor_dir(lot).glob("*.ots")):
        code, out = _ots(["upgrade", str(ots)])
        results.append({
            "file": ots.name,
            "upgraded": "Success" in out or code == 0,
            "output": out.strip()[:200],
        })
    return results


def verify(lot, ots_name=None):
    """验证时间戳证明。任何人都可以跑这个，不需要信任本站。"""
    d = anchor_dir(lot)
    files = [d / ots_name] if ots_name else sorted(d.glob("*.ots"))
    out = []
    for f in files:
        if not f.exists():
            out.append({"file": f.name, "error": "文件不存在"})
            continue
        code, txt = _ots(["verify", str(f)])
        out.append({
            "file": f.name,
            "verified": "Success!" in txt or "attests" in txt.lower(),
            "output": txt.strip()[:300],
        })
    return out


def load_anchors(lot):
    log = anchor_log(lot)
    if not log.exists():
        return []
    with log.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def status(lot):
    """锚定状态，供网站展示。"""
    anchors = load_anchors(lot)
    records = track.load_records(lot)
    head = chain_head(lot, records)
    anchored_heads = {a["head"] for a in anchors}

    return {
        "lottery": lot.key,
        "anchors": anchors,
        "total": len(anchors),
        "current_head": head,
        "current_count": len(records),
        "head_anchored": head in anchored_heads if head else False,
        "unanchored": len(records) - max(
            [a["count"] for a in anchors], default=0
        ),
    }
