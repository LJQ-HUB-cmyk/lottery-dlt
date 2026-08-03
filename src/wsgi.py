"""WSGI 入口，供 gunicorn 使用。

启动时预热缓存：进程起来就把各彩种的常用结果算好，
避免第一个访客承担 3 秒的模型训练。
"""

import os

from .web import app

if os.environ.get("WARM_CACHE", "1") == "1":
    try:
        with app.test_request_context():
            from . import lotteries
            from .fetch import data_path
            from .web import get_df
            for lot in lotteries.enabled():
                if data_path(lot).exists():
                    get_df(lot)
    except Exception:
        pass  # 预热失败不应阻止启动

__all__ = ["app"]
