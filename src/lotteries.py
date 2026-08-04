"""彩种配置：所有与具体彩票相关的规则集中在这里。

加一个新国家的彩票 = 在 LOTTERIES 里加一条配置 + 写一个抓取函数，
其余模块（预测、统计、验证、上链）全部配置驱动，不需要改动。

奖级表的键是 (前区命中数, 后区命中数)，值是 (奖级序号, 奖金)。
浮动奖用历史量级的近似值——精确金额随奖池波动，不影响相对结论。
"""

from dataclasses import dataclass, field
from datetime import date as _date
from datetime import datetime as _datetime
from datetime import timedelta
from zoneinfo import ZoneInfo


def _as_date(d):
    if isinstance(d, str):
        y, m, dd = map(int, d[:10].split("-"))
        return _date(y, m, dd)
    return d


@dataclass
class Lottery:
    key: str
    name: str
    country: str
    currency: str
    front_max: int
    front_pick: int
    back_max: int
    back_pick: int
    price: float
    prize_table: dict
    level_names: dict
    draw_days: list          # 0=周一 … 6=周日
    draw_time: str
    source: str
    enabled: bool = True
    notes: str = ""
    # draw_time 所属时区。必须显式写明：判断"预测是否早于开奖"要比较绝对
    # 时刻，用本机时区去解释 22:59 ET 会差出十几个小时。
    draw_tz: str = "Asia/Shanghai"
    # 当前规则的生效日期；早于此日期的历史数据规则不同，必须丢弃
    rules_from: str = "1900-01-01"
    # 号码分区边界，用于走势图的区间统计
    zones: list = field(default_factory=list)

    @property
    def total_combinations(self):
        from math import comb
        n = comb(self.front_max, self.front_pick)
        if self.back_pick:
            n *= comb(self.back_max, self.back_pick)
        return n

    @property
    def jackpot_odds(self):
        return self.total_combinations

    def zone_of(self, n):
        for i, edge in enumerate(self.zones):
            if n <= edge:
                return i
        return len(self.zones)

    def draw_at(self, d):
        """某个开奖日的绝对开奖时刻（带时区）。d 可以是 date 或 'YYYY-MM-DD'。

        用 ZoneInfo 而不是固定偏移，夏令时会自动跟随——Powerball 的
        ET 和 EuroMillions 的 CET 一年里都要切换两次。
        """
        d = _as_date(d)
        hh, mm = map(int, self.draw_time.split()[0].split(":"))
        return _datetime(d.year, d.month, d.day, hh, mm,
                         tzinfo=ZoneInfo(self.draw_tz))

    def next_draw_date(self, after):
        """给定上一期开奖日，按开奖日程推算下一个开奖日。"""
        after = _as_date(after)
        days = self.draw_days or list(range(7))
        for i in range(1, 15):
            nd = after + timedelta(days=i)
            if nd.weekday() in days:
                return nd
        return after + timedelta(days=1)


DLT = Lottery(
    key="dlt", name="超级大乐透", country="CN", currency="CNY",
    front_max=35, front_pick=5, back_max=12, back_pick=2, price=2,
    prize_table={
        (5, 2): (1, 6_950_000), (5, 1): (2, 139_000), (5, 0): (3, 10_000),
        (4, 2): (4, 3_000), (4, 1): (5, 300), (3, 2): (6, 200),
        (4, 0): (7, 100), (3, 1): (8, 15), (2, 2): (8, 15),
        (3, 0): (9, 5), (1, 2): (9, 5), (2, 1): (9, 5), (0, 2): (9, 5),
    },
    level_names={1: "一等奖", 2: "二等奖", 3: "三等奖", 4: "四等奖", 5: "五等奖",
                 6: "六等奖", 7: "七等奖", 8: "八等奖", 9: "九等奖"},
    draw_days=[0, 2, 5], draw_time="20:30",
    source="datachart.500.com", zones=[12, 24],
    notes="一、二等奖为浮动奖；金额按法定返奖率 50% 反推校准"
          "（依据票面公益金 5.40/15=36%，加发行费 14%）",
)

SSQ = Lottery(
    key="ssq", name="双色球", country="CN", currency="CNY",
    front_max=33, front_pick=6, back_max=16, back_pick=1, price=2,
    prize_table={
        (6, 1): (1, 5_000_000), (6, 0): (2, 150_000), (5, 1): (3, 3_000),
        (5, 0): (4, 200), (4, 1): (4, 200), (4, 0): (5, 10), (3, 1): (5, 10),
        (2, 1): (6, 5), (1, 1): (6, 5), (0, 1): (6, 5),
    },
    level_names={1: "一等奖", 2: "二等奖", 3: "三等奖", 4: "四等奖",
                 5: "五等奖", 6: "六等奖"},
    draw_days=[1, 3, 6], draw_time="21:15",
    source="datachart.500.com", zones=[11, 22],
)

POWERBALL = Lottery(
    key="powerball", name="Powerball", country="US", currency="USD",
    front_max=69, front_pick=5, back_max=26, back_pick=1, price=2,
    prize_table={
        (5, 1): (1, 20_000_000), (5, 0): (2, 1_000_000), (4, 1): (3, 50_000),
        (4, 0): (4, 100), (3, 1): (4, 100), (3, 0): (5, 7), (2, 1): (5, 7),
        (1, 1): (6, 4), (0, 1): (6, 4),
    },
    level_names={1: "Jackpot", 2: "Match 5", 3: "Match 4+PB", 4: "Match 4",
                 5: "Match 3", 6: "Match 1+PB"},
    draw_days=[0, 2, 5], draw_time="22:59 ET", draw_tz="America/New_York",
    # 暂停：开奖窗口落在本机时间上午 10:59，Mac 睡眠时容易漏掉锁定。
    # 已有的链和数据都保留，改回 True 即可继续，settle 会自动补齐。
    enabled=False,
    source="data.ny.gov（纽约州官方开放数据）", zones=[23, 46],
    rules_from="2015-10-07",
    notes="2015-10-07 起改为 5/69+1/26（此前为 5/59+1/35），早期数据已剔除；头奖为浮动累积奖",
)

EUROMILLIONS = Lottery(
    key="euromillions", name="EuroMillions", country="EU", currency="EUR",
    front_max=50, front_pick=5, back_max=12, back_pick=2, price=2.5,
    prize_table={
        (5, 2): (1, 50_000_000), (5, 1): (2, 300_000), (5, 0): (3, 30_000),
        (4, 2): (4, 1_500), (4, 1): (5, 150), (3, 2): (6, 80),
        (4, 0): (7, 50), (2, 2): (8, 20), (3, 1): (9, 15),
        (3, 0): (10, 12), (1, 2): (11, 10), (2, 1): (12, 8), (2, 0): (13, 4),
    },
    level_names={i: f"Tier {i}" for i in range(1, 14)},
    draw_days=[1, 4], draw_time="20:45 CET", draw_tz="Europe/Paris",
    # 暂停：开奖窗口落在本机时间凌晨 02:45，Mac 必然在睡眠。同上，可随时改回。
    enabled=False,
    source="euro-millions.com", zones=[17, 34],
    rules_from="2016-09-24",
    notes="2016-09-24 起幸运星增至 1..12（此前为 1..11 / 1..9），早期数据已剔除；全部为浮动奖",
)

LOTTERIES = {x.key: x for x in (DLT, SSQ, POWERBALL, EUROMILLIONS)}
DEFAULT = "dlt"


def get(key=None):
    return LOTTERIES[key or DEFAULT]


def enabled():
    return [x for x in LOTTERIES.values() if x.enabled]


def judge(lot, pred_front, pred_back, actual_front, actual_back):
    """判奖：返回 (前区命中, 后区命中, 奖级, 奖金)。"""
    hf = len(set(pred_front) & set(actual_front))
    hb = len(set(pred_back) & set(actual_back))
    lv, amt = lot.prize_table.get((hf, hb), (None, 0))
    return hf, hb, lv, amt


def expected_value(lot):
    """单注期望回报——超几何分布精确计算。

    这是"任何选号方式都改变不了"的那个数字。
    """
    from math import comb

    total = 0.0
    for (a, c), (_lv, amt) in lot.prize_table.items():
        if a > lot.front_pick or c > lot.back_pick:
            continue
        pa = (comb(lot.front_pick, a)
              * comb(lot.front_max - lot.front_pick, lot.front_pick - a)
              / comb(lot.front_max, lot.front_pick))
        if lot.back_pick:
            pc = (comb(lot.back_pick, c)
                  * comb(lot.back_max - lot.back_pick, lot.back_pick - c)
                  / comb(lot.back_max, lot.back_pick))
        else:
            pc = 1.0 if c == 0 else 0.0
        total += pa * pc * amt
    return total
