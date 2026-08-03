"""大乐透奖级判定与奖金表（基本投注，2 元/注）。

一、二等奖为浮动奖，这里取历史量级的粗略近似，仅用于估算，不代表实际奖金。
三等奖及以下为固定奖金。
"""

TICKET_PRICE = 2

# (前区命中, 后区命中) -> (奖级, 奖金)
PRIZE_TABLE = {
    (5, 2): (1, 10_000_000),
    (5, 1): (2, 200_000),
    (5, 0): (3, 10_000),
    (4, 2): (4, 3_000),
    (4, 1): (5, 300),
    (3, 2): (6, 200),
    (4, 0): (7, 100),
    (3, 1): (8, 15),
    (2, 2): (8, 15),
    (3, 0): (9, 5),
    (1, 2): (9, 5),
    (2, 1): (9, 5),
    (0, 2): (9, 5),
}

LEVEL_AMOUNT = {level: amount for level, amount in PRIZE_TABLE.values()}

LEVEL_NAMES = {
    1: "一等奖", 2: "二等奖", 3: "三等奖", 4: "四等奖", 5: "五等奖",
    6: "六等奖", 7: "七等奖", 8: "八等奖", 9: "九等奖",
}


def judge(pick_front, pick_back, win_front, win_back):
    """返回 (前区命中数, 后区命中数, 奖级或None, 奖金)。"""
    hf = len(set(pick_front) & set(win_front))
    hb = len(set(pick_back) & set(win_back))
    level, amount = PRIZE_TABLE.get((hf, hb), (None, 0))
    return hf, hb, level, amount


def guaranteed_level(t_front, t_back):
    """wheel 保证命中 t_front 个前区 + t_back 个后区时，对应哪个奖级。"""
    return PRIZE_TABLE.get((t_front, t_back), (None, 0))
