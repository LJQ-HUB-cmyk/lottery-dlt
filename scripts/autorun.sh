#!/bin/bash
# 本机自动运行：抓开奖 → 结算 → 锁定下期 → 上链 → 校验 → 空对照 → 推送
#
# 由 launchd 每天上午跑（09:00，15:00 兜底一次）。选上午是因为它离所有
# 开奖时刻都远：昨晚的结果必定已发布，今晚的开奖还有 11 小时以上——
# 抓取端不会撞上"已开奖但未发布"的空窗，写入端有充足余量。
#
# 中国两个彩种最短间隔 2 天（约 47 小时），每天跑一次有两倍余量，
# 偶尔失败一次也漏不掉。所有命令幂等：没有新开奖就什么都不做，
# 预测已存在就跳过，链头没变就不重复锚定。
#
# Mac 睡眠期间不执行，唤醒后 launchd 会补跑漏掉的调度。

set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$DIR/.venv/bin/python"
LOG_DIR="$DIR/logs"
LOG="$LOG_DIR/autorun.log"

mkdir -p "$LOG_DIR"
cd "$DIR" || exit 1

log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

if [ ! -x "$PY" ]; then
    log "错误：找不到 $PY"
    exit 1
fi

# 等网络就绪——唤醒后 Wi-Fi 可能还没连上
for i in 1 2 3 4 5 6; do
    if ping -c1 -W2 223.5.5.5 >/dev/null 2>&1 \
       || ping -c1 -W2 1.1.1.1 >/dev/null 2>&1; then
        break
    fi
    [ "$i" = 6 ] && { log "网络不可达，跳过本次"; exit 0; }
    sleep 10
done

log "===== 开始 ====="

# 1. 抓开奖 + 结算（有新数据才写）
"$PY" -m src.cron settle >> "$LOG" 2>&1
log "settle 退出码 $?"

# 2. 锁定下期预测（已存在则跳过）
"$PY" -m src.cron lock >> "$LOG" 2>&1
log "lock 退出码 $?"

# 3. 上链。紧接着 lock 做，今天锁的预测在今晚开奖前十几小时就进入比特币，
#    证据比"开奖后才确认"硬得多。链头没变时 anchor 自己会跳过，
#    所以一天跑几次都无所谓，不需要标记文件。
"$PY" -m src.cron anchor >> "$LOG" 2>&1
log "anchor 退出码 $?"

# 4. 校验哈希链，断裂时单独告警
VERIFY_OK=1
if ! "$PY" -m src.cron verify >> "$LOG" 2>&1; then
    VERIFY_OK=0
    log "⚠️ 哈希链校验失败！"
    osascript -e 'display notification "哈希链校验失败，数据可能被修改" with title "彩票验证系统"' 2>/dev/null
fi

# 5. 空对照。历史每期都在增长，这组数字也会跟着动——把它和战绩一起
#    每天记一遍，是为了让"模型没学到东西"这个结论也有时间序列，
#    而不是一次性的断言。约 10 秒。
"$PY" -m src.cron null >> "$LOG" 2>&1
log "null 退出码 $?"

# 6. 推送到公开仓库，让预测和验证结果对外可查。
#    本机是唯一写入方（GitHub Actions 的定时任务已停），所以直接推，
#    不需要 rebase。没配远端就跳过，等配好了自动生效。
if [ "$VERIFY_OK" = 0 ]; then
    log "链校验未通过，本次不推送"
elif git remote get-url origin >/dev/null 2>&1; then
    git add data/
    if git diff --staged --quiet; then
        log "无数据变更，不提交"
    else
        if git commit -q -m "数据更新 $(date '+%F %H:%M')" >> "$LOG" 2>&1 \
           && git push -q origin HEAD >> "$LOG" 2>&1; then
            log "已推送到远端"
        else
            log "推送失败，数据已在本地提交，下次自动重试"
        fi
    fi
fi

log "===== 结束 ====="

# 日志超过 5MB 就截断，只留最后 2000 行
if [ -f "$LOG" ] && [ "$(wc -c < "$LOG")" -gt 5242880 ]; then
    tail -2000 "$LOG" > "$LOG.tmp" && mv "$LOG.tmp" "$LOG"
    log "日志已截断"
fi

exit 0
