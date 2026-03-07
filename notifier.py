“””
Discord Notifier for Stock Signals
“””

import os
import json
import urllib.request
import urllib.error
from datetime import datetime
from analyzer import Signal

DISCORD_WEBHOOK_URL = os.environ.get(“DISCORD_WEBHOOK_URL”, “”)

SIGNAL_COLORS = {
“bottom”:      0x00FF88,
“take_profit”: 0xFF4444,
“buy_trend”:   0x4488FF,
}

STRENGTH_EMOJIS = {
“強”: “🔥🔥🔥”,
“中”: “⚡⚡”,
“弱”: “💡”,
}

TICKER_DISPLAY = {
“NVDA”: “NVIDIA (NVDA)”,
“PLTR”: “Palantir (PLTR)”,
}

ACTION_MAP = {
“bottom”:      “押し目買い・ナンピン検討のタイミングです”,
“take_profit”: “部分利確・ストップ引き上げを検討してください”,
“buy_trend”:   “トレンド継続中・追加購入を検討してください”,
}

def send_discord_notification(signals: list) -> bool:
if not DISCORD_WEBHOOK_URL:
print(“ERROR: DISCORD_WEBHOOK_URL not set”)
return False
if not signals:
return True

```
success = True
for signal in signals:
    color = SIGNAL_COLORS.get(signal.signal_type, 0x888888)
    ma50_diff  = (signal.current_price - signal.ma50)  / signal.ma50  * 100
    ma200_diff = (signal.current_price - signal.ma200) / signal.ma200 * 100
    strength   = STRENGTH_EMOJIS.get(signal.strength, signal.strength)
    action     = ACTION_MAP.get(signal.signal_type, "")
    ts         = signal.timestamp.strftime("%Y-%m-%d %H:%M JST")

    payload = {
        "embeds": [{
            "title": f"{signal.signal_name_jp}  |  {TICKER_DISPLAY.get(signal.ticker, signal.ticker)}",
            "color": color,
            "fields": [
                {"name": "現在価格", "value": f"${signal.current_price:.2f}", "inline": True},
                {"name": "RSI(14)", "value": f"{signal.rsi:.1f}", "inline": True},
                {"name": "強度", "value": f"{strength} {signal.strength}", "inline": True},
                {"name": "MA50",  "value": f"${signal.ma50:.2f} ({ma50_diff:+.1f}%)",  "inline": True},
                {"name": "MA200", "value": f"${signal.ma200:.2f} ({ma200_diff:+.1f}%)", "inline": True},
                {"name": "シグナル根拠", "value": signal.reason, "inline": False},
                {"name": "推奨アクション", "value": action, "inline": False},
            ],
            "footer": {"text": f"Stock Alert | {ts} | 過去6年データ分析"},
        }]
    }

    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        DISCORD_WEBHOOK_URL,
        data=data,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            print(f"Sent [{signal.ticker}] {signal.signal_type} — status {r.getcode()}")
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f"Discord HTTP error: {e.code} — {body}")
        success = False
    except Exception as e:
        print(f"Discord error: {e}")
        success = False

return success
```

def send_heartbeat(ticker_status: dict) -> None:
if not DISCORD_WEBHOOK_URL:
return
now = datetime.now().strftime(”%Y-%m-%d %H:%M JST”)
lines = “\n”.join(f”• {t}: ${p:.2f}” for t, p in ticker_status.items())
payload = {
“embeds”: [{
“title”: “ハートビート — システム正常稼働中”,
“description”: f”**現在価格**\n{lines}”,
“color”: 0x888888,
“footer”: {“text”: f”最終確認: {now}”},
}]
}
data = json.dumps(payload, ensure_ascii=False).encode(“utf-8”)
req = urllib.request.Request(
DISCORD_WEBHOOK_URL,
data=data,
headers={“Content-Type”: “application/json; charset=utf-8”},
method=“POST”,
)
try:
urllib.request.urlopen(req, timeout=15)
print(“Heartbeat sent”)
except Exception as e:
print(f”Heartbeat error: {e}”)
