"""
Discord Notifier for Stock Signals
Sends rich embed messages to Discord webhook
"""

import os
import json
import urllib.request
import urllib.error
from datetime import datetime
from analyzer import Signal


DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")

# Signal color codes (Discord embed colors)
SIGNAL_COLORS = {
    "bottom":     0x00FF88,   # Green - bottom signal
    "take_profit": 0xFF4444,  # Red - take profit
    "buy_trend":  0x4488FF,   # Blue - buy trend
}

SIGNAL_EMOJIS = {
    "bottom":     "🟢",
    "take_profit": "🔴",
    "buy_trend":  "📈",
}

STRENGTH_EMOJIS = {
    "強": "🔥🔥🔥",
    "中": "⚡⚡",
    "弱": "💡",
}

TICKER_EMOJIS = {
    "NVDA": "🟩 NVIDIA",
    "PLTR": "🔷 Palantir",
}


def build_embed(signal: Signal) -> dict:
    """Build a Discord embed object for a signal"""
    color = SIGNAL_COLORS[signal.signal_type]
    ticker_display = TICKER_EMOJIS.get(signal.ticker, signal.ticker)
    strength_display = STRENGTH_EMOJIS.get(signal.strength, signal.strength)
    ts = signal.timestamp.strftime("%Y-%m-%d %H:%M:%S JST")

    # MA distance from price
    ma50_diff = (signal.current_price - signal.ma50) / signal.ma50 * 100
    ma200_diff = (signal.current_price - signal.ma200) / signal.ma200 * 100

    ma50_str = f"${signal.ma50:.2f} ({ma50_diff:+.1f}%)"
    ma200_str = f"${signal.ma200:.2f} ({ma200_diff:+.1f}%)"

    # Action guidance
    action_map = {
        "bottom":     "💰 押し目買い・ナンピン検討のタイミングです",
        "take_profit": "📤 部分利確・ストップ引き上げを検討してください",
        "buy_trend":  "🚀 トレンド継続中・追加購入を検討してください",
    }
    action = action_map[signal.signal_type]

    embed = {
        "title": f"{signal.signal_name_jp}  |  {ticker_display}",
        "color": color,
        "fields": [
            {
                "name": "💵 現在価格",
                "value": f"**${signal.current_price:.2f}**",
                "inline": True,
            },
            {
                "name": "📊 RSI (14)",
                "value": f"**{signal.rsi:.1f}**",
                "inline": True,
            },
            {
                "name": "💪 シグナル強度",
                "value": f"{strength_display} **{signal.strength}**",
                "inline": True,
            },
            {
                "name": "📉 MA50",
                "value": ma50_str,
                "inline": True,
            },
            {
                "name": "📉 MA200",
                "value": ma200_str,
                "inline": True,
            },
            {
                "name": "\u200b",
                "value": "\u200b",
                "inline": True,
            },
            {
                "name": "🔍 シグナル根拠",
                "value": signal.reason,
                "inline": False,
            },
            {
                "name": "💡 推奨アクション",
                "value": action,
                "inline": False,
            },
        ],
        "footer": {
            "text": f"Stock Alert System  |  {ts}  |  過去6年データ分析"
        },
        "timestamp": signal.timestamp.isoformat(),
    }
    return embed


def send_discord_notification(signals: list[Signal]) -> bool:
    """Send all signals to Discord as a batch"""
    if not DISCORD_WEBHOOK_URL:
        print("ERROR: DISCORD_WEBHOOK_URL environment variable not set")
        return False

    if not signals:
        print("No signals to send")
        return True

    # Group signals by ticker for cleaner output
    embeds = [build_embed(s) for s in signals]

    # Discord allows max 10 embeds per message
    batch_size = 10
    success = True

    for i in range(0, len(embeds), batch_size):
        batch = embeds[i:i + batch_size]
        payload = {
            "username": "📊 Stock Alert Bot",
            "avatar_url": "https://cdn.discordapp.com/embed/avatars/0.png",
            "content": f"⚠️ **株価シグナル検出** — {len(signals)}件のシグナルが発生しました！",
            "embeds": batch,
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            DISCORD_WEBHOOK_URL,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                status = resp.getcode()
                if status in (200, 204):
                    print(f"Discord notification sent ({len(batch)} embeds)")
                else:
                    print(f"Unexpected Discord response: {status}")
                    success = False
        except urllib.error.HTTPError as e:
            print(f"Discord HTTP error: {e.code} — {e.read().decode()}")
            success = False
        except Exception as e:
            print(f"Discord send error: {e}")
            success = False

    return success


def send_heartbeat(ticker_status: dict) -> None:
    """Send a daily heartbeat message confirming the bot is running"""
    if not DISCORD_WEBHOOK_URL:
        return

    now = datetime.now().strftime("%Y-%m-%d %H:%M JST")
    lines = [f"• **{t}**: ${p:.2f}" for t, p in ticker_status.items()]
    status_str = "\n".join(lines)

    payload = {
        "username": "📊 Stock Alert Bot",
        "embeds": [{
            "title": "💓 ハートビート — システム正常稼働中",
            "description": f"**現在価格**\n{status_str}",
            "color": 0x888888,
            "footer": {"text": f"最終確認: {now}"},
        }]
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        DISCORD_WEBHOOK_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=10)
        print("Heartbeat sent to Discord")
    except Exception as e:
        print(f"Heartbeat send error: {e}")


if __name__ == "__main__":
    # Quick test with dummy signal
    from analyzer import Signal
    test_signal = Signal(
        ticker="NVDA",
        signal_type="bottom",
        signal_name_jp="🟢 底打ちシグナル",
        current_price=105.50,
        reason="RSI=28.3（歴史的底圏）、ボリンジャーバンド下限割れ、200日MA付近（長期サポート）",
        strength="強",
        rsi=28.3,
        ma50=115.0,
        ma200=98.5,
        timestamp=datetime.now(),
    )
    send_discord_notification([test_signal])
