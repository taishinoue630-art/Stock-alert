"""
Stock Technical Analyzer for NVDA & PLTR
Analyzes 6 years of historical data to detect:
- 底打ち (Bottom signals)
- 利確 (Take-profit signals)  
- 追加購入推奨トレンド (Buy trend signals)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional


@dataclass
class Signal:
    ticker: str
    signal_type: str       # "bottom" | "take_profit" | "buy_trend"
    signal_name_jp: str    # Japanese signal name
    current_price: float
    reason: str
    strength: str          # "強" | "中" | "弱"
    rsi: float
    ma50: float
    ma200: float
    timestamp: datetime


def fetch_data(ticker: str, years: int = 6) -> pd.DataFrame:
    """Fetch historical stock data"""
    end = datetime.now()
    start = end - timedelta(days=years * 365)
    df = yf.download(ticker, start=start, end=end, progress=False)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all technical indicators"""
    # Moving Averages
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["MA200"] = df["Close"].rolling(200).mean()

    # RSI (14-period)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    # Bollinger Bands (20-period, 2σ)
    std20 = df["Close"].rolling(20).std()
    df["BB_Upper"] = df["MA20"] + 2 * std20
    df["BB_Lower"] = df["MA20"] - 2 * std20

    # Volume ratio (vs 20-day average)
    df["Vol_Ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()

    # Stochastic RSI
    rsi_min = df["RSI"].rolling(14).min()
    rsi_max = df["RSI"].rolling(14).max()
    df["StochRSI"] = (df["RSI"] - rsi_min) / (rsi_max - rsi_min + 1e-10)

    # Rate of Change
    df["ROC20"] = df["Close"].pct_change(20) * 100

    return df


def compute_historical_thresholds(df: pd.DataFrame) -> dict:
    """
    Compute adaptive thresholds from 6 years of historical data.
    This makes signals ticker-specific and historically calibrated.
    """
    rsi_series = df["RSI"].dropna()
    roc_series = df["ROC20"].dropna()
    close = df["Close"].dropna()

    return {
        # Bottom thresholds: lower 10th percentile historically
        "rsi_bottom":       float(rsi_series.quantile(0.12)),
        "rsi_overbought":   float(rsi_series.quantile(0.88)),
        # Drawdown from 52-week high threshold
        "drawdown_bottom":  float((close / close.rolling(252).max() - 1).quantile(0.10)),
        # Strong uptrend ROC threshold
        "roc_trend_up":     float(roc_series.quantile(0.75)),
        # Overbought ROC
        "roc_overbought":   float(roc_series.quantile(0.88)),
    }


def detect_bottom_signal(row: pd.Series, thresholds: dict) -> Optional[tuple]:
    """
    底打ちシグナル detection
    Looks for confluence of oversold conditions
    """
    price = float(row["Close"])
    rsi = float(row["RSI"])
    bb_lower = float(row["BB_Lower"])
    ma200 = float(row["MA200"])
    stoch_rsi = float(row["StochRSI"])
    vol_ratio = float(row["Vol_Ratio"])
    macd_hist = float(row["MACD_Hist"])

    signals = []
    score = 0

    # RSI oversold
    if rsi < thresholds["rsi_bottom"]:
        signals.append(f"RSI={rsi:.1f}（歴史的底圏）")
        score += 2
    elif rsi < thresholds["rsi_bottom"] * 1.1:
        signals.append(f"RSI={rsi:.1f}（売られ過ぎ圏）")
        score += 1

    # Price below lower Bollinger Band
    if price < bb_lower:
        signals.append("ボリンジャーバンド下限割れ")
        score += 2

    # Stochastic RSI oversold
    if stoch_rsi < 0.15:
        signals.append(f"StochRSI={stoch_rsi:.2f}（過売）")
        score += 1

    # Price near 200 MA support or below
    if price < ma200 * 1.02:
        signals.append("200日MA付近（長期サポート）")
        score += 1

    # MACD histogram turning up (reversal signal)
    if macd_hist > 0 and score >= 2:
        signals.append("MACDヒストグラム好転")
        score += 1

    # High volume on potential bottom
    if vol_ratio > 1.5 and score >= 2:
        signals.append(f"出来高増加（{vol_ratio:.1f}x）")
        score += 1

    if score >= 3:
        strength = "強" if score >= 5 else "中" if score >= 4 else "弱"
        return (strength, "、".join(signals))
    return None


def detect_take_profit_signal(row: pd.Series, thresholds: dict) -> Optional[tuple]:
    """
    利確シグナル detection
    Looks for confluence of overbought conditions
    """
    price = float(row["Close"])
    rsi = float(row["RSI"])
    bb_upper = float(row["BB_Upper"])
    ma50 = float(row["MA50"])
    stoch_rsi = float(row["StochRSI"])
    vol_ratio = float(row["Vol_Ratio"])
    macd_hist = float(row["MACD_Hist"])
    roc20 = float(row["ROC20"])

    signals = []
    score = 0

    # RSI overbought
    if rsi > thresholds["rsi_overbought"]:
        signals.append(f"RSI={rsi:.1f}（歴史的高値圏）")
        score += 2
    elif rsi > 70:
        signals.append(f"RSI={rsi:.1f}（過熱圏）")
        score += 1

    # Price above upper Bollinger Band
    if price > bb_upper:
        signals.append("ボリンジャーバンド上限超え")
        score += 2

    # StochRSI overbought
    if stoch_rsi > 0.85:
        signals.append(f"StochRSI={stoch_rsi:.2f}（過買）")
        score += 1

    # ROC overbought (too fast rise)
    if roc20 > thresholds["roc_overbought"]:
        signals.append(f"20日騰落率={roc20:.1f}%（急騰）")
        score += 1

    # MACD histogram turning down (momentum loss)
    if macd_hist < 0 and score >= 2:
        signals.append("MACDヒストグラム悪化")
        score += 1

    # Exhaustion volume
    if vol_ratio > 2.0 and score >= 2:
        signals.append(f"出来高急増（{vol_ratio:.1f}x、天井警戒）")
        score += 1

    if score >= 3:
        strength = "強" if score >= 5 else "中" if score >= 4 else "弱"
        return (strength, "、".join(signals))
    return None


def detect_buy_trend_signal(row: pd.Series, prev_row: pd.Series, thresholds: dict) -> Optional[tuple]:
    """
    追加購入推奨トレンドシグナル detection
    Looks for confirmed uptrend with healthy momentum
    """
    price = float(row["Close"])
    rsi = float(row["RSI"])
    ma20 = float(row["MA20"])
    ma50 = float(row["MA50"])
    ma200 = float(row["MA200"])
    macd = float(row["MACD"])
    macd_signal = float(row["MACD_Signal"])
    macd_hist = float(row["MACD_Hist"])
    vol_ratio = float(row["Vol_Ratio"])
    roc20 = float(row["ROC20"])

    prev_macd_hist = float(prev_row["MACD_Hist"]) if not pd.isna(prev_row["MACD_Hist"]) else 0

    signals = []
    score = 0

    # Price above all key MAs (bullish alignment)
    if price > ma20 > ma50 > ma200:
        signals.append("MA完全強気配列（20>50>200）")
        score += 3
    elif price > ma50 > ma200:
        signals.append("中長期MA強気配列")
        score += 2

    # Golden cross region
    ma50_prev = float(prev_row["MA50"]) if not pd.isna(prev_row["MA50"]) else ma50
    ma200_prev = float(prev_row["MA200"]) if not pd.isna(prev_row["MA200"]) else ma200
    if ma50 > ma200 and ma50_prev <= ma200_prev:
        signals.append("🌟 ゴールデンクロス発生！")
        score += 4

    # MACD bullish crossover
    if macd > macd_signal and macd_hist > 0 and prev_macd_hist <= 0:
        signals.append("MACDゴールデンクロス")
        score += 2
    elif macd > macd_signal and macd_hist > 0:
        signals.append("MACD強気継続")
        score += 1

    # Healthy RSI (not overbought, but strong)
    if 50 < rsi < 70:
        signals.append(f"RSI={rsi:.1f}（健全な上昇モメンタム）")
        score += 1

    # Strong positive ROC (uptrend)
    if roc20 > thresholds["roc_trend_up"]:
        signals.append(f"20日騰落率={roc20:.1f}%（強い上昇トレンド）")
        score += 1

    # Volume confirming uptrend
    if vol_ratio > 1.3 and score >= 2:
        signals.append(f"出来高増加で上昇確認（{vol_ratio:.1f}x）")
        score += 1

    if score >= 4:
        strength = "強" if score >= 6 else "中" if score >= 5 else "弱"
        return (strength, "、".join(signals))
    return None


def analyze_ticker(ticker: str) -> list[Signal]:
    """Run full analysis for a single ticker"""
    df = fetch_data(ticker)
    df = compute_indicators(df)
    thresholds = compute_historical_thresholds(df)

    # Use last two rows for signal detection
    df_clean = df.dropna()
    if len(df_clean) < 2:
        return []

    latest = df_clean.iloc[-1]
    prev = df_clean.iloc[-2]
    now = datetime.now()

    signals = []

    # Shared values
    price = float(latest["Close"])
    rsi = float(latest["RSI"])
    ma50 = float(latest["MA50"])
    ma200 = float(latest["MA200"])

    # Check bottom signal
    result = detect_bottom_signal(latest, thresholds)
    if result:
        strength, reason = result
        signals.append(Signal(
            ticker=ticker,
            signal_type="bottom",
            signal_name_jp="🟢 底打ちシグナル",
            current_price=price,
            reason=reason,
            strength=strength,
            rsi=rsi,
            ma50=ma50,
            ma200=ma200,
            timestamp=now,
        ))

    # Check take-profit signal
    result = detect_take_profit_signal(latest, thresholds)
    if result:
        strength, reason = result
        signals.append(Signal(
            ticker=ticker,
            signal_type="take_profit",
            signal_name_jp="🔴 利確シグナル",
            current_price=price,
            reason=reason,
            strength=strength,
            rsi=rsi,
            ma50=ma50,
            ma200=ma200,
            timestamp=now,
        ))

    # Check buy trend signal
    result = detect_buy_trend_signal(latest, prev, thresholds)
    if result:
        strength, reason = result
        signals.append(Signal(
            ticker=ticker,
            signal_type="buy_trend",
            signal_name_jp="📈 追加購入推奨トレンド",
            current_price=price,
            reason=reason,
            strength=strength,
            rsi=rsi,
            ma50=ma50,
            ma200=ma200,
            timestamp=now,
        ))

    return signals


def run_analysis() -> list[Signal]:
    """Analyze all target tickers"""
    tickers = ["NVDA", "PLTR"]
    all_signals = []
    for ticker in tickers:
        try:
            signals = analyze_ticker(ticker)
            all_signals.extend(signals)
            print(f"[{ticker}] {len(signals)} signal(s) detected")
        except Exception as e:
            print(f"[{ticker}] Analysis error: {e}")
    return all_signals


if __name__ == "__main__":
    signals = run_analysis()
    for s in signals:
        print(f"\n{s.signal_name_jp} [{s.ticker}]")
        print(f"  価格: ${s.current_price:.2f}")
        print(f"  強度: {s.strength}")
        print(f"  根拠: {s.reason}")
"""
Main runner for Stock Alert System
Handles deduplication, state tracking, and orchestration
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import yfinance as yf

from analyzer import run_analysis, Signal
from notifier import send_discord_notification, send_heartbeat

# State file to prevent duplicate alerts (stored in repo root)
STATE_FILE = Path("signal_state.json")
# Cooldown period: don't re-alert same signal within N hours
COOLDOWN_HOURS = 4


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


def signal_key(signal: Signal) -> str:
    return f"{signal.ticker}:{signal.signal_type}"


def is_duplicate(signal: Signal, state: dict) -> bool:
    """Check if signal was recently sent (within cooldown window)"""
    key = signal_key(signal)
    if key not in state:
        return False
    last_sent_str = state[key].get("last_sent", "")
    if not last_sent_str:
        return False
    try:
        last_sent = datetime.fromisoformat(last_sent_str)
        cooldown = timedelta(hours=COOLDOWN_HOURS)
        return datetime.now() - last_sent < cooldown
    except Exception:
        return False


def update_state(signal: Signal, state: dict) -> None:
    key = signal_key(signal)
    state[key] = {
        "last_sent": datetime.now().isoformat(),
        "ticker": signal.ticker,
        "signal_type": signal.signal_type,
        "price": signal.current_price,
    }


def get_current_prices() -> dict:
    """Fetch current prices for heartbeat"""
    prices = {}
    for ticker in ["NVDA", "PLTR"]:
        try:
            data = yf.download(ticker, period="1d", progress=False)
            data.columns = [c[0] if isinstance(c, tuple) else c for c in data.columns]
            if not data.empty:
                prices[ticker] = float(data["Close"].iloc[-1])
        except Exception as e:
            print(f"Price fetch error for {ticker}: {e}")
    return prices


def main():
    print(f"\n{'='*50}")
    print(f"Stock Alert System — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*50}")

    # Check for heartbeat mode (run once per day via separate workflow trigger)
    if "--heartbeat" in sys.argv:
        prices = get_current_prices()
        send_heartbeat(prices)
        return

    # Run analysis
    print("\n[1/3] Running technical analysis...")
    all_signals = run_analysis()
    print(f"     → Total signals detected: {len(all_signals)}")

    if not all_signals:
        print("\n✅ No signals detected. No notification sent.")
        return

    # Deduplicate against recent alerts
    print("\n[2/3] Checking for duplicate signals...")
    state = load_state()
    new_signals = []

    for signal in all_signals:
        if is_duplicate(signal, state):
            key = signal_key(signal)
            last = state[key]["last_sent"]
            print(f"     → SKIPPED (cooldown): {key} (last sent: {last})")
        else:
            new_signals.append(signal)
            print(f"     → NEW signal: {signal.ticker} {signal.signal_name_jp} [{signal.strength}]")

    if not new_signals:
        print("\n✅ All signals are within cooldown window. No notification sent.")
        return

    # Send Discord notifications
    print(f"\n[3/3] Sending {len(new_signals)} notification(s) to Discord...")
    success = send_discord_notification(new_signals)

    if success:
        # Update state to prevent duplicate notifications
        for signal in new_signals:
            update_state(signal, state)
        save_state(state)
        print(f"\n✅ Successfully sent {len(new_signals)} notification(s)!")
    else:
        print("\n❌ Failed to send Discord notification!")
        sys.exit(1)


if __name__ == "__main__":
    main()
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
# 📊 株価アラートシステム — NVDA & PLTR

NVIDIAとPalantirの株価を自動監視し、重要なシグナルをDiscordに通知するシステムです。

---

## 🔍 検出するシグナル

過去6年間の株価データから算出した**個別閾値**を使ってシグナルを検出します。

| シグナル | 説明 | 使用指標 |
|----------|------|---------|
| 🟢 **底打ち** | 過売り・反転の兆候 | RSI・ボリンジャーバンド・StochRSI・MACD |
| 🔴 **利確** | 過買い・天井の兆候 | RSI・ボリンジャーバンド・StochRSI・騰落率 |
| 📈 **追加購入推奨** | 上昇トレンドの確認 | MA配列・ゴールデンクロス・MACD・RSI |

### シグナル強度
- 🔥🔥🔥 **強** — 複数指標が強く一致、高確度
- ⚡⚡ **中** — 複数指標が一致、要確認
- 💡 **弱** — 一部指標が反応、様子見

---

## ⚙️ セットアップ手順

### Step 1: GitHubリポジトリを作成

```bash
# このフォルダをGitHubにプッシュ
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/stock-alert
git push -u origin main
```

### Step 2: DiscordのWebhook URLを取得

1. Discordで通知を受け取りたい**チャンネル**を右クリック
2. 「チャンネルの編集」→「連携サービス」→「ウェブフック」
3. 「新しいウェブフック」をクリック
4. 「ウェブフックURLをコピー」

### Step 3: GitHubにSecretを設定

1. GitHubリポジトリの `Settings` → `Secrets and variables` → `Actions`
2. `New repository secret` をクリック
3. 以下を設定：

| Name | Value |
|------|-------|
| `DISCORD_WEBHOOK_URL` | DiscordのWebhook URL |

### Step 4: GitHub Actionsを有効化

1. リポジトリの `Actions` タブを開く
2. ワークフローを有効化する

### Step 5: 動作テスト

1. `Actions` タブ → `Stock Alert Monitor`
2. `Run workflow` → `mode: test` を選択して実行
3. Discordにテスト通知が届けば成功！

---

## ⏰ 実行スケジュール

| タイミング | 内容 |
|-----------|------|
| **平日14:00〜22:00 UTC（毎15分）** | テクニカル分析・シグナル検出 |
| **平日00:00 UTC（毎日）** | ハートビート通知（稼働確認） |
| **手動** | Actions画面から任意実行可能 |

> 💡 14:00-22:00 UTC = 23:00-07:00 JST（米国市場時間に相当）

---

## 🔔 通知の仕様

- **重複防止**: 同一シグナルは4時間以内に再通知しない
- **通知先**: Discord Webhook（指定チャンネル）
- **状態管理**: `signal_state.json` に最終通知時刻を記録

---

## 📁 ファイル構成

```
stock-alert/
├── .github/
│   └── workflows/
│       └── stock_monitor.yml   # GitHub Actions設定
├── analyzer.py                 # テクニカル分析エンジン
├── notifier.py                 # Discord通知モジュール
├── main.py                     # メインスクリプト
├── requirements.txt            # Python依存パッケージ
├── signal_state.json           # シグナル状態（自動生成）
└── README.md                   # このファイル
```

---

## 📐 使用テクニカル指標

| 指標 | 期間 | 用途 |
|------|------|------|
| RSI | 14日 | 買われ過ぎ・売られ過ぎ |
| Stochastic RSI | 14日 | RSIの過熱度 |
| MACD | 12/26/9 | トレンド転換 |
| ボリンジャーバンド | 20日・2σ | 価格の逸脱 |
| 移動平均 (MA) | 20/50/200日 | トレンド・サポート |
| 出来高比率 | 20日平均比 | 動きの信頼性 |
| ROC | 20日 | モメンタム強度 |

---

## ⚠️ 免責事項

このシステムはテクニカル分析に基name: 📊 Stock Alert Monitor

on:
  # Run every 15 minutes during US market hours (ET = UTC-5/UTC-4)
  # Market hours: 9:30 AM - 4:00 PM ET = 14:30 - 21:00 UTC
  # Also runs 30min before open and 30min after close for pre/post market
  schedule:
    - cron: "*/15 14-21 * * 1-5"   # Every 15min, Mon-Fri, 14:00-21:59 UTC

  # Daily heartbeat at 9:00 AM JST (00:00 UTC)
    - cron: "0 0 * * 1-5"

  # Allow manual trigger from GitHub Actions UI
  workflow_dispatch:
    inputs:
      mode:
        description: "Run mode"
        required: false
        default: "normal"
        type: choice
        options:
          - normal
          - heartbeat
          - test

permissions:
  contents: write   # needed to commit signal_state.json

jobs:
  stock-monitor:
    name: 📈 Analyze & Notify
    runs-on: ubuntu-latest
    timeout-minutes: 10

    steps:
      # ── Checkout ──────────────────────────────────────────────
      - name: Checkout repository
        uses: actions/checkout@v4
        with:
          fetch-depth: 1

      # ── Python setup ──────────────────────────────────────────
      - name: Set up Python 3.11
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: "pip"

      - name: Install dependencies
        run: pip install -r requirements.txt

      # ── Determine run mode ────────────────────────────────────
      - name: Determine run mode
        id: mode
        run: |
          GITHUB_EVENT="${{ github.event_name }}"
          INPUT_MODE="${{ github.event.inputs.mode }}"

          # Daily heartbeat cron (00:00 UTC)
          CURRENT_MINUTE=$(date -u +"%M")
          CURRENT_HOUR=$(date -u +"%H")

          if [ "$INPUT_MODE" = "heartbeat" ]; then
            echo "mode=heartbeat" >> $GITHUB_OUTPUT
          elif [ "$INPUT_MODE" = "test" ]; then
            echo "mode=test" >> $GITHUB_OUTPUT
          elif [ "$CURRENT_HOUR" = "00" ] && [ "$CURRENT_MINUTE" = "00" ]; then
            echo "mode=heartbeat" >> $GITHUB_OUTPUT
          else
            echo "mode=normal" >> $GITHUB_OUTPUT
          fi

      # ── Restore signal state ──────────────────────────────────
      - name: Restore signal state
        run: |
          if [ -f signal_state.json ]; then
            echo "Signal state file found:"
            cat signal_state.json
          else
            echo "No state file found, starting fresh"
            echo "{}" > signal_state.json
          fi

      # ── Run analysis ──────────────────────────────────────────
      - name: Run stock analysis (normal)
        if: steps.mode.outputs.mode == 'normal'
        env:
          DISCORD_WEBHOOK_URL: ${{ secrets.DISCORD_WEBHOOK_URL }}
        run: python main.py

      - name: Run heartbeat
        if: steps.mode.outputs.mode == 'heartbeat'
        env:
          DISCORD_WEBHOOK_URL: ${{ secrets.DISCORD_WEBHOOK_URL }}
        run: python main.py --heartbeat

      - name: Run test (sends test signal to Discord)
        if: steps.mode.outputs.mode == 'test'
        env:
          DISCORD_WEBHOOK_URL: ${{ secrets.DISCORD_WEBHOOK_URL }}
        run: |
          python -c "
          from notifier import send_discord_notification
          from analyzer import Signal
          from datetime import datetime
          s = Signal(
              ticker='NVDA',
              signal_type='bottom',
              signal_name_jp='🟢 底打ちシグナル',
              current_price=105.50,
              reason='テスト通知 — システム正常稼働中',
              strength='中',
              rsi=32.5,
              ma50=115.0,
              ma200=98.5,
              timestamp=datetime.now(),
          )
          send_discord_notification([s])
          print('Test notification sent!')
          "

      # ── Commit updated state ──────────────────────────────────
      - name: Commit signal state
        if: steps.mode.outputs.mode == 'normal'
        run: |
          git config user.name  "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add signal_state.json
          # Only commit if there are changes
          git diff --staged --quiet || git commit -m "chore: update signal state [skip ci]"
          git push
づく情報提供を目的としており、**投資助言ではありません**。
投資判断は自己責任で行ってください。
yfinance>=0.2.40
pandas>=2.0.0
numpy>=1.24.0
