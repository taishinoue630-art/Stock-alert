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
