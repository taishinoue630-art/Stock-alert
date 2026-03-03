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
