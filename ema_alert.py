import json
import os
import sys
from datetime import datetime, timezone
import pandas as pd
import yfinance as yf
import requests

CONFIG_PATH = "config.json"
STATE_PATH = "state.json"

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

# yfinance needs enough history for a stable EMA20
PERIOD_FOR_INTERVAL = {
    "5m": "5d",
    "15m": "1mo",
}


def load_json(path, default):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return default


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def send_telegram(message: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram not configured, skipping send. Message was:\n", message)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    resp = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": message})
    if resp.status_code != 200:
        print("Telegram send failed:", resp.text)


def compute_ema(df: pd.DataFrame, period: int) -> pd.DataFrame:
    df = df.copy()
    df["EMA"] = df["Close"].ewm(span=period, adjust=False).mean()
    return df


def check_symbol(symbol: str, interval: str, ema_period: int, state: dict):
    period = PERIOD_FOR_INTERVAL.get(interval, "5d")
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
    except Exception as e:
        print(f"Fetch failed for {symbol} {interval}: {e}")
        return

    if df is None or df.empty or len(df) < ema_period + 2:
        print(f"Not enough data for {symbol} {interval}")
        return

    # yfinance sometimes returns MultiIndex columns for a single ticker; flatten if so
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = compute_ema(df, ema_period)

    # Drop the last row: it's the still-forming candle, not a closed one
    closed = df.iloc[:-1]

    key = f"{symbol}_{interval}"
    last_processed = state.get(key, {}).get("last_timestamp")

    # Only look at candles newer than the last one we already processed
    new_candles = closed
    if last_processed:
        new_candles = closed[closed.index.astype(str) > last_processed]

    for ts, row in new_candles.iterrows():
        idx_pos = closed.index.get_loc(ts)
        if idx_pos == 0:
            continue  # need a previous candle to detect a crossover
        prev_row = closed.iloc[idx_pos - 1]

        ema_val = row["EMA"]
        high, low, close = row["High"], row["Low"], row["Close"]
        prev_close, prev_ema = prev_row["Close"], prev_row["EMA"]

        touched = low <= ema_val <= high
        crossed_up = prev_close < prev_ema and close > ema_val
        crossed_down = prev_close > prev_ema and close < ema_val

        if touched or crossed_up or crossed_down:
            reason = []
            if crossed_up:
                reason.append("crossed ABOVE EMA20")
            if crossed_down:
                reason.append("crossed BELOW EMA20")
            if touched and not (crossed_up or crossed_down):
                reason.append("EMA20 is inside candle range (touch)")

            msg = (
                f"📊 {symbol} [{interval}]\n"
                f"Time: {ts}\n"
                f"Close: {close:.2f} | EMA20: {ema_val:.2f}\n"
                f"Signal: {', '.join(reason)}"
            )
            send_telegram(msg)
            print(msg)

    if len(closed) > 0:
        state[key] = {"last_timestamp": str(closed.index[-1])}


def main():
    config = load_json(CONFIG_PATH, {"stocks": [], "timeframes": ["5m", "15m"], "ema_period": 20})
    state = load_json(STATE_PATH, {})

    for symbol in config["stocks"]:
        for interval in config["timeframes"]:
            check_symbol(symbol, interval, config.get("ema_period", 20), state)

    save_json(STATE_PATH, state)


if __name__ == "__main__":
    main()