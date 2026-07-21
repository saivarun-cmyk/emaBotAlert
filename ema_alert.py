import json
import os
from datetime import datetime, time as dtime
import pandas as pd
import yfinance as yf
import requests

CONFIG_PATH = "config.json"
STATE_PATH = "state.json"

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

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


def parse_hhmm(s: str) -> dtime:
    h, m = map(int, s.split(":"))
    return dtime(h, m)


def is_within_market_hours(ts, start: dtime, end: dtime) -> bool:
    t = ts.time()
    return start <= t <= end


def check_symbol(symbol: str, interval: str, ema_period: int, state: dict,
                  alert_on_touch: bool, market_start: dtime, market_end: dtime):
    period = PERIOD_FOR_INTERVAL.get(interval, "5d")
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
    except Exception as e:
        print(f"Fetch failed for {symbol} {interval}: {e}")
        return

    if df is None or df.empty or len(df) < ema_period + 2:
        print(f"Not enough data for {symbol} {interval}")
        return

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = compute_ema(df, ema_period)
    closed = df.iloc[:-1]  # drop the still-forming candle

    key = f"{symbol}_{interval}"
    is_first_run = key not in state

    if is_first_run:
        if len(closed) > 0:
            state[key] = {"last_timestamp": str(closed.index[-1])}
        print(f"Bootstrapped {key}, no backlog alerts sent.")
        return

    last_processed = state[key]["last_timestamp"]
    new_candles = closed[closed.index.astype(str) > last_processed]

    for ts, row in new_candles.iterrows():
        idx_pos = closed.index.get_loc(ts)
        if idx_pos == 0:
            continue

        # Skip anything outside market hours (pre-open/post-close candles)
        if not is_within_market_hours(ts, market_start, market_end):
            continue

        prev_row = closed.iloc[idx_pos - 1]
        ema_val = row["EMA"]
        high, low, close = row["High"], row["Low"], row["Close"]
        prev_close, prev_ema = prev_row["Close"], prev_row["EMA"]

        touched = low <= ema_val <= high
        crossed_up = prev_close < prev_ema and close > ema_val
        crossed_down = prev_close > prev_ema and close < ema_val

        should_alert = crossed_up or crossed_down or (alert_on_touch and touched)

        if should_alert:
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
    default_cfg = {
        "stocks": [],
        "ema_period": 20,
        "timeframes": {"5m": {"alert_on_touch": False}, "15m": {"alert_on_touch": True}},
        "market_start": "09:15",
        "market_end": "15:15",
    }
    config = load_json(CONFIG_PATH, default_cfg)
    state = load_json(STATE_PATH, {})

    market_start = parse_hhmm(config.get("market_start", "09:15"))
    market_end = parse_hhmm(config.get("market_end", "15:15"))
    ema_period = config.get("ema_period", 20)

    for symbol in config["stocks"]:
        for interval, tf_settings in config["timeframes"].items():
            alert_on_touch = tf_settings.get("alert_on_touch", False)
            check_symbol(symbol, interval, ema_period, state, alert_on_touch, market_start, market_end)

    save_json(STATE_PATH, state)


if __name__ == "__main__":
    main()