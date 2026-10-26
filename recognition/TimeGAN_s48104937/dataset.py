import numpy as np
import pandas as pd

# --- Data files ---
BOOK_PATH = "data/{ticker}_2012-06-21_34200000_57600000_orderbook_10.csv"
MSG_PATH = "data/{ticker}_2012-06-21_34200000_57600000_message_10.csv"
TICKERS: list[str]  = ["AMZN", "AAPL", "GOOG", "INTC", "MSFT"]

# --- Data Constants ---
# Message file
MSG_COLS = ["time", "event_type", "order_id", "size", "price", "direction"]

# Orderbook file
DEPTH = 10
LEVELS = range(1, DEPTH + 1)
BOOK_COLS = [f"{s}_{k}_{l}" for l in LEVELS for s in ("ask", "bid") for k in ("px", "sz")]
ASK_COLS = [f"ask_{k}_{l}" for l in LEVELS for k in ("px", "sz")]
BID_COLS = [f"bid_{k}_{l}" for l in LEVELS for k in ("px", "sz")]
PX_COLS = [f"{s}_px_{l}" for l in LEVELS for s in ("ask", "bid")]
SZ_COLS = [f"{s}_sz_{l}" for l in LEVELS for s in ("ask", "bid")]
ASK_PX_COLS = [f"ask_px_{l}" for l in LEVELS]
ASK_SZ_COLS = [f"ask_sz_{l}" for l in LEVELS]
BID_PX_COLS = [f"bid_px_{l}" for l in LEVELS]
BID_SZ_COLS = [f"bid_sz_{l}" for l in LEVELS]
PX_TICK = 100

def load_books(ticker: str, resample: str | None = None) -> pd.DataFrame:
    raw_books = pd.read_csv(BOOK_PATH.format(ticker=ticker), header=None, names=BOOK_COLS)
    raw_msgs  = pd.read_csv(MSG_PATH.format(ticker=ticker),  header=None, names=MSG_COLS)

    is_valid = raw_msgs["event_type"] <= 4
    books = raw_books[is_valid]
    books[PX_COLS] //= PX_TICK
    books.index = pd.to_timedelta(raw_msgs.loc[is_valid, "time"], unit="s")
    books = books[(books[SZ_COLS] > 0).all(axis=1)]

    if resample:
        books = books.resample(resample).last().ffill()

    return books.astype("int64")

def build_features(books: pd.DataFrame) -> pd.DataFrame:
    ask_px = books[ASK_PX_COLS]
    bid_px = books[BID_PX_COLS]

    mid2 = ask_px.iloc[:, 0] + bid_px.iloc[:, 0]   # 2 x mid, stays an integer
    mid_lo = mid2 // 2      # floor(mid)
    mid_hi = -(-mid2 // 2)  # ceil(mid)

    feats = {
        "mid2_change": mid2.diff(),
        "ask_gap_1": ask_px.iloc[:, 0] - mid_lo,
        "bid_gap_1": mid_hi - bid_px.iloc[:, 0],
    }
    for l in range(2, DEPTH + 1):
        feats[f"ask_gap_{l}"] = books[f"ask_px_{l}"] - books[f"ask_px_{l - 1}"]
        feats[f"bid_gap_{l}"] = books[f"bid_px_{l - 1}"] - books[f"bid_px_{l}"]
    for c in SZ_COLS:
        feats[c] = books[c]

    # First row has no previous mid, so mid2_change is NaN there
    return pd.DataFrame(feats).iloc[1:].astype("int64")

