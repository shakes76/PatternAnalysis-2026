# --- Data files ---
BOOK_PATH = "data/{ticker}_2012-06-21_34200000_57600000_orderbook_10.csv"
MSG_PATH = "data/{ticker}_2012-06-21_21_34200000_57600000_message_10.csv"
TICKERS = ["AMZN", "AAPL", "GOOG", "INTC", "MSFT"]

# --- Data Constants ---
# Message file
MSG_COLS = ["time", "event_type", "order_id", "size", "price", "direction"]

# Orderbook file
DEPTH = 10
LEVELS = range(1, DEPTH + 1)
BOOK_COLS = [
    f"{side}_{kind}_{l}"
    for l in LEVELS
    for side in ("ask", "bid")
    for kind in ("px", "sz")
]
ASK_COLS = [c for c in BOOK_COLS if c.startswith("ask_")]
BID_COLS = [c for c in BOOK_COLS if c.startswith("bid_")]
PX_COLS  = [c for c in BOOK_COLS if "_px_" in c]
SZ_COLS  = [c for c in BOOK_COLS if "_sz_" in c]