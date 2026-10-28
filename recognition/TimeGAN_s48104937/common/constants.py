import torch

# Book constants
TICKERS = ["AMZN", "AAPL", "GOOG", "INTC", "MSFT"]
DEPTH = 10
LEVELS = range(1, DEPTH + 1)

# Unit ticks
PX_TICK: int = 100
SZ_TICK: int = 1

# PyTorch constants
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Directories/Paths
FIGURES_DIR = "figures"
DATA_DIR = "data"
BOOK_PATH = "data/{ticker}_2012-06-21_34200000_57600000_orderbook_10.csv"
MSG_PATH = "data/{ticker}_2012-06-21_34200000_57600000_message_10.csv"
