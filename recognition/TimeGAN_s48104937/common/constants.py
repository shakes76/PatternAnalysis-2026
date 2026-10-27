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
