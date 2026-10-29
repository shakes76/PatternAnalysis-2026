import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset

TICKERS = ["AAPL", "AMZN", "GOOG", "INTC", "MSFT"]


def stock_files(ticker, data_dir="data"):
    """Paths to one stock's LOBSTER orderbook and message files."""
    name = f"{data_dir}/{ticker}_2012-06-21_34200000_57600000"
    return name + "_orderbook_10.csv", name + "_message_10.csv"

def load_lobster(book_path, msg_path):
    """Return times (N,) in seconds after midnight, and book (N, 40)."""
    # read both CSVs 
    book_df = pd.read_csv(book_path, header=None)
    msg_df = pd.read_csv(msg_path, header=None)
    # times = column 0 of the message file, as a numpy array
    times = msg_df[0].to_numpy()
    # book = the whole orderbook file, as a numpy array
    book = np.array(book_df)
    return times, book

def check_rules(book):
    """Return the number of rows breaking each rule: spread, ladder, volume."""
    ask_price, ask_size = book[:, 0::4], book[:, 1::4]   # every 4th column, starting at 0 / 1
    bid_price, bid_size = book[:, 2::4], book[:, 3::4]
    
    # spread: best bid (col 0) must be below best ask (col 0)
    bad_spread = bid_price[:, 0] >= ask_price[:, 0]
    n_spread = bad_spread.sum()

    # ladder (ask): each level must be more expensive than the one before
    ask_gaps = np.diff(ask_price, axis=1) # ask2-ask1, ask3-ask2
    bad_ask = (ask_gaps <= 0).any(axis=1)
    
    # ladder (bid): each level must be cheaper than the one before
    bid_gaps = np.diff(bid_price, axis=1) # ask2-ask1, ask3-ask2
    bad_bid = (bid_gaps >= 0).any(axis=1)

    n_ladder = (bad_ask | bad_bid).sum()
    
    # volume: rows where any size < 0
    bad_volume = (ask_size < 0).any(axis=1) | (bid_size < 0).any(axis=1)
    n_volume = bad_volume.sum()

    return n_spread, n_ladder, n_volume

def resample(times, book, dt=1.0):
    """Book state at the end of every dt seconds from 9:30 to 16:00."""
    grid = np.arange(9.5 * 3600 + dt, 16 * 3600, dt)
    # for each grid time, the index of the LAST event at or before it
    idx = np.searchsorted(times, grid, side="right") - 1
    # return grid, book[those indices]
    return grid, book[idx]
    # empty seconds reuse the last event's book: searchsorted finds the same index again ("carry forward")

def encode(snapshots):
    """Snapshots (T, 40) -> features (T-1, 41): return, 20 price offsets, 20 log sizes."""
    ask_price, ask_size = snapshots[:, 0::4], snapshots[:, 1::4]
    bid_price, bid_size = snapshots[:, 2::4], snapshots[:, 3::4]

    mid = (ask_price[:, 0] + bid_price[:, 0]) / 2
    log_ret = np.diff(np.log(mid))

    # drop the first row from mid, ask_price, bid_price, ask_size, bid_size
    mid = mid[1:]
    ask_price = ask_price[1:]
    bid_price = bid_price[1:]
    ask_size = ask_size[1:]
    bid_size = bid_size[1:]   

    # price offsets in ticks: how far each level sits from the mid
    # 1 tick = 1 cent = 100 in LOBSTER units
    ask_off = (ask_price - mid[:, None]) / 100
    bid_off = (bid_price - mid[:, None]) / 100

    # sizes: log squashes 1..34,000 into a small range
    log_sizes = np.log(np.concatenate([ask_size, bid_size], axis=1))

    # glue everything side by side: 1 + 10 + 10 + 20 = 41 columns
    return np.concatenate([log_ret[:, None], ask_off, bid_off, log_sizes], axis=1)

def split_by_time(grid, features):
    """Split features into train / val / test by time of day."""
    t = grid[1:]                      # times that line up with the feature rows

    train = features[t < 14 * 3600]
    val   = features[(t >= 14 * 3600) & (t < 15 * 3600)]
    test  = features[t >= 15 * 3600]
    return train, val, test

def normalise(train, val, test):
    """Scale each column to mean 0, std 1 using TRAINING statistics only."""
    mean = train.mean(axis=0)          # one mean per column → shape (41,)
    std = train.std(axis=0) + 1e-8     # + tiny number: avoids dividing by 0 if a column never changes

    train = (train - mean) / std
    val = (val - mean) / std
    test = (test - mean) / std
    return train, val, test, mean, std

class LOBWindows(Dataset):
    """Cuts a (rows, 41) array into windows of seq_len consecutive rows."""

    def __init__(self, data, seq_len=60, stride=1, stock_id=0):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.seq_len = seq_len
        self.starts = range(0, len(data) - seq_len + 1, stride)
        self.stock_id = stock_id

    def __len__(self):
        return len(self.starts)
    
    def __getitem__(self, i):
        start = self.starts[i]
        return self.data[start:start + self.seq_len], self.stock_id

def decode(features, mean, std, start_mid):
    """Features (T, 41) -> order book (T, 40): the exact reverse of normalise + encode.
    The mid-price path is rebuilt by compounding the log-returns from start_mid."""
    x = features * std + mean

    log_ret = x[:, 0]                         # column 0
    ask_off = x[:, 1:11]                      # columns 1-10
    bid_off = x[:, 11:21]                     # columns 11-20
    log_sizes = x[:, 21:41]                   # columns 21-40

    mid = start_mid * np.exp(np.cumsum(log_ret))     
    ask_price = mid[:, None] + ask_off * 100                           
    bid_price = mid[:, None] + bid_off * 100
    sizes = np.exp(log_sizes)                               

    book = np.zeros((len(x), 40))
    book[:, 0::4] = ask_price                 # put everything back in LOBSTER's column order
    book[:, 1::4] = sizes[:, :10]                       # ask sizes = first 10 columns of sizes
    book[:, 2::4] = bid_price
    book[:, 3::4] = sizes[:, 10:]                       # bid sizes = last 10 columns of sizes
    return book

def make_loaders(tickers=TICKERS, seq_len=500, batch_size=128, eval_stride=50):
    """All stocks -> train/val/test DataLoaders of (window, stock label), plus the shared mean and std."""
    # stage 1: each stock through the chain separately
    parts = []                                       # one (train, val, test) per stock
    for ticker in tickers:
        times, book = load_lobster(*stock_files(ticker))                            # load this stock (stock_files helps)
        grid, snapshots = resample(times, book)
        features = encode(snapshots)
        parts.append(split_by_time(grid, features))

    # stage 2: one shared normaliser from every stock's TRAINING rows
    all_train = np.concatenate([train for train, val, test in parts])
    mean = all_train.mean(axis=0) + 1e-8                                       # same as in normalise()
    std = all_train.std(axis=0) + 1e-8

    # stage 3: windows, labelled with each stock's number
    train_sets, val_sets, test_sets = [], [], []
    for stock_id, (train, val, test) in enumerate(parts):
        train_sets.append(LOBWindows((train - mean) / std, seq_len, stride=1, stock_id=stock_id))
        val_sets.append(LOBWindows((val - mean) / std, seq_len, stride=eval_stride, stock_id=stock_id))                         # val rows, stride=eval_stride
        test_sets.append(LOBWindows((test - mean) / std, seq_len, stride=eval_stride, stock_id=stock_id))                        # test rows, stride=eval_stride

    train_loader = DataLoader(ConcatDataset(train_sets), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ConcatDataset(val_sets), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(ConcatDataset(test_sets), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, mean, std

if __name__ == "__main__":
    for ticker in TICKERS:
        times, book = load_lobster(*stock_files(ticker))
        dummies = (np.abs(book) >= 9999999999).sum()
        print(ticker, "| events:", len(book), "| rule breaks:", check_rules(book), "| dummy levels:", dummies)

    book_file, msg_file = stock_files("AMZN")

    times, book = load_lobster(book_file, msg_file)
    print("events:", len(times), "| book shape:", book.shape)

    n_spread, n_ladder, n_volume = check_rules(book)
    print("rule violations (spread, ladder, volume):", n_spread, n_ladder, n_volume)

    grid, snapshots = resample(times, book)
    print("snapshots:", len(snapshots))

    features = encode(snapshots)
    print("features:", features.shape)
    print("ask offsets all positive:", (features[:, 1:11] > 0).all())
    print("bid offsets all negative:", (features[:, 11:21] < 0).all())

    train, val, test = split_by_time(grid, features)
    print("train / val / test rows:", len(train), len(val), len(test))

    train, val, test, mean, std = normalise(train, val, test)
    print("train mean ~0:", train.mean().round(3), "| train std ~1:", train.std().round(3))
    print("val mean:", val.mean().round(3), "| val std:", val.std().round(3))

    train_ds = LOBWindows(train, stride=1)
    val_ds = LOBWindows(val, stride=60)
    test_ds = LOBWindows(test, stride=60)
    print("windows (train / val / test):", len(train_ds), len(val_ds), len(test_ds))

    loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    windows, labels = next(iter(loader))                                    # CHANGED: a batch is now a pair
    print("one batch:", windows.shape, "| labels:", labels[:5])

    train_loader, val_loader, test_loader, mean, std = make_loaders()
    print("windows (train / val / test):", len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))
    windows, labels = next(iter(train_loader))
    print("5-stock batch:", windows.shape, "| labels:", labels[:10])
    
    start_mid = (snapshots[0, 0] + snapshots[0, 2]) / 2
    rebuilt = decode((features - mean) / std, mean, std, start_mid)
    print("round-trip max error:", np.abs(rebuilt - snapshots[1:]).max())