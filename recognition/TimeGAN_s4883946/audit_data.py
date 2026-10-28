"""Data audit for the LOBSTER AMZN Level-10 sample.

Checks the microstructure invariants of the *real* order book (so that any
violation in synthetic data can be attributed to the model), and summarises the
statistics that make the generation task hard (zero-return mass, heavy tails,
volatility clustering). Used as feasibility evidence in the README.

Usage: python audit_data.py <data_dir>
"""
import sys
import numpy as np

PREFIX = "AMZN_2012-06-21_34200000_57600000_"


def main(data_dir):
    ob = np.loadtxt(f"{data_dir}/{PREFIX}orderbook_10.csv", delimiter=",")
    # message columns: time, type, (order id skipped), size, price, direction
    msg = np.loadtxt(f"{data_dir}/{PREFIX}message_10.csv", delimiter=",",
                     usecols=(0, 1, 3, 4, 5))
    print("orderbook", ob.shape, "messages", msg.shape)

    # Orderbook columns repeat per level: ask price, ask size, bid price, bid size
    ask_p, ask_v = ob[:, 0::4], ob[:, 1::4]
    bid_p, bid_v = ob[:, 2::4], ob[:, 3::4]

    print("crossed or locked books:", int((bid_p[:, 0] >= ask_p[:, 0]).sum()))
    print("non-monotonic ask ladders:", int((np.diff(ask_p, axis=1) < 0).any(1).sum()),
          "bid ladders:", int((np.diff(bid_p, axis=1) > 0).any(1).sum()))
    print("negative depth:", int((ask_v < 0).sum() + (bid_v < 0).sum()),
          "zero depth:", int((ask_v == 0).sum() + (bid_v == 0).sum()))

    spread = (ask_p[:, 0] - bid_p[:, 0]) / 1e4  # prices are in 1/10000 dollars
    print("spread ($) min/median/mean/max:",
          spread.min(), np.median(spread), spread.mean(), spread.max())

    mid = (ask_p[:, 0] + bid_p[:, 0]) / 2e4
    ret = np.diff(np.log(mid))
    kurt = ((ret - ret.mean()) ** 4).mean() / ret.var() ** 2
    print("event log-return std:", ret.std(),
          "fraction zero:", (ret == 0).mean(), "kurtosis:", kurt)

    # Volatility clustering: autocorrelation of absolute returns
    a = np.abs(ret) - np.abs(ret).mean()
    acf = [float((a[:-k] * a[k:]).mean() / a.var()) for k in (1, 5, 20)]
    print("abs-return autocorrelation at lags 1/5/20:", [round(x, 3) for x in acf])

    t = msg[:, 0]
    print("time range (s after midnight):", t[0], t[-1],
          "duplicate-timestamp fraction:", (np.diff(t) == 0).mean())
    print("event types:", {int(k): int((msg[:, 1] == k).sum()) for k in np.unique(msg[:, 1])})
    print("depth median/max:", np.median(ask_v), ask_v.max())
    print("events per hour:", [int(((t >= h) & (t < h + 3600)).sum())
                               for h in range(34200, 57600, 3600)])


if __name__ == "__main__":
    main(sys.argv[1])
