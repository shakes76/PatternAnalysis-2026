import numpy as np
import joblib
import torch
from sklearn.preprocessing import StandardScaler

CENT = 100  # LOBSTER prices are dollars * 10000, so 1 cent = 100 units


def load_orderbook(path):
    # LOBSTER orderbook files have no header: 40 columns, one row per event
    return np.loadtxt(path, delimiter=",", dtype=np.float64)

def chronological_split(X, train=0.7, val=0.15):
    n = len(X)
    i, j = int(n * train), int(n * (train + val)) # calculate indexes for 70:15:15 split
    return X[:i], X[i:j], X[j:]

def make_features(X):
    X = np.asarray(X, dtype=np.float64)

    # ask prices start at the first column and increase in level every four columns
    # ask volumes start at the second column and increase in level every four columns
    # bid prices start at the third column and increase in level every four columns
    # bid volumes start at the fourth column and increase in level every four columns
    ask_p, ask_v = X[:, 0::4], X[:, 1::4]
    bid_p, bid_v = X[:, 2::4], X[:, 3::4]

    # calculate price gaps by finding difference between consecutive price columns
    ask_gap = np.diff(ask_p, axis=1) / CENT
    bid_gap = -np.diff(bid_p, axis=1) / CENT

    # calculate spread by finding difference between level 1 asks and level 1 bids
    spread = (ask_p[:, 0] - bid_p[:, 0]) / CENT

    # ensure gaps and spread are minimum of 1 tick
    ask_gap = np.maximum(ask_gap, 1.0)
    bid_gap = np.maximum(bid_gap, 1.0)
    spread = np.maximum(spread, 1.0)

    # log transform price gaps and volumes
    prices = np.log(np.concatenate([ask_gap, bid_gap], axis=1))
    volumes = np.log1p(np.concatenate([ask_v, bid_v], axis=1))

    # calculate mid-price: average between best ask and best bid
    mid = (ask_p[:, 0] + bid_p[:, 0]) / 2

    # Per-event midpoint log-return; prepend makes it length N (first value 0)
    ret = np.diff(np.log(mid), prepend=np.log(mid[0]))
    return prices, volumes, mid, np.log(spread), ret



class LOBNormaliser:
    def fit(self, X_train):
        P, V, _, S, R = make_features(X_train)

        # Calculate and store statistics for standard distribution
        self.p_scaler = StandardScaler().fit(P)  # Price gaps (18 features)
        self.v_scaler = StandardScaler().fit(V)  # Volumes (20 features)
        self.s_scaler = StandardScaler().fit(S[:, None])  # Spread (1 feature)
        self.r_scaler = StandardScaler().fit(R[:, None])  # Mid-price returns (1 feature)
        return self

    def transform(self, X):
        P, V, mid, spread, ret = make_features(X)
        # Transform data using corresponding statistics and concat into standardised feature matrix
        Z = np.concatenate([
            self.p_scaler.transform(P),
            self.v_scaler.transform(V),
            self.s_scaler.transform(spread[:, None]),
            self.r_scaler.transform(ret[:, None]),      # column 39
        ], axis=1).astype(np.float32)
        return Z, mid     # mid is needed for the inverse

    # reverse the normalisation transformations done above
    def inverse(self, Z, mid):
        Z = np.asarray(Z, dtype=np.float64)

        # undo standardisation and log transforms
        P = np.exp(self.p_scaler.inverse_transform(Z[:, :18]))
        P = np.maximum(P, 1.0) * CENT  # >= 1 tick
        V = np.clip(np.expm1(self.v_scaler.inverse_transform(Z[:, 18:38])), 0, None)
        spread = np.exp(self.s_scaler.inverse_transform(Z[:, 38:39]).ravel())
        spread = np.maximum(spread, 1.0) * CENT

        # Recover best ask and bid from midprice and spread
        ask_best = mid + spread / 2
        bid_best = mid - spread / 2

        # Accumulate price differences outwards from best prices
        ask_prices = np.column_stack([
            ask_best,
            ask_best[:, None] + np.cumsum(P[:, :9], axis=1)
        ])
        bid_prices = np.column_stack([
            bid_best,
            bid_best[:, None] - np.cumsum(P[:, 9:], axis=1)
        ])

        # create empty feature matrix (len(Z) rows, 40 columns)
        X = np.empty((len(Z), 40), dtype=np.float64)

        # put prices and volumes in
        X[:, 0::4], X[:, 1::4] = ask_prices, V[:, :10]
        X[:, 2::4], X[:, 3::4] = bid_prices, V[:, 10:]

        return X # back to LOB form

    # Reconstruct midprice path given midprice returns
    def mid_path(self, Z, mid0):
        Z = np.asarray(Z, dtype=np.float64)

        # midprice returns are from column 39 and reverse standardisation
        r = self.r_scaler.inverse_transform(Z[:, :, 39].reshape(-1, 1)).reshape(Z.shape[0], -1)

        # discard the first return value because we are given the midprice to start (mid0) and we
        # use every subsequent return value to calculate subsequent midprices
        # E.g. Event 1 uses mid0, event 2 uses mid0 + r[1], event 3 uses event 2's midprice + r[2],
        # etc.
        cum = np.cumsum(r, axis=1) - r[:, :1]

        # convert cumulative log returns abck into absolute midprice values
        return np.asarray(mid0)[:, None] * np.exp(cum)

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)


class LOBWindows(torch.utils.data.Dataset):
    def __init__(self, Z, mid, window=64, stride=8):
        if window <= 0 or stride <= 0:
            raise ValueError("window and stride must be positive")
        if len(Z) < window:
            raise ValueError("Not enough events for one window")

        self.Z = torch.from_numpy(Z)
        self.mid = torch.from_numpy(mid.astype(np.float64))
        self.window, self.stride = window, stride

        # calculate the number of windows we can create from the data
        self.n = (len(Z) - window) // stride + 1

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        s = i * self.stride
        return self.Z[s:s + self.window], self.mid[s:s + self.window]


def get_datasets(path, window=64, stride=8):
    X = load_orderbook(path) # load csv
    X_train, X_val, X_test = chronological_split(X) # split
    norm = LOBNormaliser().fit(X_train)             # train split only
    sets = []

    # create separate train, val and test datasets with windowing
    for part in (X_train, X_val, X_test):
        Z, mid = norm.transform(part)
        sets.append(LOBWindows(Z, mid, window, stride))
    return *sets, norm

# main
if __name__ == "__main__":
    X = load_orderbook("order_sample.csv")
    X_train, X_val, X_test = chronological_split(X)

    # Test preprocessing round-trip
    norm = LOBNormaliser().fit(X_train)
    Z, mid = norm.transform(X_val)
    X_reconstructed = norm.inverse(Z, mid)

    err = np.abs(X_reconstructed - X_val)
    print("Max price error:", err[:, 0::2].max())
    print("Max volume error:", err[:, 1::2].max())

    # Load datasets as usual
    train_set, val_set, test_set, norm = get_datasets("order_sample.csv")
    print(len(train_set), len(val_set), len(test_set))

    x, m = train_set[0]
    print(x.shape, m.shape)