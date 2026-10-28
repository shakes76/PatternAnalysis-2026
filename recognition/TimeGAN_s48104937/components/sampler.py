import numpy as np
from torch.utils.data import Sampler


class BatchSampler(Sampler):
    """Batches that each hold windows of a single ticker, in shuffled order.

    Every epoch, each ticker's windows are shuffled and cut into full batches
    (the remainder is dropped, keeping batch statistics stable), then the
    batches of all tickers are shuffled together.
    """
    def __init__(self, labels: np.ndarray, batch_size: int, seed: int = 0):
        self.groups = [np.flatnonzero(labels == k) for k in np.unique(labels)]
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        batches = []
        for idx in self.groups:
            idx = self.rng.permutation(idx)
            n_full = len(idx) // self.batch_size * self.batch_size
            batches.extend(np.split(idx[:n_full], n_full // self.batch_size))
        for i in self.rng.permutation(len(batches)):
            yield batches[i].tolist()

    def __len__(self) -> int:
        return sum(len(idx) // self.batch_size for idx in self.groups)
