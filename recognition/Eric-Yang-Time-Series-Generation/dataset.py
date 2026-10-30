import pandas as pd
import torch

def load_lobster_csv(orderbook_path: str, nrows: int = None) -> torch.Tensor:
    """load the orderbook CSV file and computes features"""
    obs_df = pd.read_csv(orderbook_path, header=None, nrows=nrows)
    obs_tensor = torch.tensor(obs_df.values, dtype=torch.float32)

    ask_price_1 = obs_tensor[:, 0] / 10000.0
    bid_price_1 = obs_tensor[:, 2] / 10000.0
    mid_price = (ask_price_1 + bid_price_1) / 2.0
    
    returns = torch.zeros_like(mid_price)
    returns[1:] = torch.log(mid_price[1:] / (mid_price[:-1] + 1e-8))
    spread = (ask_price_1 - bid_price_1)

    # Combine returns, spread, and orderbook data columns
    combined = torch.cat([
        returns.unsqueeze(1),
        spread.unsqueeze(1),
        obs_tensor
    ], dim=1)
    
    return combined

def preprocess_lob_data(raw_data: torch.Tensor) -> torch.Tensor:
    """normalises limit order book data"""
    data_min = raw_data.min(dim=0, keepdim=True).values
    data_max = raw_data.max(dim=0, keepdim=True).values
    processed_data = (raw_data - data_min) / (data_max - data_min + 1e-8)
    return processed_data

def create_sliding_windows(data: torch.Tensor, seq_len: int = 20, step: int = 1) -> torch.Tensor:
    """creates sliding windows"""
    sequences = []
    total_len = data.size(0)
    if total_len < seq_len:
        return torch.empty(0, seq_len, data.size(1))
    for i in range(0, total_len - seq_len + 1, step):
        sequences.append(data[i : i + seq_len])
    return torch.stack(sequences) if sequences else torch.empty(0)

def extract_time_tensor(data_tensor: torch.Tensor):
    """returns valid time tensors for sliding windows"""
    batch_size = data_tensor.size(0)
    seq_len = data_tensor.size(1)
    time_tensor = torch.full((batch_size,), seq_len, dtype=torch.int32)
    return time_tensor, seq_len

def get_lobster_timeseries_dataset(orderbook_path: str, seq_len: int = 20, train_rate: float = 0.8, nrows: int = None):
    """loads and preprocesses the LOBSTER dataset"""
    raw_data = load_lobster_csv(orderbook_path, nrows=nrows)
    processed_data = preprocess_lob_data(raw_data)
    sequences_tensor = create_sliding_windows(processed_data, seq_len=seq_len)

    split_idx = int(sequences_tensor.size(0) * train_rate)
    train_data = sequences_tensor[:split_idx]
    test_data = sequences_tensor[split_idx:]
    
    train_time, _ = extract_time_tensor(train_data)
    test_time, _ = extract_time_tensor(test_data)
    
    return train_data, test_data, train_time, test_time