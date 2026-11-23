import tensorflow as tf
import numpy as np

from modules import timegan

if __name__ == "__main__":
    # dummy time series data with same format as lobster depth 10 dataset
    row = 30
    col = 24
    depth = 10
    mock_ori_data = np.random.rand(row, col, depth)

    parameters = {
        'hidden_dim': 16,
        'num_layer': 2,
        'iterations': 5,   
        'batch_size': 8,
        'module': 'gru',
    }

    # run TimeGAN function
    synthetic_data = timegan(mock_ori_data, parameters)
    print(f"length: {len(synthetic_data)}")
    print(f"shape: {synthetic_data[0].shape}")
    print(mock_ori_data)
    print("new line")
    print(synthetic_data)