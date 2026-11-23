import os
import time
import torch
import matplotlib.pyplot as plt

from modules import TimeGAN
from dataset import get_lobster_timeseries_dataset

def main():
    # hyperparameters (change depending on hardware)
    BATCH_SIZE = 64
    SEQ_LEN = 24
    HIDDEN_DIM = 32
    NOISE_DIM = 32
    NUM_LAYERS = 2
    EPOCHS = 10 
    NROWS=10000
    train_rate = 0.8
    
    # put your pathes for the dataset here
    orderbook_path = 'LOBSTER_SampleFile_AMZN_2012-06-21_10/AMZN_2012-06-21_34200000_57600000_orderbook_10.csv'
    # message csv not used for this implementation, but can be used for more complex implementations
    message_path = 'LOBSTER_SampleFile_AMZN_2012-06-21_10/AMZN_2012-06-21_34200000_57600000_message_10.csv'

    # message_path = '/home/Student/s4883721/timeGAN-LOB/amazon_message.csv'
    # orderbook_path = '/home/Student/s4883721/timeGAN-LOB/amazon_orderbook.csv'

    # use gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    # load and preprocess LOBSTER dataset
    print('loading dataset')
    dataset_start = time.perf_counter()
    train_data, test_data, train_time, test_time = get_lobster_timeseries_dataset(
        orderbook_path=orderbook_path,
        seq_len=SEQ_LEN,
        train_rate=train_rate,
        nrows=NROWS
    )
    dataset_end = time.perf_counter()
    
    input_dim = train_data.size(-1)

    # initialise timeGAN model
    model = TimeGAN(
        input_dim=input_dim, 
        hidden_dim=HIDDEN_DIM, 
        noise_dim=NOISE_DIM, 
        num_layers=NUM_LAYERS
    ).to(device)

    # move training tensor to target device
    train_data = train_data.to(device)

    start = time.perf_counter()
    
    # train the model
    print('starting training')
    train_start = time.perf_counter()
    model.train_timegan(train_data, epochs=EPOCHS, batch_size=BATCH_SIZE)
    train_end = time.perf_counter()

    # generate synthetic time-series data
    print('generating synthetic sequences')
    generate_start = time.perf_counter()
    synthetic_data = model.generate(
        num_samples=40,
        seq_len=SEQ_LEN,
        device=device
    )
    generate_end = time.perf_counter()

    os.makedirs('output', exist_ok=True)
    torch.save(synthetic_data, 'output/synthetic_lob_train.pt')
    torch.save(test_data, 'output/real_lob_test.pt')

    end = time.perf_counter()
    print(f"Pipeline finished in {end - start:.2f} seconds.")
    print("Synthetic Data Shape:", synthetic_data.shape)

    print(f"dataset finished in {dataset_end - dataset_start:.2f} seconds")
    print(f"training finished in {train_end - train_start:.2f} seconds")
    print(f"generating finished in {generate_end - generate_start:.2f} seconds")
    
    plt.figure(figsize=(12, 5))
    
    # plot spread comparison
    plt.plot(train_data[0].mean(dim=1).detach().cpu(), label='Real Mean Depth', color='blue', alpha=0.7)
    plt.plot(synthetic_data[0].mean(dim=1).detach().cpu(), label='Synthetic Mean Depth', color='orange', alpha=0.7)
    plt.title('Real vs Synthetic Depth Profile')
    plt.xlabel('Time Step')
    plt.ylabel('Normalised Volume')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('output/train_metrics_comparison.png', dpi=300)
    plt.close()

    print("training pipeline finished")

if __name__ == '__main__':
    main()