# TimeGAN for AMZN Limit Order Book Simulation

Edward Tennyson, s46482079

## Feasibility review

### 1. User need, scope and acceptance criteria

Quant Researchers that specialise in market microstructure research often find themselves needing to model realistic market data to test trading algorithms. Under controlled conditions, it is extremely useful to have accurate simulations at hand that can do this. The purpose of this project is to investigate whether a TimeGAN model can reproduced Amazon Level-10 orderbook behaviour more accurately than a simple autoregressive basline while maintaing valid market conditions.

The prototype will use the given LOBSTER sample from 21 June 2012 with 32 market updates, meaing that one training example will contain 32 consecutive recorded market events (i.e. a new order, trade, cancellation etc.) Genereated price changes will be reconstructed using supplied beginning prices with results limited by trading day.

The proposed effectiveness critera include:

    1. KL divergence ≤ 0.1 of spreads and price mid-point returns when comparing real and generated distributions.
    2. Heatmap depth structural similarity SSIM > 0.6.
    3. No observed structural violations in outputs, including: crossed orderbooks, invalid price levels, negative quantities.
    4. Improvement in volatility clustering compared to the baseline, which will be measured by a lower error (against the real autocorrelation return with lags 1-10).
    5. Peak GPU memory uses ≤ 7 GiB on this computer's RTX 4060 (which has 8GB VRAM). Training time and generation throughput will be recorded to measure efficiency benchmarks.

### 2. Model choice and course concepts

TimeGAN is the selected model according the task sheet. It combines a 'learned embedding with supervised temporal learning and adversarial training' (Yoon, Jarrett, van der Scharr, 2019). The model will be used to investigate the quantified temporal patterns in orderbook data can re-produce structurally valid orderbooks according the pre-defined market conditions. Relevant course concepts include recurrent networks, generative adversial learning, and representation learning.

An implementation in PyTorch will be compared with a simple first-order autogressive (AR) model/generator per sampled residuals in the data. Data splits will be the same for each model, following 70/15/15% split between training, validation, and testing. To simplify the data mathematically, which is done commonly across quantitative research desks, we will take the logarithm of prices to calculate log returns with scaling fit to the training data.

### 3. Preliminary feasibility evidence

The initial audit verified 269748 aligned message and order-book rows, with 40 orderbook features (10 orderbook levels per bid/ask side). The checks found no missing price levels, invalid spreads (non-positive), incorrectly ordered price ladders or invalid quantities. Proprocessing produced 5507, 1081, and 1839 training, validation, and test sequences respectfully. This is resulting of the 70/15/15% day-split. The first 70% of the day is used for training, next 15% for validation, and final 15% for testing. Each complete sequency contains 32 steps by 40 features, ready for the model.

The next initial check was the RTX 4060, which successfully was able to process a batch containing 64 sequences of market data. These results demonstrate working and audited data and GPU compatability.

### 4. Risks, budget and fallback

The main risks with the model include unstable adversarial training, repetitive outputs, structural violations (of orderbook constraints), and unrealistic computing times. Also, the dataset covers 1 trading day of Amazon, limiting conclusions about extending the analysis to other stocks and market conditions that the data wasn't able to capture.

Initial experimenting will be done using the RTX 4060, which will be capped to a memory budget of 7 out of the 8 GiB VRAM available. Next steps will include running small TimeGANS to measure training stability, memory usage, and throughput. If memory requirements are excessive, batch size and hidden-layer width will need to be reduced, and computing on rangpur will be considered of local requirements are not enough.

Finally, for fallback, if the generated sequences are unrealistic, the baselines AR comparison and structural diagnostics will be utilised to identify the limitations of the TimeGAN and assist in making a final recommendation for backtesting in quantitative research.

### References

https://proceedings.neurips.cc/paper_files/paper/2019/file/c9efe5f26cd17ba6216bbe2a7d26d490-Paper.pdf
