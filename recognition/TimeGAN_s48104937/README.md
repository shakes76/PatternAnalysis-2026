# TimeGAN LOB Simulation

## Project Overview

The original TimeGAN architecture was in fact extended however, with conditioning per ticker, and a minimal 1D Wasserstein distance loss to match the shape of real feature distrbution.

![timegan_architecture](figures/final/misc/timegan_architecture.png)

## Dataset and Preprocessing

#### LOBSTER Dataset

We only have access to the LOBSTER sample data, which covers a single trading day (21 June 2012) of tick-by-tick Level 10 order book snapshots and the accompanying messages for five tickers. An initial audit, however, shows that we have to trim our usable data even further if we want chronological splits to evaluate against, due to the extreme regime changes in both price and volume behaviour at the open and close. Thus the first and final 15 minutes of each day are trimmed.

![data_split](figures/final/misc/data_split.png)

Even with trimming there will be differences between the training and evaluation periods with spread declining through the day, unfortunately this is an unavoidable limitation of chronological splits on a single day.

Another change which reduces the available training data is resampling. Since each LOBSTER snapshot is the outcome of a single event each step has only 1 true change and steps are not evenly spread over time. Resampling to 1s partially addresses this, however, we reduce our total usable data from 1,823,146 rows to 108,000.

The final key observation with including the five ticks is the distinct behaviour of small-tick (AAPL, AMZN, GOOG) and large-tick (INTC, MSFT) stocks. With large-tick stocks having their spread pinned at one tick and a mid price that rarely moves, while small-tick stocks change price far more often.
![regime_behaviour](figures/final/misc/regime_behaviour.png)

#### Feature Engineering

Through the baseline test its clear LOB violations are common under naive modelling, thus the major change made when moving to TimeGAN will be to explicitly design the features to make the order book valid by construction.

![feature_engineering](figures/final/misc/feature_engineering.png)

explain we use mid-price relative to start of window, (briefly explain spread and gap system (images does most)), explain what we did with volume (not much). explain that we needed to add flats due to jitter.

Rather than modelling the price of each level, we model the mid-price relative to the start of the window, the spread, and the gap from each quote to the one preceding it. The drawback of this representation is having to pass reference price to anchor synthetic windows on. Another major change made halfway through is a flag for each feature allowing it to update, reasoning and impacts of which are discussed in Development.

The scaler used to fit the features for the model was a custom implementation of a Signed-log compression followed by per-feature min-max scaling to [0, 1]. This was an appropriate option since....

## Development

#### Feature Flags

This was included as initial tests without this flag resulted in unrealistic jitter with synthetic features changing in XXX% of steps compared to XX% of steps in the real data.

| Unchanged features per step | Real  | Flags on | Flags off |
|-----------------------------| ----: |---------:|----------:|
| Top prices                  | 80.5% |    75.9% |     57.7% |
| All prices                  | 66.7% |    69.3% |     36.7% |
| All sizes                   | 68.1% |    72.5% |  **3.0%** |

We can also visually inspect this difference by comparing a synthetic sample with and without the feature flags. As expected this leads to significant jitters caused by...

![flags_comparison](figures/final/flags_comparison.png)

#### Grouped MSE Loss

Adding the previously discussed flags led to a collapse in variation for the synthetic windows. However, this was not mode collapse in the GAN since the cause this was found to be the autoencoder since now critical features like spread, mid, and gaps were having to compete with the flags within the MSE loss.

The reason competing with the flags was such a problem is that real flags are 0 or 1, so a missed flag costs up to 1 in squared error while the value features typically cost around 0.01. To fix this the embedder's MSE loss was tuned to be weighted by feature group, giving each group an equal share of the loss.

| Group     | Channels | Relative weight per channel (vs default MSE) |
| --------- | --------:| --------------------------------------------:|
| `mid_rel` | 1        | 16×                                          |
| `spread`  | 1        | 16×                                          |
| Gaps      | 18       | 0.9×                                         |
| Sizes     | 20       | 0.8×                                         |
| Flags     | 40       | 0.4×                                         |

Comparing this new autoencoder trained with a grouped loss shows a clear improvement, MSE across all inputs dropping from 0.018 to 0.003, though most importantly R² for mid and spread improved from 0.57 and 0.18 to 0.997 and 0.97 respectively. Flag accuracy also improved from 97.1% to 99.75%, despite the flags' lower weight. There still however is room to improve since R² for gaps and sizes got better but only improving to 0.69 and 0.72 from 0.12 and 0.06 respectively.

![autoencoder_comparison](figures/final/autoencoder_comparison.png)

#### 1D Wasserstein Loss

Finally, a 1D Wasserstein loss was added to encourage the fat tails in the real spread and mid-price return distributions. At a weight of 10, a new failure mode appeared: the mid-price stayed within a narrow band while the best quotes moved symmetrically in opposite directions, likely because the Wasserstein term outweighed the discriminator. A weight of 1 avoided this, as the metrics below show.

| Metric                              | Real  | Weight ×1 | Weight ×10 |
| ----------------------------------- | -----:| ---------:| ----------:|
| Mid-price range per window (¢)      | 32.99 | 29.44     | 7.19       |
| Lag-1 volatility clustering         | 0.141 | 0.137     | 0.037      |
| Price level spacing variability (¢) | 1.22  | 1.21      | 0.82       |
| Best quotes  co-movement            | 0.37  | 0.57      | −0.29      |

We can also visually see this difference by comparing a sample generated from the x10 and x10 models, where the symmetric jump behaviour where the best quote move in opposite diractions on the same step is clear.

![wasserstien_windows_comparison](figures/final/wasserstien_loss_comparison.png)

## Evaluation

talk somewhere about how we didn't mode collapse

#### Training and Stability

Mention how I used adam optimser

show loss curves and talk about discriminator going back and forth and stuff. Also talk about stability here (changing seed does not hurt it, etc)

#### Synthetic Windows and Generation

Do things like just look at nice synthetic windows, compare large and small tick stocks. etc.


Long range synthetic generation.

![timegan_long_range_sample](figures/final/timegan_long_range_sample.png)

#### Metrics and Baseline Comparison

Due to the baseline models naive design choices leading to consistent LOB violations it's instantly clear that the developed TimeGAN setup is a superiour option. However the GRU has the advantage of being autoregressive which means it could do better at some metrics (EXPLAIN FURHTER).

Comparing the two models against our acceptance criteria from the feasibility review we see that...

SSID

| Ticker | SSIM Real vs Real | SSIM Real vs Synthetic | SSIM Ratio |
| :----- | ----------------: | ---------------------: | ---------: |
| AMZN   | 0.115 ± 0.002     | 0.089 ± 0.006          |       0.78 |
| AAPL   | 0.049 ± 0.002     | 0.071 ± 0.002          |       1.44 |
| GOOG   | 0.105 ± 0.003     | 0.085 ± 0.003          |       0.81 |
| INTC   | 0.525 ± 0.006     | 0.544 ± 0.003          |       1.04 |
| MSFT   | 0.585 ± 0.005     | 0.581 ± 0.007          |       0.99 |

**AMZN: model comparison**

| Model   | Spread KL ↓   | ∆Mid KL ↓     | ∆VWAP KL ↓    | ∆VWAP Autocorr. | Depth SSIM Ratio ↑ |
| :------ | ------------: | ------------: | ------------: | --------------: | -----------------: |
| AR-GRU  |               |               |               |                 |                    |
| TimeGAN | 0.086 ± 0.011 | 0.022 ± 0.002 | 0.052 ± 0.004 |                 |                    |

**TimeGAN across tickers**

| Ticker   | Spread KL ↓   | ∆Mid KL ↓     | ∆VWAP KL ↓    | ∆VWAP Autocorr. | Depth SSIM Ratio ↑ |
| :------- | ------------: | ------------: | ------------: | --------------: | -----------------: |
| AMZN     | 0.086 ± 0.011 | 0.022 ± 0.002 | 0.052 ± 0.004 |                 |                    |
| AAPL     | 0.133 ± 0.013 | 0.050 ± 0.010 | 0.046 ± 0.001 |                 |                    |
| GOOG     | 0.030 ± 0.003 | 0.018 ± 0.001 | 0.043 ± 0.001 |                 |                    |
| INTC     | 0.107 ± 0.063 | 0.079 ± 0.023 | 0.277 ± 0.012 |                 |                    |
| MSFT     | 0.051 ± 0.051 | 0.084 ± 0.032 | 0.365 ± 0.017 |                 |                    |
| **Mean** | 0.081 ± 0.018 | 0.051 ± 0.010 | 0.157 ± 0.003 |                 |                    |

*± is the standard deviation over N seeds. ↓ lower is better, ↑ higher is better.*


Info: Model, SSIM ratio, KL spread, KL mid returns, VWAP returns, Lagged VWAP autocorrelation

Other

...


#### Structural Fails

Despite being a clear improvement over the baseline model and boasting a 0% LOB violation rate our TimeGAN model still has several structural pitfalls. 

- **Price Volatility at Depth:** Synthetic changes its px at depth a lot more than real
- **Volume Persistence and Clarity:** Synthetic fails to properly generalize the relation between and actual price of a level and the volume there over time.
- **Tick by Tick Jitter:** Note that this may be too linked to the prev two fails, thus consider dropping.
- **Lack of Diverse Training Data:** We basically only had a single day, which even if the model was a perfect fit would not give a good trust worthy representation of the market from the perspective of a quantitative reseracher.
- 

## Recommendations

#### Viability in Industry

Do not deploy, firstly it just is not accurate enough.

Since the time gan architecture generates an entire window in an instant it makes it so that if we try to test from trading algorithm on it, its not much better then just using historic data, as for many HFT (high-frequency trading) algorithms its not just what the market is doing that is important, but also how the market reacts to your algorhitms actions.

Simply need more data

#### Alternative Architecture

Had the constraint of implementing a TimeGAN model not been in place, there would be the potential to develop a far better model. The alternative is a maximum-likelihood event model conditioned on the current L10 book and history, outputting actual market events (order submissions, cancellations and executions) to a matching engine maintaining the LOB. A GAN-like model could also be used in this system to generate new level 10 quotes when the matching engine runs out of quotes on one side, such as when a level is cancelled or executed.

This approach would be far superior, as all market behaviour that our TimeGAN is attempting to emulate is emergent from this underlying event process. Furthermore, it would let the quantitative researcher see how the market reacts to their own trades in backtests, since any order the researcher's trading algorithm places can be fed to the event model like any other event.

Recent literature points towards...

I believe that any model developed should fundamentally be conditioned on the previous book such that model can correct drifts to spread, volume, gaps, and more happening to the book over time.

## Setup and Usage

#### Dependencies

Developed with Python 3.11. PyTorch is installed from its CUDA 13.0 index, since a plain `pip install` gives the CPU-only build:

```bash
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
pip install numpy==2.4.6
pip install pandas==3.1.0
pip install matplotlib==3.11.2 
pip install scipy==1.17.1 
pip install scikit-learn==1.9.1 
pip install scikit-image==0.26.0
```

or alternatively, `pip install -r requirements.txt` after installing PyTorch.

#### Reproducibility

All parameters are set in `config.toml` in the project folder. Running `train.py` downloads dataset (about 580 MB) if not already present, then trains the model and saves its weights. Running `predict.py` evaluates the saved model and saves figures.

```bash
cd recognition/TimeGAN_s48104937
python train.py      # download data (first run only), train, save weights
python predict.py    # evaluate the saved model, save figures
```

All random number generators are seeded from `config.toml`, with each modelling storing the configuration it was trained with so runs are repeatable. Note small differences between runs or GPUs are still possible, because cuDNN's GRU kernels are not fully deterministic.

## Artificial Intelligence Usage Disclosure

Helping prepose causes to structural failures and potential remedies.

Helping edit and provide feedback on readme and documentation.

Minimal autocomplete with PyCharm built-in copilot.

AI was used to write and help with sections of the README, but all text was read, understood, edited, and clarified by myself.