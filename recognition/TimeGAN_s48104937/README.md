# TimeGAN LOB Simulation

## Project Overview

The original TimeGAN architecure was infact extended however, with conditioning per ticker, and a minimal 1D Wasserstein distance loss to match the shape of real feature distrbution.

![timegan_architecture](figures/final/timegan_architecture.png)

## Dataset and Preprocessing

#### LOBSTER Dataset

We only have access to the LOBSTER sample data, which covers a single trading day (21 June 2012) of tick-by-tick Level 10 order book snapshots and the accompanying messages for five tickers. An initial audit, however, shows that we have to trim our usable data even further if we want chronological splits to evaluate against, due to the extreme regime changes in both price and volume behaviour at the open and close. Thus the first and final 15 minutes of each day are trimmed.

![data_split](figures/final/data_split.png)

Even with trimming there will be differences between the training and evaluation periods with spread declining through the day, unfortunately this is an unavoidable limitation of chronological splits on a single day.

Another change which reduces the available training data is resampling. Since each LOBSTER snapshot is the outcome of a single event each step has only 1 true change and steps are not evenly spread over time. Resampling to 1s partially addresses this, however, we reduce our total usable data from 1,823,146 rows to 108,000.

The final key observation with including the five ticks is the distinct behaviour of small-tick (AAPL, AMZN, GOOG) and large-tick (INTC, MSFT) stocks. With large-tick stocks having their spread pinned at one tick and a mid price that rarely moves, while small-tick stocks change price far more often.
![regime_behaviour](figures/final/regime_behaviour.png)

#### Feature Engineering

Through the baseline test its clear LOB violations are common under naive modelling, thus the major change made when moving to TimeGAN will be to explicitly design the features to make the order book valid by construction.

![feature_engineering](figures/final/feature_engineering.png)

explain we use mid-price relative to start of window, (briefly explain spread and gap system (images does most)), explain what we did with volume (not much). explain that we needed to add flats due to jitter.

Rather than modelling the price of each level, we model the mid-price relative to the start of the window, the spread, and the gap from each quote to the one preceding it. The drawback of this representation is having to pass reference price to anchor synthetic windows on. Another major change made halfway through is a flag for each feature allowing it to update, reasoning and impacts of which are discussed in Development.

The scaler used to fit the features for the model was a custom implementation of a Signed-log compression followed by per-feature min-max scaling to [0, 1]. This was an appropriate option since....

## Development

#### Multi-Ticker Conditioning

talk about batch sampler and how/why i chose to train the way i did.

#### Feature Flags

This was included as initial tests without this flag resulted in unrealistic jitter with synthetic features changing in XXX% of steps compared to XX% of steps in the real data.

#### Grouped MSE Loss

Adding the previously discussed flags led to a collapse in variation for the synthetic windows. However this was not mode collapse in the GAN since the cause this was found to be the autoencoder since now critical features like spread, mid, and gaps were having to compete with the flags within the MSE loss.

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

To overcome fat tails not matching the real data a 1D Wasserstien loss was added to match distributions for spread, and mid-price. Mid-price was used over VWAP here due to speed of computation and that VWAP would work if mid-price works because model all fits together and that.

However, this wasn't a one-shot fix as initial experiments with this new loss produced unrealistic behaviors where the bid and ask would both jump simulatiously in opposite directions (a result of over-rewarding this distrubtion loss on spread). The soluition was to reduce the weighing for the Wasserstien loss from 10 to 1, which retaind the benifits of distrubiton matching whilst fixing the symmetric jumps.

#### Failed Experiments

Text...

## Evaluation

talk somewhere about how we didn't mode collapse

#### Training and Stability

Mention how I used adam optimser

show loss curves and talk about discriminator going back and forth and stuff. Also talk about stability here (changing seed does not hurt it, etc)

#### Synthetic Windows and Generation

Do things like just look at nice synthetic windows, compare large and small tick stocks. etc.

#### Metrics and Baseline Comparison

Due to the baseline models naive design choices leading to consistent LOB violations it's instantly clear that the developed TimeGAN setup is a superiour option. However the GRU has the advantage of being autoregressive which means it could do better at some metrics (EXPLAIN FURHTER).

Comparing the two models against our acceptance criteria from the fieablity review we see that...

#### Structural Fails

Despite being a clear improvement over the baseline model and boasting a 0% LOB violation rate our TimeGAN model still has several structural pitfalls. 

- **Price Volatility at Depth:** Synthetic changes its px at depth a lot more than real
- **Volume Persistence and Clarity:** Synthetic fails to properly generalize the relation between and actual price of a level and the volume there over time.
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

## Feasibility Review

### User Need, Scope & Acceptance Criteria

Consider a quantitative research team stress-testing a market-making algorithm that operates on Level-2 limit order books (LOB) at depth 10. Historical LOB data is expensive and limited, so the team wants to generate synthetic LOB time series to test their algorithm on more scenarios than the real data provides. The initial scope is to generate 64-second windows of AMZN order books at 1-second resolution, trained on a single day of LOBSTER sample data. The synthetic data must never violate microstructure constraints (crossed books, misordered price levels or non-positive volume), and must match the statistical behaviour of real market data.

TODO: Change SSID match to mean 60% of the real-vs-real reference.

- **LOB violations:** crossed books, non-monotonic ladders and non-positive volume, ≤ 0.1% each.
- **Distribution fidelity:** KL diverence ≤0.1 for spread and VWAP log-returns (VWAP also captures volume).
- **Temporal features:** |return| autocorrelation at lags 1–5 within 0.05 of real and unchanged features per step are within 5% of real.
- **Depth structure:** depth-heatmap SSIM > 0.6, alongside a real-vs-real reference.
- **Resources:** ≤ 30 min training and ≤2 GB peak VRAM on one RTX 3070.

> Note that because our user is a market maker realistic price action is more important then resting liquidity.

### Model Choice & Course Concepts

Implemention will use a TimeGAN model (Yoon et al., 2019), rated at hard difficulty, with a 70/15/15 chronological train/validate/test split over a single day of data to avoid look-ahead bias. We resample our data to 1s using the last snapshot at each point, as LOBSTER describes each snapshot as the outcome of a single market event, whereas TimeGAN expects steps spaced equally over time. TimeGAN build on course content by combining an autoencoder (embedder/recovery), a supervisor trained on next-step prediction, and an adversarial generator/discriminator, which are all GRU-based. The supervised loss targets temporal dynamics, which a plain GAN learns only indirectly, so it directly tests the question: does a joint supervised-adversarial objective learn order book structure better than a standard generator?

### Preliminary Feasibility Evidence

The data covers 23,400 seconds of AMZN trading at 1 s resolution. The book is highly persistent: best prices are unchanged in about 19.6% of seconds, and order sizes are heavy-tailed (median 100 shares, maximum about 34,000).

As a smoke test, I trained a plain autoregressive GRU on the 40 raw price and size columns. Nothing enforces book structure, so it must be learned. It will serve as the baseline for the TimeGAN comparison. The figure compares a real window (left) with a generated one (right).

![baseline_real_vs_synth](figures/final/baseline_real_vs_synth.png)

The pipeline runs end to end, but the baseline breaks book invariants: about 4% of steps have crossed books and 99.96% have non-monotonic price ladders, although all sizes are valid. This baseline model also fails to mimic the price action of real books with the synthetic prices changing in 82.95% of steps compared to the mentioned 19.6% of steps like in the real data.

### Risks, Budget & Fallback

The largest risk is that TimeGAN does not match how the data is generated. LOBSTER records discrete events (submissions, cancellations and executions) at irregular times, with integer prices and sizes, whereas TimeGAN assumes evenly spaced steps and continuous values, so resampling to 1s may lose important structure. A second risk comes from the chronological split of a single day: the model trains on the morning and is tested on the late afternoon, which may behave quite differently. I will compare metrics across the train, validation and test splits and report any gap. If either risk proves limiting then the fallback is an event-based model that generates order events and applies them to the book through a simple matching engine, which will more directly model the underlying process rather then estimating it. 

All development and final runs fit on a local RTX 3070 (total of 24 GB available) within the resource targets with the next step in the development process being to replace our baseline autoregressive GRU with the actual TimeGAN architecture.

## Artificial Intelligence Usage Disclosure

Helping prepose causes to structual failures and potential remidies.

Helping edit and provide feedback on readme and documentation.

Help with programming.
.....