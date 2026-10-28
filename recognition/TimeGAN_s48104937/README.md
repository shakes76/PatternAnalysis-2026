# TimeGAN LOB Simulation

## Project Overview
The original TimeGAN architecure was infact extended however, with conditioning per ticker, and a minimal 1D Wasserstein distance loss to match the shape of real feature distrbution.

![timegan_architecture](https://hackmd.io/_uploads/r1qU_p03fx.png)


## Dataset and Preprocessing

#### LOBSTER Dataset
We only have access to the LOBSTER sample data, which covers a single trading day (21 June 2012) of tick-by-tick Level 10 order book snapshots and the accompanying messages for five tickers. An initial audit, however, shows that we have to trim our usable data even further if we want chronological splits to evaluate against, due to the extreme regime changes in both price and volume behaviour at the open and close. Thus the first and final 15 minutes of each day are trimmed.

![data_split](https://hackmd.io/_uploads/BJZN7kJTMg.png)

Even with trimming there will be differences between the training and evaluation periods with spread declining through the day, unfortunately this is an unavoidable limitation of chronological splits on a single day.

Another change which reduces the avalible training data is resampling. Since each LOBSTER snapshot is the outcome of a single event each step has only 1 true change and steps are not evenly spread over time. Resampling to 1s partially addresses this, however, we reduce our total usable data from 1,823,146 rows to 108,000.

The final key observation with including the five ticks is the distinct behaviour of small-tick (AAPL, AMZN, GOOG) and large-tick (INTC, MSFT) stocks. With large-tick stocks having their spread pinned at one tick and a mid price that rarely moves, while small-tick stocks change price far more often.
![regime_behaviour](https://hackmd.io/_uploads/SJE2LkkaMe.png)


#### Feature Engineering
Through the baseline test its clear LOB violations are common under naive modelling, thus the major change made when moving to TimeGAN will be to explicitly design the features to make the order book valid by construction.

![feature_engineering](https://hackmd.io/_uploads/B1obdkypMe.png)


explain we use mid price relative to start of window, (briefly explain spread and gap system (imgages does most)), explain what we did with volume (not much). explain that we needed to add flats due to jitter.


Rather than modelling the price of each level, we model the mid-price relative to the start of the window, the spread, and the gap from each quote to the one preceding it. The drawback of this representation is having to pass reference price to anchor synthetic windows on. Another major change made halfway through is a flag for each feature allowing it to update, reasoning and impacts of which are discussed in Development.


## Development

#### Ticker Conditioning
talk about batch sampler and how/why i chose to train the way i did.

#### Feature Flags
This was included as initial tests without this flag resulted in unrealistic jitter with synethic features changing in XXX% of steps compared to XX% of steps in the real data.

#### Embedder Issues
issue with being at the mean (balanced loss -> grouped loss), and then issue with training overfit (layers 2 -> 1), and then issue with not being able to learn enough (hidden_dim 64 -> 128).




#### Grouped MSE Loss
Was having issues where volume and spread was reverting to the mean, having done tests it was discovered this was not an issue with the generator or feature engineering, but an issue with the embedder. Training more was not an appropiate solution as ????? (actually look into this). The solution was acutally to segement the loss for the embedder into groups (price, size, spread), this makes sense as before spread was a single feature amoung 85, but now 
...

#### 1D Wasserstein Loss
To overcome fat tails not matching the real data a 1D Wasserstien loss was added to match distibuitions for spread, and mid-price. Mid-price was used over VWAP here due to speed of computation and that VWAP would work if mid-price works because model all fits together and that.

However this wasn't a one-shot fix as inital experiments with this new loss produced unrealsitc behavious where the bid and ask would both jump simulatiously in opposite directions (a result of over-rewarding this distrubtion loss on spread). The soluition was to reduce the weighing for the Wasserstien loss from 10 to 1, which retaind the benifits of distrubiton matching whilst fixing the symmetric jumps.


## Evaulation

#### TimeGAN Training
show loss curves and talk about discrimator going back and forth and stuff. Also talk about stablity here (changing seed does not hurt it, etc)

#### Baseline Comparison
talk about things that funamandtally broke


#### Structural Fails
talk about things that funamandtally broke
##### Price Volatility at Depth
Synethic changes its px at depth a lot more then real
##### 

## Recommendations

#### Viability in Industry
...

#### Alternative Architecure
Had the constraint of implementing a TimeGAN model not been in place, there would be the potential to develop a far better model. The alternative is a maximum-likelihood event model conditioned on the current L10 book and history, outputting actual market events (order submissions, cancellations and executions) to a matching engine maintaining the LOB. A GAN-like model could also be used in this system to generate new level 10 quotes when the matching engine runs out of quotes on one side, such as when a level is cancelled or executed.

This approach would be far superior, as all market behaviour that our TimeGAN is attempting to emulate is emergent from this underlying event process. Furthermore, it would let the quantitative researcher see how the market reacts to their own trades in backtests, since any order the researcher's trading algorithm places can be fed to the event model like any other event.

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

![baseline_real_vs_synth](https://hackmd.io/_uploads/SJ027GAnMx.png)

The pipeline runs end to end, but the baseline breaks book invariants: about 4% of steps have crossed books and 99.96% have non-monotonic price ladders, although all sizes are valid. This baseline model also fails to mimic the price action of real books with the synthetic prices changing in 82.95% of steps compared to the mentioned 19.6% of steps like in the real data.

### Risks, Budget & Fallback  
The largest risk is that TimeGAN does not match how the data is generated. LOBSTER records discrete events (submissions, cancellations and executions) at irregular times, with integer prices and sizes, whereas TimeGAN assumes evenly spaced steps and continuous values, so resampling to 1s may lose important structure. A second risk comes from the chronological split of a single day: the model trains on the morning and is tested on the late afternoon, which may behave quite differently. I will compare metrics across the train, validation and test splits and report any gap. If either risk proves limiting then the fallback is an event-based model that generates order events and applies them to the book through a simple matching engine, which will more directly model the underlying process rather then estimating it. 

All development and final runs fit on a local RTX 3070 (total of 24 GB available) within the resource targets with the next step in the development process being to replace our baseline autoregressive GRU with the actual TimeGAN architecture.


## Artificial Intelligence Usage Disclosure

Helping prepose causes to structual failures and potential remidies.

Helping edit and provide feedback on readme and documentation.

Help with programming.
.....