
# TimeGAN LOB Simulation

## Project Overview

## Dataset, Setup and Usage
Talk about regimes, featrues ensuring vlalidity by construction


## Results

Instead of directly attempting to model the price and volume, we can intential design our features to ensure our LOB is valid at every step by construction. To do this we encode each snapshot as the change in mid from the previous timestep, the current spread, and then the gap between each of the quotes. The constraint of this is that now a refernece price needs to be passed to anchor each window synehtic window.

The result of this instnatnly showed promise with the  



## Recommendations

## Artificial Intellgence Discolsure
.....

## Data and Feature Engineering
Talk about regimes, featrues ensuring vlalidity by construction

## Model Architecure
give overview of timegan architecure

## Model Development
Talk about things that went wrong, what i fixed, repeat.

## Model Evaluation
All the generic eval stuff

## 

## Feasibility Review


### User Need, Scope & Acceptance Criteria

Consider a quantitative research team stress-testing a market-making algorithm that operates on Level-2 limit order books (LOB) at depth 10. Historical LOB data is expensive and limited, so the team wants to generate synthetic LOB time series to test their algorithm on more scenarios than the real data provides. The initial scope is to generate 64-second windows of AMZN order books at 1-second resolution, trained on a single day of LOBSTER sample data. The synthetic data must never violate microstructure constraints (crossed books, misordered price levels or non-positive volume), and must match the statistical behaviour of real market data.
  
- **LOB violations:** crossed books, non-monotonic ladders and non-positive volume, ≤ 0.1% each.
- **Distribution fidelity:** KL ≤ 0.1 for spread and VWAP log-returns (VWAP also captures volume).
- **Temporal features:** |return| autocorrelation at lags 1–5 within 0.05 of real; unchanged best prices within 5 pp of real.
- **Depth structure:** depth-heatmap SSIM > 0.6, alongside a real-vs-real reference.
- **Resources:** ≤ 30 min training and ≤2 GB peak VRAM on one RTX 3070.
  
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

## 