# TimeGAN Experiment Log

## V4.0–V4.5: Model Improvement Experiments

### V4.0 — Mid-price Return Representation
Added mid-price log returns as the 41st feature.

- Return KL: 3.180144
- Generated return standard deviation: 9.38e-6
- Finding: The model can generate returns, but output diversity remains limited.

### V4.1 — Asinh and Z-score Normalisation
Replaced Min-Max scaling of the return feature with asinh transformation and Z-score normalisation.

- Return KL: 2.133785
- Generated return standard deviation: 1.28e-7
- Finding: Histogram KL improved, but return diversity deteriorated.

### V4.2 — Identity Recovery Activation
Changed the Recovery output activation from Sigmoid to Identity.

- Return KL: 2.133785
- Generated return standard deviation: 1.24e-7
- Finding: Training-level noise sensitivity improved, but generated returns still exhibited severe collapse.

### V4.3 — Weighted Return Reconstruction
Added a reconstruction objective that gives higher weight to nonzero mid-price returns.

- Validation reconstruction correlation: 0.9750
- Generated return KL: 3.762186
- Generated return standard deviation: 3.61e-6
- Finding: Return reconstruction improved substantially, but generated returns did not reproduce the real market's zero-return frequency and volatility.

### V4.4 — Independent Empirical Tick Baseline
Sampled discrete mid-price movements from the training-set empirical distribution.

- Return KL: 0.006692
- Lag-1 absolute-return correlation: -0.0026
- Finding: Marginal distribution matching improved, but the independent baseline failed to reproduce temporal dependence.

### V4.5 — Markov Tick Baseline
Modelled the probability of a price movement conditional on whether the previous observation contained a movement.

- Return KL: 0.004997
- Lag-1 absolute-return correlation: 0.0432
- Real test lag-1 absolute-return correlation: 0.2350
- Finding: First-order movement dependence improved temporal structure, but did not fully capture volatility clustering.

## Key Research Findings

1. Approximately 89% of training-set observations have zero mid-price returns.
2. All observed nonzero changes in the raw doubled mid-price were integer multiples of 100.
3. Weighted reconstruction losses significantly improved the model's ability to reconstruct sparse financial returns.
4. Matching marginal return distributions does not guarantee realistic time dependence.
5. Statistical tick baselines are not substitutes for a complete TimeGAN-based limit order book generator.

## Next Research Direction — V5.0

Investigate a tick-aware TimeGAN with explicit price-movement modelling, while maintaining consistency between generated mid-price paths and the ten-level order book.

## 2026-10-19 — V5.1–V5.3 Constraint-aware LOB Experiments

### Objective
Investigate whether a constraint-aware LOB decoder can improve market-structure validity while preserving reconstruction quality and generative diversity.

### V5.1: Structural parameterization
- Verified positive bid-ask spreads and ordered ask/bid levels.
- Represented 20 relative price features using one spread and 18 level gaps.
- Maximum price reconstruction error: 2.84e-14.
- Structural parameterization test passed.

### V5.2: Constraint-aware LOB Head
- Implemented a differentiable PyTorch LOB head.
- Verified Min-Max normalization and gradient propagation.
- Frozen-embedder test normalized LOB MSE: 0.184861.
- Original V4.3 Recovery test normalized LOB MSE: 0.011923.
- Found excessively large initial spread and gap values.

### V5.3: Mean-informed initialization
- Initialized output biases using training-set statistics only.
- Training-set mean spread: 0.139864.
- Mean ask gap: 0.019693.
- Mean bid gap: 0.019951.
- Test normalized LOB MSE: 0.010218.
- Original V4.3 Recovery test normalized LOB MSE: 0.011923.
- Test spread MAE: 0.051112 versus 0.051315 for V4.3.
- All tested market-structure constraints passed.
- Initial validation MSE was already lower than the best trained-epoch MSE, suggesting initialization was the primary improvement.

### Generative evaluation
- Reused the frozen V4.3 Generator and Supervisor with the V5.3 LOB Head.
- Noise distribution: uniform.
- Generated 512 sequences of length 24.
- Real spread standard deviation: 0.041816.
- Generated spread standard deviation: 0.000158.
- Real pairwise LOB difference: 0.396822.
- Generated pairwise LOB difference: 0.0000128.
- Generated market-structure constraints passed.
- Severe lack of generative diversity remained.

### Conclusion
Mean-informed initialization substantially improved constrained LOB reconstruction. However, strong reconstruction results did not translate into realistic generative diversity. The constraint-aware decoder ensures structural validity but does not by itself resolve TimeGAN mode collapse.

These experiments demonstrate the distinction between reconstruction quality, market validity, and generative realism.

### Project decision
With the submission deadline on 2026-10-30, stop expanding the model architecture. Preserve V3.3 as the stable baseline and document V4.3 and V5.3 as experimental investigations. Prioritize final evaluation, reproducibility, documentation, and submission.


**Final submission clarification (2026-10-29):** The decision above was a provisional project plan recorded during the V5.3 experiments, not the final model-selection decision. Subsequent V5.x experiments continued to investigate mode collapse, market validity, return dynamics and liquidity structure. The final submitted model is the V5.38 integrated hybrid generator, as described in `README.md`. V3.3 remains a historical stable baseline and is not the final submitted generator.
## 2026-10-20 — V5.4 Diversity-aware TimeGAN Experiments

### Objective
Investigate mode collapse in the TimeGAN Generator, Supervisor, and constrained LOB decoder.

### Diagnosis
- Generator hidden pairwise difference: 0.027942.
- Supervisor hidden pairwise difference: 0.001535.
- Bypassing Supervisor increased generated LOB pairwise difference by approximately 17.19x.
- However, diversity remained far below real market data.

### V5.4a: Output-space diversity loss
- Diversity weight: 0.5.
- Training: 1 epoch.
- Normalized generated LOB pairwise difference decreased from 3.81e-6 to 2.90e-6.
- Gradient audit found diversity gradient norms much smaller than original generator loss gradients.

### V5.4b: Hidden-space diversity loss
- Diversity weight: 0.5.
- Training: 1 epoch.
- Normalized generated LOB pairwise difference increased to 6.16e-6.
- Supervisor hidden pairwise difference improved by approximately 35.4% against the V4.3 baseline in a fixed-noise comparison.

### V5.4c: Stronger hidden-space diversity
- Diversity weight: 2.0.
- Training: 3 epochs from the V4.3 checkpoint.
- Initial normalized generated LOB pairwise difference: 3.81e-6.
- Epoch 1: 4.27e-5.
- Epoch 2: 4.54e-5.
- Epoch 3: 3.99e-5.
- Final generated spread standard deviation: approximately 1.47e-4.
- Saved checkpoint corresponds to Epoch 3, not the best-diversity Epoch 2.

### Conclusion
Hidden-space diversity regularization improved generated LOB differences, but the absolute variation remained much smaller than the real market. The experiment did not solve mode collapse or establish realistic spread dynamics.

### Next Experiment
Evaluate spread-aware distribution matching as a potential improvement. The V5.5 experiment remains unverified at the time of this entry.

## 2026-10-21 — V5.7 and V5.8 Decoder Ablations

### V5.7: Unconstrained MLP Decoder
- Trained an MLP decoder on frozen V4.3 Embedder representations.
- Selected epoch: 4.
- Test normalized LOB MSE: 0.010063.
- Test pairwise difference: 0.000997.
- Test temporal change: 0.001034.
- Market structure audit: 0% of reconstructed snapshots satisfied all constraints simultaneously.
- Conclusion: Nonlinear decoding improved variation, but did not guarantee valid market structure.

### V5.8: MLP Constraint-aware Decoder
- Combined a two-hidden-layer MLP with constrained spread and price-gap construction.
- Initialised output biases using training-set statistics.
- Selected epoch: 2.
- Test normalized LOB MSE: 0.010110.
- Test pairwise difference: 0.000583.
- Test temporal change: 0.000536.
- All tested structural constraints passed.
- Conclusion: Improved dynamic reconstruction over V5.6 while retaining structural validity, but remained substantially less variable than real LOB data.

### Remaining limitation
Both decoders retained substantially less temporal and cross-sample variability than real order-book sequences. Reconstruction MSE remained close to a training-mean baseline.

### Next experiment
V5.9 will investigate feature-wise variance-aware reconstruction losses while preserving the constrained decoder architecture.

## V5.9 — Variance-aware Constrained LOB Reconstruction

### Experiment
- Compared continued training with and without a feature-wise variance loss.
- Both arms started from the same V5.8 checkpoint.
- Used three epochs and the same training data, learning rate and batch ordering.
- The V4.3 Embedder remained frozen.

### Test results
- Control MSE: 0.010110
- Variance-aware MSE: 0.011186
- Control pairwise difference: 0.000583
- Variance-aware pairwise difference: 0.011541
- Real pairwise difference: 0.050133
- Control temporal change: 0.000536
- Variance-aware temporal change: 0.010178
- Real temporal change: 0.008069
- All tested LOB structure constraints passed.

### Feature-group findings
- Generated reconstruction spread mean: 0.150189
- Generated reconstruction spread std: 0.027284
- Price-offset pairwise difference: 0.013521
- Transformed-size pairwise difference: 0.108488

### Interpretation
Variance-aware reconstruction substantially improved variability
but increased pointwise reconstruction error.
These results concern reconstruction from real embedded data,
not unconditional TimeGAN generation.

### Next steps
Evaluate feature-group realism and test the learned decoder
on the Generator-Supervisor output distribution.

## V5.10 — Latent Distribution Matching

### Objective
Improve generated hidden-state diversity and distribution matching.

### Method
- Continued from the V5.4 diversity-aware training approach.
- Added a channel-wise latent mean and standard deviation matching loss.
- Used latent matching weight 5.0, diversity weight 2.0,
  three epochs, and two generator updates.
- Evaluated the trained generator using the V5.9
  variance-aware, constraint-aware LOB decoder.

### Generation comparison
V5.4c + V5.9:
- LOB pairwise difference: 0.003361
- LOB temporal change: 0.001965
- Spread standard deviation: 0.004080

V5.10 + V5.9:
- LOB pairwise difference: 0.003603
- LOB temporal change: 0.002134
- Spread standard deviation: 0.004442
- All tested market structure constraints passed.

### Limitations
- The generated order books still show substantial diversity collapse.
- Results are based on a single fixed random-seed comparison.
- The current improved generation path produces relative
  LOB features, not a complete absolute-price LOB sequence.

### Next objective
Implement a complete generated mid-price path and use it
to reconstruct absolute 10-level ask/bid prices.
