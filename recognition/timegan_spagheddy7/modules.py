"""PyTorch model components for the AMZN order-book project."""

import torch


@torch.jit.script
def reference_gru_sequence(values, gate_kernel, gate_bias, candidate_kernel, candidate_bias):
    """TF1 GRU equations: apply the reset gate before the candidate matrix."""
    width = values.size(2)
    hidden = values.new_zeros((values.size(0), candidate_bias.numel()))
    gates_x = values @ gate_kernel[:width] + gate_bias
    candidate_x = values @ candidate_kernel[:width] + candidate_bias
    outputs = []
    for step in range(values.size(1)):
        reset, update = torch.sigmoid(gates_x[:, step] + hidden @ gate_kernel[width:]).chunk(2, dim=1)
        candidate = torch.tanh(candidate_x[:, step] + (reset * hidden) @ candidate_kernel[width:])
        hidden = update * hidden + (1 - update) * candidate
        outputs.append(hidden)
    return torch.stack(outputs, dim=1)


class ReferenceGRU(torch.nn.Module):
    """Stacked reference cells, with input projections shared across all times.

    Equation/initialisation reference: TensorFlow 1.15 GRUCell in
    tensorflow/python/ops/rnn_cell_impl.py. This is an independent torch port.
    """

    def __init__(self, inputs, hidden, layers):
        super().__init__()
        self.cells = torch.nn.ModuleList()
        for width in [inputs] + [hidden] * (layers - 1):
            cell = torch.nn.ParameterDict({"gate_kernel": torch.nn.Parameter(torch.empty(width + hidden, 2 * hidden)),
                                           "gate_bias": torch.nn.Parameter(torch.ones(2 * hidden)),
                                           "candidate_kernel": torch.nn.Parameter(torch.empty(width + hidden, hidden)),
                                           "candidate_bias": torch.nn.Parameter(torch.zeros(hidden))})
            torch.nn.init.xavier_uniform_(cell["gate_kernel"])
            torch.nn.init.xavier_uniform_(cell["candidate_kernel"])
            self.cells.append(cell)

    def forward(self, values):
        for cell in self.cells:
            values = reference_gru_sequence(values, cell["gate_kernel"], cell["gate_bias"], cell["candidate_kernel"], cell["candidate_bias"])
        return values, None


class SequenceNetwork(torch.nn.Module):
    """Causal GRU with a per-update output projection."""

    def __init__(self, inputs, hidden, outputs, layers, bounded=True, cell="pytorch", context_dim=0):
        super().__init__()
        self.context_dim = context_dim
        self.gru = ReferenceGRU(inputs + context_dim, hidden, layers) if cell == "reference" else torch.nn.GRU(inputs + context_dim, hidden, layers, batch_first=True)
        self.projection = torch.nn.Linear(hidden, outputs)
        if cell == "reference":
            torch.nn.init.xavier_uniform_(self.projection.weight)
            torch.nn.init.zeros_(self.projection.bias)
        self.bounded = bounded

    def forward(self, values, context=None):
        if self.context_dim:
            if context is None or context.shape != (len(values), self.context_dim):
                raise ValueError("Conditional networks require one context vector per sequence.")
            values = torch.cat((values, context[:, None].expand(-1, values.shape[1], -1)), dim=-1)
        elif context is not None:
            raise ValueError("This reference network does not accept context.")
        hidden, _ = self.gru(values)
        projected = self.projection(hidden)
        return torch.sigmoid(projected) if self.bounded else projected


class PriceLadderDecoder(torch.nn.Module):
    """Learn an ordered log-price ladder, then return scaled log-return targets.

    Architecture extension: one common bid return and 19 positive relative gaps.
    Gap changes are bounded by training-only ranges, not sorted or repaired.
    Buffers add no trainable parameters. Volumes retain the sigmoid output.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("minimum", torch.zeros(40, dtype=torch.float64))
        self.register_buffer("span", torch.ones(40, dtype=torch.float64))
        self.register_buffer("gap_scale", torch.ones(19, dtype=torch.float64))

    @torch.no_grad()
    def configure(self, minimum, span, gap_scale):
        for name, value in (("minimum", minimum), ("span", span), ("gap_scale", gap_scale)):
            value = torch.as_tensor(value, dtype=torch.float64, device=self.minimum.device)
            target = getattr(self, name)
            if value.shape != target.shape or not torch.isfinite(value).all() or (name != "minimum" and (value <= 0).any()):
                raise ValueError("Invalid training-only ladder decoder statistics.")
            target.copy_(value)

    def forward(self, logits, anchors):
        if anchors is None or anchors.shape != (len(logits), 20):
            raise ValueError("Ladder recovery requires one 20-price anchor per sequence.")
        anchors = anchors.to(device=logits.device, dtype=torch.float64)
        if not torch.isfinite(anchors).all() or (anchors <= 0).any():
            raise ValueError("Ladder anchors must be finite and positive.")
        initial = anchors.log()
        asks, bids = initial[:, ::2], initial[:, 1::2]
        gaps = torch.cat((asks[:, :1] - bids[:, :1], asks[:, 1:] - asks[:, :-1], bids[:, :-1] - bids[:, 1:]), dim=-1)
        if (gaps <= 0).any():
            raise ValueError("Ladder anchors must be strictly ordered with positive spread.")
        raw = logits[..., ::2].double()
        bid_returns = raw[..., 1].sigmoid() * self.span[2] + self.minimum[2]
        best_bid = initial[:, None, 1] + bid_returns.cumsum(dim=1)
        gap_logits = torch.cat((raw[..., :1], raw[..., 2:]), dim=-1)
        future_gaps = gaps[:, None] * (gap_logits.tanh() * self.gap_scale).exp()
        best_ask = best_bid + future_gaps[..., 0]
        ask_path = torch.cat((best_ask[..., None], best_ask[..., None] + future_gaps[..., 1:10].cumsum(dim=-1)), dim=-1)
        bid_path = torch.cat((best_bid[..., None], best_bid[..., None] - future_gaps[..., 10:].cumsum(dim=-1)), dim=-1)
        log_prices = torch.stack((ask_path, bid_path), dim=-1).flatten(-2)
        previous = torch.cat((initial[:, None], log_prices[:, :-1]), dim=1)
        returns = (log_prices - previous - self.minimum[::2]) / self.span[::2]
        return torch.stack((returns.to(logits.dtype), logits[..., 1::2].sigmoid()), dim=-1).flatten(-2)


class TimeGAN(torch.nn.Module):
    """Five-network TimeGAN for fixed-length feature sequences.

    Architecture reference: Yoon, Jarrett and van der Schaar (NeurIPS 2019),
    https://github.com/jsyoon0823/TimeGAN . Independent PyTorch implementation.
    Sigmoid feature outputs are bounded by training ranges, not book constraints.
    By default starting prices enter only the external decoder. Optional static
    context enters every network; this is a labelled conditional extension.
    """

    def __init__(self, features=40, hidden=32, layers=2, noise_dim=40, cell="pytorch", context_dim=0, price_decoder="none"):
        super().__init__()
        if min(features, hidden, noise_dim) < 1 or layers < 2 or not isinstance(context_dim, int) or context_dim < 0:
            raise ValueError("Use positive widths and at least two main GRU layers.")
        if cell not in ("reference", "pytorch"):
            raise ValueError("cell must be reference or pytorch (legacy pilot checkpoints).")
        self.config = dict(features=features, hidden=hidden, layers=layers, noise_dim=noise_dim, cell=cell)
        if context_dim:
            self.config["context_dim"] = context_dim
        if price_decoder not in ("none", "ladder") or (price_decoder == "ladder" and features != 40):
            raise ValueError("Use none or ladder; the ladder decoder requires 40 features.")
        if price_decoder != "none":
            self.config["price_decoder"] = price_decoder
        self.embedder = SequenceNetwork(features, hidden, hidden, layers, cell=cell, context_dim=context_dim)
        self.recovery = SequenceNetwork(hidden, hidden, features, layers, bounded=price_decoder == "none", cell=cell, context_dim=context_dim)
        self.generator = SequenceNetwork(noise_dim, hidden, hidden, layers, cell=cell, context_dim=context_dim)
        self.supervisor = SequenceNetwork(hidden, hidden, hidden, layers - 1, cell=cell, context_dim=context_dim)
        self.discriminator = SequenceNetwork(hidden, hidden, 1, layers, bounded=False, cell=cell, context_dim=context_dim)
        self.price_ladder = PriceLadderDecoder() if price_decoder == "ladder" else None

    def recover(self, latent, context=None, anchors=None):
        recovered = self.recovery(latent, context)
        return recovered if self.price_ladder is None else self.price_ladder(recovered, anchors)

    def forward(self, noise, context=None, anchors=None):
        return self.recover(self.supervisor(self.generator(noise, context), context), context, anchors)

    def reconstruct(self, features, context=None, anchors=None):
        return self.recover(self.embedder(features, context), context, anchors)

    # Cached TorchScript autograd graphs require normal tensors even when gradients are disabled.
    @torch.inference_mode(False)
    @torch.no_grad()
    def generate(self, count, sequence_length=32, seed=42, batch_size=64, context=None, anchors=None):
        """Seeded generation; conditional models require preprocessed anchor context."""
        if min(count, sequence_length, batch_size) < 1 or seed < 0 or seed >= 2**63:
            raise ValueError("Invalid generation size or seed.")
        device = next(self.parameters()).device
        rng = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.rand(count, sequence_length, self.config["noise_dim"], generator=rng)
        if context is not None:
            context = torch.as_tensor(context, dtype=noise.dtype, device="cpu").clone()
            if context.shape != (count, self.config.get("context_dim", 0)) or not torch.isfinite(context).all():
                raise ValueError("Invalid generation context.")
        elif self.config.get("context_dim", 0):
            raise ValueError("Conditional generation needs anchor context.")
        if anchors is not None:
            anchors = torch.as_tensor(anchors, dtype=torch.float64, device="cpu")
            if anchors.shape != (count, 20):
                raise ValueError("Invalid generation anchors.")
        was_training = self.training
        self.eval()
        try:
            return torch.cat([self(noise[i:i + batch_size].to(device), None if context is None else context[i:i + batch_size].to(device),
                                   None if anchors is None else anchors[i:i + batch_size].to(device)).cpu()
                              for i in range(0, count, batch_size)])
        finally:
            self.train(was_training)


def temporal_loss(latent, predicted):
    """Prediction at t is compared with the real latent state at t+1."""
    if latent.shape != predicted.shape or latent.ndim != 3 or latent.shape[1] < 2:
        raise ValueError("Temporal supervision needs matching (batch, updates>=2, latent) tensors.")
    return torch.nn.functional.mse_loss(predicted[:, :-1], latent[:, 1:])


def moment_loss(real, generated):
    """Match batch means/standard deviations at each update and feature."""
    if real.shape != generated.shape:
        raise ValueError("Moment matching needs equal real/generated shapes.")
    means = (real.mean(dim=0) - generated.mean(dim=0)).abs().mean()
    standard_deviations = ((real.var(dim=0, unbiased=False) + 1e-6).sqrt()
                          - (generated.var(dim=0, unbiased=False) + 1e-6).sqrt()).abs().mean()
    return means + standard_deviations


def order_penalty(features, anchors, minimum, span, margin=.01):
    """Mean hinge penalty on all 19 price gaps, in units of the chosen USD margin.

    Engineering extension, not an original TimeGAN loss. Training anchors only;
    all returns remain unconstrained and inference applies no repair. Float64
    decoding resolves small gaps between prices near hundreds of dollars.
    """
    if features.ndim != 3 or features.shape[-1] != 40 or anchors is None or anchors.shape != (len(features), 20) or margin <= 0:
        raise ValueError("Expected (batch, updates, 40), (batch, 20) and a positive margin.")
    returns = features[..., ::2].double() * span[::2].double() + minimum[::2].double()
    prices = anchors.double()[:, None] * returns.cumsum(dim=1).exp()
    asks, bids = prices[..., ::2], prices[..., 1::2]
    gaps = torch.cat((asks[..., 1:] - asks[..., :-1], bids[..., :-1] - bids[..., 1:], (asks[..., :1] - bids[..., :1])), dim=-1)
    return torch.relu(1 - gaps / margin).mean()


def lagged_pairs(values, lags=1):
    """Return newest-first lag blocks and next-row targets within one segment."""
    if not isinstance(lags, int) or lags < 1:
        raise ValueError("lags must be a positive integer.")
    if values.ndim != 2 or len(values) <= lags or values.shape[1] == 0:
        raise ValueError("Need a feature matrix with more rows than lags.")
    previous = torch.cat([values[lags - lag - 1:len(values) - lag - 1] for lag in range(lags)], dim=1)
    return previous, values[lags:]


class VectorAR(torch.nn.Module):
    """VAR(p): next vector depends on p past vectors plus a sampled residual.

    Residual vectors are sampled jointly across features and independently
    across time. This preserves empirical simultaneous noise dependencies,
    but does not enforce order-book constraints or conditional volatility.
    """

    def __init__(self, weight, bias, residuals, context_weight=None):
        super().__init__()
        if weight.ndim != 2 or weight.shape[0] < 1 or weight.shape[1] < 1:
            raise ValueError("weight must be a non-empty matrix.")
        if weight.shape[1] % weight.shape[0] != 0:
            raise ValueError("weight must contain a whole number of feature-sized lag blocks.")
        size = weight.shape[0]
        if bias.shape != (size,) or residuals.ndim != 2 or residuals.shape[1] != size or len(residuals) == 0:
            raise ValueError("Incompatible bias or residual dimensions.")
        for name, value in (("weight", weight), ("bias", bias), ("residuals", residuals)):
            value = value.detach().to(device="cpu", dtype=torch.float64).clone()
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} contains non-finite values.")
            self.register_buffer(name, value)
        if context_weight is not None:
            if context_weight.ndim != 2 or context_weight.shape[0] != size or context_weight.shape[1] < 1 or not torch.isfinite(context_weight).all():
                raise ValueError("Invalid exogenous context coefficients.")
            context_weight = context_weight.detach().to(device="cpu", dtype=torch.float64).clone()
        self.register_buffer("context_weight", context_weight)

    @classmethod
    @torch.no_grad()
    def fit(cls, training_features, ridge=1e-4, lags=1):
        """Fit one penalty; use VARFit directly to share work across a ridge grid."""
        return VARFit(training_features, lags).fit(ridge)

    @property
    def lags(self):
        return self.weight.shape[1] // len(self.bias)

    @property
    def spectral_radius(self):
        """Companion-matrix radius; a stability diagnostic, not book validity."""
        size = self.weight.shape[1]
        companion = self.weight.new_zeros((size, size))
        companion[:len(self.bias)] = self.weight
        if self.lags > 1:
            companion[len(self.bias):, :-len(self.bias)] = torch.eye(size - len(self.bias), dtype=self.weight.dtype,
                                                                   device=self.weight.device)
        return float(torch.linalg.eigvals(companion).abs().max())

    def forward(self, previous, context=None):
        """Mean from newest-first flat lags, or chronological (batch, p, D) history."""
        if previous.ndim == 3:
            if previous.shape[1:] != (self.lags, len(self.bias)):
                raise ValueError("History shape must match the model's lag order and features.")
            previous = previous.flip(1).flatten(1)
        if previous.shape[-1] != self.weight.shape[1]:
            raise ValueError("Input width must match the model's lag blocks.")
        mean = previous @ self.weight.T + self.bias
        if self.context_weight is not None:
            if context is None or context.shape != (len(previous), self.context_weight.shape[1]):
                raise ValueError("VARX requires matching static context.")
            mean = mean + context @ self.context_weight.T
        elif context is not None:
            raise ValueError("This VAR does not accept context.")
        return mean

    @torch.no_grad()
    def generate(self, initial_features, sequence_length=32, seed=42, context=None):
        """Generate future updates; the supplied initial state is not returned."""
        if sequence_length < 1 or seed < 0:
            raise ValueError("sequence_length must be positive and seed non-negative.")
        history = initial_features.detach().to(device=self.weight.device, dtype=self.weight.dtype)
        if history.ndim == 2 and self.lags == 1:
            history = history[:, None, :]
        if history.ndim != 3 or history.shape[1:] != (self.lags, len(self.bias)) or len(history) == 0:
            raise ValueError("Expected (batch, lags, features) history, oldest to newest.")
        if not torch.isfinite(history).all():
            raise ValueError("Initial features contain non-finite values.")
        if context is not None:
            context = torch.as_tensor(context, dtype=history.dtype, device=history.device)
            if not torch.isfinite(context).all():
                raise ValueError("Context contains non-finite values.")
        rng = torch.Generator(device=self.weight.device).manual_seed(seed)
        generated = history.new_empty((len(history), sequence_length, len(self.bias)))
        for step in range(sequence_length):
            indices = torch.randint(len(self.residuals), (len(history),), generator=rng, device=history.device)
            state = self(history, context) + self.residuals[indices]
            generated[:, step] = state
            history = torch.cat((history[:, 1:], state[:, None, :]), dim=1)
        if not torch.isfinite(generated).all():
            raise FloatingPointError("Autoregressive rollout became non-finite; no clipping was applied.")
        return generated


class VARFit:
    """One centred SVD per lag order, reused for every ridge penalty.

    For X = U diag(s) V.T, W.T = V diag(s / (s**2 + n*ridge)) U.T Y.
    This is the same least-squares objective as the augmented-system solver;
    the intercept is unpenalised and no covariance matrix is inverted.
    """

    @torch.no_grad()
    def __init__(self, training_features, lags=1):
        values = training_features.detach().to(device="cpu", dtype=torch.float64)
        if not torch.isfinite(values).all():
            raise ValueError("Training features contain non-finite values.")
        previous, following = lagged_pairs(values, lags)
        self.context_dim = 0
        self.factor(previous, following)

    @classmethod
    @torch.no_grad()
    def from_pairs(cls, previous, following, context=None):
        """Fit supplied within-training pairs, optionally with static exogenous inputs."""
        previous, following = previous.double().cpu(), following.double().cpu()
        if previous.ndim != 2 or following.ndim != 2 or len(previous) != len(following) or not len(previous):
            raise ValueError("Expected aligned nonempty predictor and target matrices.")
        if previous.shape[1] % following.shape[1]:
            raise ValueError("Predictors must contain complete lag blocks.")
        result = cls.__new__(cls)
        if context is not None:
            if context.ndim != 2 or len(context) != len(previous) or not context.shape[1]:
                raise ValueError("Expected aligned context rows.")
            previous = torch.cat((previous, context.double().cpu()), dim=1)
        result.context_dim = 0 if context is None else context.shape[1]
        if not torch.isfinite(previous).all() or not torch.isfinite(following).all():
            raise ValueError("Training pairs must be finite.")
        result.factor(previous, following)
        return result

    def factor(self, previous, following):
        self.x_mean, self.y_mean = previous.mean(0), following.mean(0)
        self.x, self.y = previous - self.x_mean, following - self.y_mean
        u, self.s, self.vh = torch.linalg.svd(self.x, full_matrices=False)
        self.uty = u.T @ self.y  # Release tall U after construction; retain only small decomposition factors.

    @torch.no_grad()
    def fit(self, ridge=1e-4):
        if ridge < 0 or not torch.isfinite(torch.tensor(ridge)).item():
            raise ValueError("ridge must be finite and non-negative.")
        if ridge == 0:
            tolerance = torch.finfo(self.s.dtype).eps * max(self.x.shape) * self.s.max()
            factors = torch.where(self.s > tolerance, self.s.reciprocal(), torch.zeros_like(self.s))
        else:
            factors = self.s / (self.s.square() + len(self.x) * ridge)
        weight = (self.vh.T @ (factors[:, None] * self.uty)).T.contiguous()
        bias = self.y_mean - self.x_mean @ weight.T
        residuals = self.y - self.x @ weight.T
        residuals -= residuals.mean(0)
        return VectorAR(weight[:, :-self.context_dim], bias, residuals, weight[:, -self.context_dim:]) if self.context_dim else VectorAR(weight, bias, residuals)
