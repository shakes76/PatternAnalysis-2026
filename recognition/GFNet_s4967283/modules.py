"""
modules.py

Model definitions for the ADNI Alzheimer's classification project
(AD = Alzheimer's Disease, NC = Cognitively Normal).

Two models live here:

  1. BaselineCNN - a plain convolutional network. This is the comparison point.
                   The assignment asks me to implement a simple baseline myself
                   and benchmark the main model against it on identical splits,
                   so this is half of the experiment, not filler.

  2. GFNet      - the main model. A Global Filter Network, which replaces the
                  self-attention of a Vision Transformer with a learnable
                  filter applied in the frequency domain.

Note on imports: the assignment says the model code must not depend on NumPy or
anything outside PyTorch, so everything here is pure torch.

Reference for GFNet:
  Y. Rao, W. Zhao, Z. Zhu, J. Lu, J. Zhou, "Global filter networks for image
  classification", IEEE TPAMI 45(6), 2023.
"""

import torch
import torch.nn as nn


# ===========================================================================
# Baseline CNN
# ===========================================================================

class ConvBlock(nn.Module):
    """
    One convolution step: convolution -> batch norm -> ReLU -> max pool.

    I pulled this out into its own class because the baseline repeats the same
    shape four times with different channel counts. Writing it once means there
    is only one place to change if I want to adjust the block later.

    Why each piece is here:
      - Conv2d with padding=1 on a 3x3 kernel keeps the width and height the
        same, so only the pooling layer changes the spatial size. That makes the
        size arithmetic easy to follow.
      - BatchNorm2d steadies training. MRI slices vary a lot in overall
        brightness between scanners and patients, so normalising the activations
        helps the network stop caring about absolute intensity.
      - MaxPool2d halves the width and height each block, which is what lets the
        later layers see a larger area of the original image.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.convolution = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            bias=False,  # no bias needed, the batch norm right after has its own shift
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.convolution(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pooling(x)
        return x


class BaselineCNN(nn.Module):
    """
    A straightforward CNN for classifying a brain MRI slice as AD or NC.

    Shape of the thing: four ConvBlocks that grow the channel count while
    shrinking the image, then pool down to a single value per channel, then a
    small fully connected head.

    Design decisions worth being able to explain:

      - I deliberately kept this plain. It is meant to be a fair but ordinary
        reference point. If I tuned it heavily it would stop being a baseline
        and the comparison against GFNet would say less.

      - AdaptiveAvgPool2d((1, 1)) instead of flattening a fixed grid. Global
        average pooling collapses each channel to one number no matter what the
        input size was, so the model works whatever resolution I settle on. It
        also cuts the parameter count a lot compared to flattening, which makes
        it less likely to overfit on a dataset this size.

      - The final layer outputs 2 logits rather than 1 value with a sigmoid.
        Two logits plus CrossEntropyLoss gives me a proper probability for each
        class after a softmax, and I need calibrated per-class confidence later
        for the uncertainty investigation. One sigmoid would work too but this
        keeps the confidence handling identical between the baseline and GFNet.

      - Dropout before the final layer, because the AD/NC split has relatively
        few patients and the network will happily memorise them otherwise.
    """

    def __init__(self, in_channels=1, num_classes=2, dropout_rate=0.3):
        super().__init__()

        # Channel counts double each block: 1 -> 32 -> 64 -> 128 -> 256.
        # Doubling as the image shrinks is the usual pattern, it keeps the
        # amount of information per layer roughly steady.
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )

        # Squash each of the 256 channels down to a single number.
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),           # shape becomes (batch, 256)
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        """
        x comes in with shape (batch, channels, height, width).
        Returns raw logits with shape (batch, num_classes).

        Logits, not probabilities - CrossEntropyLoss applies the softmax
        itself, so applying one here would be doing it twice.
        """
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x


# ===========================================================================
# GFNet
# ===========================================================================

class PatchEmbedding(nn.Module):
    """
    Cut the image into square patches and turn each one into a vector.

    The trick is that a convolution whose kernel size equals its stride does
    exactly this in one operation: each output position looks at one
    non-overlapping patch and produces embed_dim numbers for it. No reshaping
    gymnastics needed.

    For a 224x224 image with 16x16 patches that gives a 14x14 grid, so 196
    tokens, each a vector of length embed_dim.
    """

    def __init__(self, image_size=224, patch_size=16, in_channels=1, embed_dim=256):
        super().__init__()

        if image_size % patch_size != 0:
            raise ValueError(
                "image_size (%d) must divide evenly by patch_size (%d)"
                % (image_size, patch_size)
            )

        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size      # 14 for 224/16
        self.num_patches = self.grid_size ** 2         # 196

        self.projection = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        # x: (batch, channels, height, width)
        x = self.projection(x)              # (batch, embed_dim, grid, grid)
        x = x.flatten(2)                    # (batch, embed_dim, num_patches)
        x = x.transpose(1, 2)               # (batch, num_patches, embed_dim)
        return x


class GlobalFilter(nn.Module):
    """
    The heart of GFNet. Mixes information between all tokens using a learnable
    filter in the frequency domain instead of self-attention.

    What it does, step by step:

      1. Reshape the flat token sequence back into its 2D grid.
      2. Take the 2D real FFT across the two spatial dimensions. The result is
         complex: each entry says how much of a particular spatial frequency is
         present.
      3. Multiply elementwise by a learnable complex weight of the same shape.
         These weights ARE the parameters this layer learns - they are frequency
         gains, deciding which spatial frequencies to amplify or suppress.
      4. Inverse FFT back to real space.
      5. Flatten back into a token sequence.

    Why this counts as global mixing. By the convolution theorem, multiplying in
    the frequency domain is the same as convolving across the whole grid in
    normal space. A convolution that spans the entire grid means every token's
    output depends on every token's input. So this achieves what self-attention
    achieves - all-pairs interaction - but it costs O(N log N) for the FFT
    rather than O(N^2) for attention.

    Why it suits Alzheimer's classification specifically. The disease shows up
    as large-scale structural change: enlarged ventricles, thinner cortex,
    shrunken hippocampus. Those are low-frequency features. A layer whose
    parameters are literally frequency weights is a natural fit, and I can plot
    the learned weights afterwards to see which frequencies the model decided
    mattered. That gives me a figure that explains the model rather than just
    reporting its score.

    The shape detail that catches people out. rfft2 takes advantage of the fact
    that a real-valued input has a symmetric spectrum, so it only returns a bit
    over half the frequencies - the width becomes grid_w // 2 + 1, which is 8
    for a 14-wide grid, not 14. The learnable weight must match THAT shape. Get
    it wrong and you either get a shape error or, worse, a silent broadcast that
    trains badly for no visible reason.

    Storage of the complex weight. PyTorch parameters have to be real tensors,
    so the weight is stored as (grid_h, freq_w, dim, 2) - a trailing dimension
    of size 2 holding the real and imaginary parts - and torch.view_as_complex
    reinterprets it as complex at use time without copying.
    """

    def __init__(self, dim, grid_h, grid_w, init_scale=0.02):
        super().__init__()

        self.dim = dim
        self.grid_h = grid_h
        self.grid_w = grid_w

        # Width of the rfft2 output. This is the half-spectrum size.
        self.freq_w = grid_w // 2 + 1

        # Small random initialisation, as in the GFNet paper. Starting near zero
        # means the filter barely passes any signal at first and the block
        # behaves close to its residual path, which makes early training stable.
        self.complex_weight = nn.Parameter(
            torch.randn(grid_h, self.freq_w, dim, 2) * init_scale
        )

    def forward(self, x):
        # x: (batch, num_tokens, dim)
        batch_size, num_tokens, dim = x.shape

        if num_tokens != self.grid_h * self.grid_w:
            raise ValueError(
                "expected %d tokens for a %dx%d grid but got %d"
                % (self.grid_h * self.grid_w, self.grid_h, self.grid_w, num_tokens)
            )

        # Back to the 2D grid: (batch, grid_h, grid_w, dim)
        x = x.view(batch_size, self.grid_h, self.grid_w, dim)

        # FFTs want float32. If this is running under mixed precision the input
        # might be float16, which torch.fft refuses, so cast explicitly.
        x = x.to(torch.float32)

        # Forward transform over the two spatial dimensions only - dim 0 is the
        # batch and dim 3 is the channel, and neither should be mixed.
        # norm="ortho" makes the forward and inverse transforms exact inverses
        # of each other, which is what makes the identity test below work.
        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")

        # Reinterpret the stored real tensor as complex and apply the filter.
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight

        # Back to real space. s= is required so the inverse knows the original
        # width, since the half-spectrum is ambiguous between odd and even sizes.
        x = torch.fft.irfft2(x, s=(self.grid_h, self.grid_w), dim=(1, 2), norm="ortho")

        # Flatten the grid back into a token sequence.
        x = x.reshape(batch_size, num_tokens, dim)
        return x


class FeedForward(nn.Module):
    """
    The standard two-layer MLP that follows the mixing step in a transformer
    block.

    The global filter moves information between tokens. This moves information
    between channels within each token. Both are needed: mixing alone cannot
    build new features, and per-token processing alone cannot share context.

    GELU rather than ReLU, which is the usual choice in transformer blocks - it
    is smooth near zero rather than having a hard corner.
    """

    def __init__(self, dim, hidden_dim, dropout_rate=0.0):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        return self.network(x)


class DropPath(nn.Module):
    """
    Stochastic depth: drop a whole residual branch at random during training.

    Why this exists
    ---------------
    Ordinary dropout zeroes individual activations. Stochastic depth zeroes the
    ENTIRE branch for a sample, so that sample passes through the block as if
    the block were not there. Over a batch, the network trains as an ensemble
    of networks of differing depth.

    This is the standard regulariser for the ViT and GFNet family, and the
    GFNet paper uses it. Without it, extra depth mostly buys extra overfitting,
    which is what my patch-size experiment showed when I added capacity a
    different way.

    The drop probability rises linearly with block index, so early blocks are
    kept almost always and late blocks are dropped most often. Early features
    are shared by everything downstream, so they should be stable.

    At evaluation time this is the identity function.
    """

    def __init__(self, drop_probability=0.0):
        super().__init__()
        self.drop_probability = float(drop_probability)

    def forward(self, x):
        if self.drop_probability == 0.0 or not self.training:
            return x

        keep_probability = 1.0 - self.drop_probability

        # Guard the edge case. At a drop probability of 1.0 the rescale below
        # would divide by zero and return NaN, which would poison the residual
        # instead of removing the branch from it. Returning zeros is the
        # correct meaning: the branch contributes nothing and the identity path
        # carries the signal through unchanged.
        if keep_probability <= 0.0:
            return torch.zeros_like(x)

        # One Bernoulli draw per sample, broadcast over the other dimensions.
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.rand(shape, dtype=x.dtype, device=x.device) < keep_probability

        # Divide by keep_probability so the expected value is unchanged. Without
        # this the activations would shrink during training and the model would
        # behave differently at evaluation time.
        return x * mask / keep_probability

    def extra_repr(self):
        return "drop_probability=%.3f" % self.drop_probability


class GFNetBlock(nn.Module):
    """
    One GFNet block: normalise, filter, normalise, MLP, with a residual.

    A note on the residual structure, because this is a real design decision I
    should be able to defend. A standard Vision Transformer block has TWO
    residual connections, one around the attention and one around the MLP:

        x = x + attention(norm1(x))
        x = x + mlp(norm2(x))

    The GFNet paper instead fuses them into a single residual spanning both the
    filter and the MLP:

        x = x + mlp(norm2(filter(norm1(x))))

    I have followed the paper's fused form so that my implementation matches the
    architecture I am citing. I should check this against the paper directly
    before defending it in the report rather than taking it on trust.
    """

    def __init__(self, dim, grid_h, grid_w, mlp_ratio=4.0, dropout_rate=0.0,
                 drop_path_rate=0.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.filter = GlobalFilter(dim, grid_h, grid_w)
        self.norm2 = nn.LayerNorm(dim)
        self.feed_forward = FeedForward(dim, int(dim * mlp_ratio), dropout_rate)
        self.drop_path = DropPath(drop_path_rate)

    def forward(self, x):
        # The branch is wrapped in drop_path, NOT the residual. The identity
        # path must always survive, or a dropped block would output zero
        # instead of passing its input through.
        return x + self.drop_path(
            self.feed_forward(self.norm2(self.filter(self.norm1(x))))
        )


class GFNet(nn.Module):
    """
    Global Filter Network for AD vs NC classification.

    The whole pipeline:

        image  ->  patch embedding  ->  + position embedding
               ->  N x GFNetBlock
               ->  final LayerNorm  ->  average over tokens  ->  linear head

    Why a position embedding is needed. The global filter works on the token
    grid, so it does know about spatial arrangement - but the patch embedding
    produces an unordered set as far as the later Linear layers are concerned,
    and the learned position embedding lets the model represent "this patch is
    near the centre" explicitly. Anatomical position matters a lot here: the
    hippocampus and ventricles sit in specific places, so where a feature
    appears is as informative as what it is.

    Why mean pooling over tokens rather than a class token. A class token is the
    usual ViT choice, but it relies on attention to gather information into that
    one token. There is no attention here, so mean pooling over all tokens is
    the more natural read-out and is what the GFNet paper uses.

    Default size. embed_dim 256 with depth 8 gives roughly 5 million parameters.
    That is deliberately modest - with only 915 training patients, a larger model
    would overfit well before it ran out of capacity. It is still over ten times
    the baseline's 421k, which is part of what the resource comparison is meant
    to expose: the question is not just whether GFNet wins, but whether it wins
    enough to justify the cost.
    """

    def __init__(
        self,
        image_size=224,
        patch_size=16,
        in_channels=1,
        num_classes=2,
        embed_dim=256,
        depth=8,
        mlp_ratio=4.0,
        dropout_rate=0.1,
        drop_path_rate=0.0,
    ):
        super().__init__()

        self.patch_embedding = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )

        grid_size = self.patch_embedding.grid_size
        num_patches = self.patch_embedding.num_patches

        # One learnable vector per patch position, added to the patch embeddings.
        self.position_embedding = nn.Parameter(
            torch.zeros(1, num_patches, embed_dim)
        )
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

        self.embedding_dropout = nn.Dropout(dropout_rate)

        # Drop probability rises linearly from 0 at the first block to
        # drop_path_rate at the last. This is the schedule the ViT and GFNet
        # papers use.
        drop_path_schedule = [
            drop_path_rate * index / max(depth - 1, 1) for index in range(depth)
        ]

        self.blocks = nn.ModuleList([
            GFNetBlock(
                dim=embed_dim,
                grid_h=grid_size,
                grid_w=grid_size,
                mlp_ratio=mlp_ratio,
                dropout_rate=dropout_rate,
                drop_path_rate=drop_path_schedule[index],
            )
            for index in range(depth)
        ])

        self.final_norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self.apply(self._initialise_weights)

    @staticmethod
    def _initialise_weights(module):
        """
        Sensible starting weights. Truncated normal for Linear layers is the
        standard transformer choice - it avoids the occasional very large value
        a plain normal would produce, which can destabilise the first few steps.
        """
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x):
        """
        x: (batch, channels, height, width)
        Returns raw logits: (batch, num_classes)
        """
        x = self.patch_embedding(x)              # (batch, num_patches, embed_dim)
        x = x + self.position_embedding
        x = self.embedding_dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        x = x.mean(dim=1)                        # average over tokens
        x = self.head(x)
        return x

    def get_filter_magnitudes(self):
        """
        Return the magnitude of each block's learned frequency filter, averaged
        over the channel dimension.

        This exists for the report. Each returned tensor has shape
        (grid_h, freq_w) and says how strongly that block weights each spatial
        frequency. Plotting these shows what the model actually learned to look
        for - and if the low-frequency corner dominates, that supports the
        argument that the model is keying on large-scale atrophy rather than
        fine texture.

        This is the figure that explains GFNet rather than just scoring it, and
        a CNN baseline has no equivalent.
        """
        magnitudes = []
        for block in self.blocks:
            weight = torch.view_as_complex(block.filter.complex_weight)
            magnitudes.append(weight.abs().mean(dim=-1).detach().cpu())
        return magnitudes


# ===========================================================================
# Helpers
# ===========================================================================

def count_trainable_parameters(model):
    """
    Count how many parameters the model will actually learn.

    The assignment asks for a resource profiling table comparing the baseline
    and the main model, and parameter count is one of the three numbers it wants
    (alongside peak GPU memory and inference time), so this lives here rather
    than being worked out by hand.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def build_model(model_name, image_size=224, in_channels=1, num_classes=2,
                dropout_rate=None, patch_size=16, depth=8, drop_path_rate=0.0):
    """
    One place that maps a name to a model, so train.py and predict.py cannot
    drift apart on how a model is constructed.

    patch_size only affects GFNet. It sets how finely the image is cut into
    tokens, and through that, how many frequency components the global filter
    has to work with:

        patch 16 -> 14 x 14 tokens -> rfft2 gives 14 x 8  -> 112 components
        patch  8 -> 28 x 28 tokens -> rfft2 gives 28 x 15 -> 420 components

    Alzheimer's atrophy is subtle and spatially local, so 112 components may be
    too coarse an instrument for it. BaselineCNN ignores this argument.
    """
    if model_name == "baseline":
        return BaselineCNN(
            in_channels=in_channels,
            num_classes=num_classes,
            dropout_rate=0.3 if dropout_rate is None else dropout_rate,
        )

    if model_name == "gfnet":
        return GFNet(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            num_classes=num_classes,
            depth=depth,
            dropout_rate=0.1 if dropout_rate is None else dropout_rate,
            drop_path_rate=drop_path_rate,
        )

    raise ValueError("Unknown model name: %s" % model_name)

# ===========================================================================
# Self-tests
# ===========================================================================

def _test_filter_identity():
    """
    The most important test in this file.

    If I set every filter weight to exactly 1 + 0i, then the block does
    FFT, multiply by one, inverse FFT. That should hand back the original input
    unchanged. If it does, the transform plumbing - the shapes, the dimensions I
    transform over, the normalisation, the s= argument on the inverse - is all
    correct.

    If this fails, no amount of training will fix it, and the model would train
    badly for reasons that look like a bad architecture rather than a bug.
    """
    print("test: filter with unit weights returns its input unchanged")

    grid = 14
    dim = 8
    filter_layer = GlobalFilter(dim, grid, grid)

    # Set the weight to 1 + 0i everywhere.
    with torch.no_grad():
        filter_layer.complex_weight[..., 0] = 1.0   # real part
        filter_layer.complex_weight[..., 1] = 0.0   # imaginary part

    original = torch.randn(2, grid * grid, dim)
    result = filter_layer(original)

    largest_difference = (result - original).abs().max().item()
    print("   largest absolute difference: %.3e" % largest_difference)

    assert largest_difference < 1e-4, (
        "FFT round trip is not an identity - the transform setup is wrong"
    )
    print("   passed")


def _test_filter_shapes():
    """
    Check the half-spectrum width is what I think it is, and that the filter
    preserves the token shape.
    """
    print("test: filter shapes")

    for grid in (7, 14, 16):
        dim = 4
        filter_layer = GlobalFilter(dim, grid, grid)

        expected_freq_width = grid // 2 + 1
        assert filter_layer.freq_w == expected_freq_width
        assert filter_layer.complex_weight.shape == (grid, expected_freq_width, dim, 2)

        x = torch.randn(3, grid * grid, dim)
        out = filter_layer(x)
        assert out.shape == x.shape, "filter changed the token shape"

        print("   grid %2dx%-2d -> rfft width %d, weight %s  OK"
              % (grid, grid, expected_freq_width, tuple(filter_layer.complex_weight.shape)))

    print("   passed")


def _test_filter_rejects_wrong_token_count():
    """The error for a mismatched token count should be clear, not a crash."""
    print("test: filter rejects the wrong number of tokens")

    filter_layer = GlobalFilter(dim=4, grid_h=14, grid_w=14)
    try:
        filter_layer(torch.randn(1, 100, 4))   # 100 tokens, not 196
    except ValueError as error:
        print("   raised as expected: %s" % error)
        print("   passed")
        return

    raise AssertionError("expected a ValueError for the wrong token count")


def _test_gradients_reach_the_filter():
    """
    Confirm gradients actually flow back into the complex filter weights.

    Autograd through FFT operations works, but it is worth proving rather than
    assuming - if the gradient were zero the filter would never learn and the
    model would quietly be a plain MLP over patches.
    """
    print("test: gradients reach the filter weights")

    model = GFNet(image_size=64, patch_size=16, embed_dim=32, depth=2)
    images = torch.randn(2, 1, 64, 64)
    labels = torch.tensor([0, 1])

    logits = model(images)
    loss = nn.CrossEntropyLoss()(logits, labels)
    loss.backward()

    for index, block in enumerate(model.blocks):
        gradient = block.filter.complex_weight.grad
        assert gradient is not None, "block %d filter got no gradient at all" % index

        gradient_size = gradient.abs().mean().item()
        assert gradient_size > 0, "block %d filter gradient is exactly zero" % index
        print("   block %d mean |grad| = %.3e" % (index, gradient_size))

    print("   passed")


def _test_model_shapes_and_sizes():
    """Forward pass shapes, and the parameter counts for the report."""
    print("test: model output shapes and parameter counts")

    batch = 4
    images = torch.randn(batch, 1, 224, 224)

    baseline = BaselineCNN()
    baseline_out = baseline(images)
    assert baseline_out.shape == (batch, 2)

    gfnet = GFNet(image_size=224)
    gfnet_out = gfnet(images)
    assert gfnet_out.shape == (batch, 2)

    baseline_params = count_trainable_parameters(baseline)
    gfnet_params = count_trainable_parameters(gfnet)

    print("   BaselineCNN  output %s   parameters %9d"
          % (tuple(baseline_out.shape), baseline_params))
    print("   GFNet        output %s   parameters %9d"
          % (tuple(gfnet_out.shape), gfnet_params))
    print("   GFNet is %.1fx the size of the baseline"
          % (gfnet_params / baseline_params))
    print("   passed")


def _test_filter_magnitudes_helper():
    """The reporting helper should return one map per block, of the right shape."""
    print("test: get_filter_magnitudes")

    gfnet = GFNet(image_size=224, depth=3)
    magnitudes = gfnet.get_filter_magnitudes()

    assert len(magnitudes) == 3
    for index, magnitude_map in enumerate(magnitudes):
        assert magnitude_map.shape == (14, 8), (
            "expected a 14x8 map for a 14x14 grid, got %s" % (tuple(magnitude_map.shape),)
        )
    print("   %d maps, each %s  OK" % (len(magnitudes), tuple(magnitudes[0].shape)))
    print("   passed")


if __name__ == "__main__":
    # Run every check with: python modules.py
    #
    # These are quick, CPU-only, and need no data at all - which is the point.
    # I want to know the architecture is correct before spending GPU time on it.

    torch.manual_seed(0)

    print("=" * 70)
    print("GFNet and baseline self-tests")
    print("=" * 70)
    print()

    _test_filter_identity()
    print()
    _test_filter_shapes()
    print()
    _test_filter_rejects_wrong_token_count()
    print()
    _test_gradients_reach_the_filter()
    print()
    _test_model_shapes_and_sizes()
    print()
    _test_filter_magnitudes_helper()

    print()
    print("=" * 70)
    print("all checks passed")
    print("=" * 70)
