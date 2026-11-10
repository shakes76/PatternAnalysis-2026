"""Components for adapting Stable Diffusion 1.5 to ADNI brain MRI slices.

The pretrained autoencoder, text encoder and denoising UNet are the SD1.5
checkpoint (Rombach et al., "High-Resolution Image Synthesis with Latent
Diffusion Models", CVPR 2022), loaded through diffusers' model classes at a
pinned revision. Everything that turns them into a trainable MRI generator is
implemented here in PyTorch: prompt conditioning with dropout, offset noise,
the noise-prediction objective, weight averaging, and the classifier-free
guidance sampler.

Checkpoint: https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler, UNet2DConditionModel
from safetensors.torch import load_file, save_file
from transformers import CLIPTextModel, CLIPTokenizer

# The class prompts are the same as in the rank-4 LoRA run (E0), so the two
# adaptations differ only in what is trained and how, not in conditioning.
PROMPTS = {
    "AD": "a brain MRI slice, Alzheimer disease",
    "NC": "a brain MRI slice, normal control",
    "empty": "",
}
LABELS = ("AD", "NC")


def autocast(device: torch.device):
    """bfloat16 autocast on CUDA; disabled on CPU so tests compare exact float32."""
    return torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    )


@torch.no_grad()
def encode_prompts(model_dir: Path, device: torch.device) -> dict[str, torch.Tensor]:
    """Encode the fixed prompts once. The text encoder stays frozen and is freed."""
    tokenizer = CLIPTokenizer.from_pretrained(
        model_dir, subfolder="tokenizer", local_files_only=True
    )
    encoder = (
        CLIPTextModel.from_pretrained(
            model_dir,
            subfolder="text_encoder",
            variant="fp16",
            local_files_only=True,
            use_safetensors=True,
        )
        .eval()
        .requires_grad_(False)
        .to(device, dtype=torch.float32)
    )
    embeddings = {}
    for key, prompt in PROMPTS.items():
        tokens = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        embeddings[key] = encoder(tokens.input_ids.to(device))[0]
    return embeddings


def load_vae(vae_dir: Path, device: torch.device) -> AutoencoderKL:
    """Frozen SD1.5 autoencoder: 256x256 images <-> 4x32x32 latents."""
    return (
        AutoencoderKL.from_pretrained(
            vae_dir, subfolder="vae", local_files_only=True, use_safetensors=True
        )
        .eval()
        .requires_grad_(False)
        .to(device, dtype=torch.float32)
    )


def load_unet(model_dir: Path, device: torch.device) -> UNet2DConditionModel:
    """Pretrained SD1.5 denoiser in float32 master weights."""
    return UNet2DConditionModel.from_pretrained(
        model_dir,
        subfolder="unet",
        variant="fp16",
        local_files_only=True,
        use_safetensors=True,
    ).to(device, dtype=torch.float32)


# B3 (docs/b3-ldm-scratch-protocol.md): SD1.5's block layout and conditioning at
# about a ninth of the width, 92.2 M parameters, randomly initialised. It reads the
# same 4x32x32 latents and the same 768-wide CLIP prompt embeddings as SD1.5.
SCRATCH_UNET = dict(
    sample_size=32,
    in_channels=4,
    out_channels=4,
    layers_per_block=2,
    block_out_channels=(128, 256, 384, 384),
    down_block_types=("CrossAttnDownBlock2D",) * 3 + ("DownBlock2D",),
    up_block_types=("UpBlock2D",) + ("CrossAttnUpBlock2D",) * 3,
    cross_attention_dim=768,
    attention_head_dim=8,
)


def build_unet(
    model_dir: Path, device: torch.device, init: str = "pretrained"
) -> UNet2DConditionModel:
    """The pretrained SD1.5 UNet, or the smaller B3 UNet from random weights."""
    if init == "pretrained":
        return load_unet(model_dir, device)
    if init == "scratch":
        return UNet2DConditionModel(**SCRATCH_UNET).to(device, dtype=torch.float32)
    raise ValueError(f"Unknown UNet initialisation: {init}")


def load_noise_schedule(model_dir: Path) -> DDPMScheduler:
    """The 1000-step training schedule shipped with the checkpoint."""
    return DDPMScheduler.from_pretrained(
        model_dir, subfolder="scheduler", local_files_only=True
    )


def add_lora(unet: UNet2DConditionModel, rank: int) -> None:
    """Attach the E0 adapter: rank-r LoRA (Hu et al., ICLR 2022) on every
    attention projection, through the peft library."""
    from peft import LoraConfig

    unet.add_adapter(
        LoraConfig(
            r=rank,
            lora_alpha=rank,
            lora_dropout=0.0,
            init_lora_weights="gaussian",
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        )
    )


def configure_trainable(
    unet: UNet2DConditionModel, mode: str, lora_rank: int = 4
) -> dict[str, torch.nn.Parameter]:
    """Choose what is trained and return exactly those parameters by name.

    "unet" fine-tunes every UNet weight (E1). "lora" freezes the UNet and trains
    only attention adapters (E0), kept for the ablation that isolates capacity.
    """
    if mode == "unet":
        unet.requires_grad_(True)
    elif mode == "lora":
        unet.requires_grad_(False)
        add_lora(unet, lora_rank)
    else:
        raise ValueError(f"Unknown trainable mode: {mode}")
    trainable = {n: p for n, p in unet.named_parameters() if p.requires_grad}
    if not trainable:
        raise RuntimeError("No trainable parameters")
    if mode == "lora" and any("lora_" not in name for name in trainable):
        raise RuntimeError("LoRA mode would train base UNet weights")
    return trainable


def training_noise(
    latents: torch.Tensor, offset: float, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Gaussian noise plus an optional per-image, per-channel constant.

    SD1.5's schedule leaves some signal at its final timestep, so a model trained
    on zero-mean noise rarely learns to move an image's overall brightness, and
    samples drift toward mid-grey (Guttenberg, "Diffusion with Offset Noise",
    2023; Lin et al., "Common Diffusion Noise Schedules and Sample Steps are
    Flawed", WACV 2024). Brain slices are mostly black background, which is
    that case. The constant term gives the model a target for the global mean.
    """
    noise = torch.randn(
        latents.shape, generator=generator, device=latents.device, dtype=latents.dtype
    )
    if offset:
        shift = torch.randn(
            (latents.shape[0], latents.shape[1], 1, 1),
            generator=generator,
            device=latents.device,
            dtype=latents.dtype,
        )
        noise = noise + offset * shift
    return noise


def condition_batch(
    label_index: torch.Tensor,
    embeddings: dict[str, torch.Tensor],
    dropout: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Prompt embedding per sample, with a fraction replaced by the empty prompt.

    Classifier-free guidance mixes an unconditional and a conditional prediction
    (Ho & Salimans, "Classifier-Free Diffusion Guidance", 2022), so the
    unconditional branch must be trained as well. Without dropout it keeps the
    natural-image prior and guidance extrapolates away from it.
    label_index holds 0 for AD and 1 for NC, matching LABELS.
    """
    table = torch.cat([embeddings[label] for label in LABELS])
    conditioning = table[label_index]
    if dropout:
        drop = torch.rand(
            label_index.shape[0], generator=generator, device=label_index.device
        )
        conditioning = torch.where(
            (drop < dropout)[:, None, None],
            embeddings["empty"].expand_as(conditioning),
            conditioning,
        )
    return conditioning


def diffusion_loss(
    unet: UNet2DConditionModel,
    latents: torch.Tensor,
    conditioning: torch.Tensor,
    schedule: DDPMScheduler,
    noise: torch.Tensor,
    timesteps: torch.Tensor | None = None,
) -> torch.Tensor:
    """Noise-prediction objective of latent diffusion (Rombach et al., eq. 2).

    z_t = sqrt(abar_t) * z_0 + sqrt(1 - abar_t) * eps
    L   = || eps - eps_theta(z_t, t, c) ||^2,  with t uniform over the schedule.
    """
    if timesteps is None:
        timesteps = torch.randint(
            0,
            schedule.config.num_train_timesteps,
            (latents.shape[0],),
            device=latents.device,
        )
    noisy = schedule.add_noise(latents, noise, timesteps)
    if schedule.config.prediction_type == "epsilon":
        target = noise
    elif schedule.config.prediction_type == "v_prediction":
        target = schedule.get_velocity(latents, noise, timesteps)
    else:
        raise ValueError(
            f"Unsupported prediction type: {schedule.config.prediction_type}"
        )
    with autocast(latents.device):
        prediction = unet(noisy, timesteps, conditioning).sample
    return F.mse_loss(prediction.float(), target.float())


def latents_from_moments(
    mean: torch.Tensor,
    std: torch.Tensor,
    scaling_factor: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Draw z ~ N(mean, std^2) from cached VAE posteriors, then scale for the UNet.

    Equivalent to vae.encode(x).latent_dist.sample() * scaling_factor, without
    re-running the encoder on every step.
    """
    eps = torch.randn(
        mean.shape, generator=generator, device=mean.device, dtype=mean.dtype
    )
    return (mean + std * eps) * scaling_factor


class ExponentialMovingAverage:
    """Shadow copy of the trainable weights, updated after every optimiser step.

    Single-step diffusion weights are noisy, so samples are drawn from averaged
    weights (Ho et al., 2020). The decay warms up as (1 + n) / (10 + n) so the
    average is not anchored to the pretrained starting point for thousands of
    steps.
    """

    def __init__(self, parameters: dict[str, torch.nn.Parameter], decay: float):
        if not 0.0 < decay < 1.0:
            raise ValueError("decay must lie strictly between 0 and 1")
        self.decay = decay
        self.updates = 0
        self.shadow = {n: p.detach().clone().float() for n, p in parameters.items()}

    def current_decay(self) -> float:
        return min(self.decay, (1 + self.updates) / (10 + self.updates))

    @torch.no_grad()
    def update(self, parameters: dict[str, torch.nn.Parameter]) -> None:
        self.updates += 1
        weight = 1.0 - self.current_decay()
        for name, param in parameters.items():
            self.shadow[name].lerp_(param.detach().float(), weight)

    @contextlib.contextmanager
    def applied_to(self, parameters: dict[str, torch.nn.Parameter]):
        """Temporarily load the averaged weights; the live weights come back after."""
        live = {n: p.detach().clone() for n, p in parameters.items()}
        with torch.no_grad():
            for name, param in parameters.items():
                param.copy_(self.shadow[name])
        try:
            yield
        finally:
            with torch.no_grad():
                for name, param in parameters.items():
                    param.copy_(live[name])


def save_weights(
    weights: dict[str, torch.Tensor], path: Path, dtype=torch.float16
) -> None:
    """Store trained weights as safetensors, float16 by default (about 1.7 GB
    for a full UNet)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    save_file(
        {n: w.detach().to("cpu", dtype).contiguous() for n, w in weights.items()},
        str(temporary),
    )
    temporary.replace(path)


def load_weights(unet: UNet2DConditionModel, path: Path) -> str:
    """Load weights written by save_weights into a pretrained UNet.

    Adapter files are recognised by their parameter names; the matching adapter
    is attached first. Returns "lora" or "unet" so callers can report it.
    """
    state = load_file(str(path))
    mode = "unet"
    if any("lora_" in name for name in state):
        mode = "lora"
        rank = next(w.shape[0] for n, w in state.items() if "lora_A" in n)
        add_lora(unet, rank)
    parameters = dict(unet.named_parameters())
    unknown = set(state) - set(parameters)
    if unknown:
        raise KeyError(f"{len(unknown)} saved weights do not match the UNet")
    if mode == "unet" and set(state) != set(parameters):
        raise KeyError("Full-UNet file does not cover every UNet parameter")
    with torch.no_grad():
        for name, value in state.items():
            parameters[name].copy_(value.to(parameters[name].dtype))
    return mode


@torch.no_grad()
def guided_ddim(
    unet: UNet2DConditionModel,
    schedule_config,
    initial: torch.Tensor,
    condition: torch.Tensor,
    uncondition: torch.Tensor,
    guidance: float,
    steps: int = 30,
    eta: float = 0.0,
) -> torch.Tensor:
    """DDIM sampling (Song et al., 2021) with classifier-free guidance.

    eps = eps(x_t, empty) + w * (eps(x_t, c) - eps(x_t, empty))

    Mirrors diffusers' StableDiffusionPipeline step for step, so cohorts from
    this sampler compare directly with the E0 cohorts generated by that
    pipeline; tests/test_modules.py checks the two agree.
    """
    scheduler = DDIMScheduler.from_config(schedule_config)
    scheduler.set_timesteps(steps, device=initial.device)
    latents = initial * scheduler.init_noise_sigma
    guided = guidance > 1.0
    # One prompt embedding serves a whole batch of starting latents.
    condition = condition.expand(len(initial), -1, -1)
    uncondition = uncondition.expand(len(initial), -1, -1)
    context = torch.cat([uncondition, condition]) if guided else condition
    for timestep in scheduler.timesteps:
        model_input = torch.cat([latents, latents]) if guided else latents
        model_input = scheduler.scale_model_input(model_input, timestep)
        with autocast(initial.device):
            prediction = unet(model_input, timestep, context).sample.float()
        if guided:
            unconditional, conditional = prediction.chunk(2)
            prediction = unconditional + guidance * (conditional - unconditional)
        latents = scheduler.step(prediction, timestep, latents, eta=eta).prev_sample
    return latents


@torch.no_grad()
def ddim_invert(
    unet: UNet2DConditionModel,
    schedule_config,
    latents: torch.Tensor,
    condition: torch.Tensor,
    steps: int = 30,
) -> torch.Tensor:
    """Deterministic DDIM inversion without guidance: a clean latent -> the
    starting noise that guided_ddim(..., guidance=1.0) maps back to it.

    Runs the DDIM update with the time order reversed, using the noise predicted
    at the current (less noisy) step, the usual first-order approximation.
    """
    scheduler = DDIMScheduler.from_config(schedule_config)
    scheduler.set_timesteps(steps, device=latents.device)
    alphas = scheduler.alphas_cumprod.to(latents.device)
    timesteps = scheduler.timesteps.flip(0)  # from little noise to the most
    condition = condition.expand(len(latents), -1, -1)
    previous_alpha = scheduler.final_alpha_cumprod.to(latents.device)
    for timestep in timesteps:
        with autocast(latents.device):
            noise = unet(latents, timestep, condition).sample.float()
        alpha = alphas[timestep]
        clean = (latents - (1 - previous_alpha).sqrt() * noise) / previous_alpha.sqrt()
        latents = alpha.sqrt() * clean + (1 - alpha).sqrt() * noise
        previous_alpha = alpha
    return latents / scheduler.init_noise_sigma


def slerp(a: torch.Tensor, b: torch.Tensor, weight: float) -> torch.Tensor:
    """Spherical interpolation between two noise tensors, which keeps the norm
    a Gaussian sample is expected to have; linear mixing would shrink it
    (Shoemake, "Animating Rotation with Quaternion Curves", SIGGRAPH 1985)."""
    flat_a, flat_b = a.flatten(), b.flatten()
    cosine = torch.dot(flat_a, flat_b) / (flat_a.norm() * flat_b.norm())
    angle = torch.arccos(cosine.clamp(-1, 1))
    if angle.abs() < 1e-6:
        return (1 - weight) * a + weight * b
    return (
        torch.sin((1 - weight) * angle) * a + torch.sin(weight * angle) * b
    ) / torch.sin(angle)


@torch.no_grad()
def decode(vae: AutoencoderKL, latents: torch.Tensor) -> torch.Tensor:
    """Latents -> RGB images in [0, 1]."""
    images = vae.decode(latents / vae.config.scaling_factor).sample
    if not torch.isfinite(images).all():
        raise RuntimeError("Non-finite decoded image")
    return (images / 2 + 0.5).clamp(0, 1)


def initial_noise(seed: int, size: int, device: torch.device) -> torch.Tensor:
    """Seeded starting latent, drawn exactly as for the E0 cohorts."""
    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.randn((1, 4, size // 8, size // 8), generator=generator, device=device)


def to_grayscale(images: torch.Tensor) -> torch.Tensor:
    """Average the RGB channels. MRI is single-channel; any spread is an artefact."""
    return images.mean(dim=1, keepdim=True)
