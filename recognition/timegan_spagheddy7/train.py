"""Train AR or TimeGAN on the chronological training period."""

import argparse
import json
import hashlib
import os
import time
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from dataset import FEATURE_NAMES, PRICE_COLUMNS, PRICE_DIVISOR, PROJECT_DIR, prepare_datasets, sha256_file
from modules import VectorAR, TimeGAN, temporal_loss, moment_loss, order_penalty
from evaluation import conditioning_history, ValidationEvaluator, DepthSSIM, decode_batch, metric_specification, validation_windows, validity_rates
from protocol import comparison_protocol, comparison_signature, fit_anchor_context, anchor_context


def write_json(path, value):
    """Replace complete JSON atomically, retrying transient Windows file locks."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for attempt in range(20):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(.1 + .05 * attempt)


def skip_supervisor_draws(sampler, count, steps, device="cpu"):
    """Keep joint minibatch draws paired when the supervisor warm-up is ablated."""
    for _ in range(steps):
        torch.randperm(count, generator=sampler, device=device)


def reconstruction_price_weight(training_values, power=1.):
    """Balance price/volume group variances using training windows only."""
    if not np.isfinite(power) or not 0 <= power <= 1:
        raise ValueError("Reconstruction balance power must be between zero and one.")
    if training_values.ndim != 3 or training_values.shape[-1] % 2 or training_values.shape[-1] < 2:
        raise ValueError("Expected windows with alternating price and volume features.")
    variance = training_values.double().var(dim=(0, 1), unbiased=False)
    price, volume = variance[0::2].mean(), variance[1::2].mean()
    if not bool(torch.isfinite(variance).all()) or min(float(price), float(volume)) <= 0:
        raise ValueError("Price and volume groups need finite, positive training variance.")
    return float(volume / price) ** power


class TimeGANTrainer:
    """Separate optimisers and gradient routes for TimeGAN's three stages.

    Reference losses by default, independent Adam states and 2:1 G/E-to-D schedule.
    An optional, explicitly labelled order penalty uses training anchors only.
    No gradient clipping or extra epsilon in reconstruction/supervised roots.
    """

    def __init__(self, model, learning_rate=1e-3, supervised_weight=100., moment_weight=100., gamma=1.,
                 structure_weight=0., minimum=None, span=None, margin=.01, price_reconstruction_weight=1., no_supervision=False):
        self.model = model
        self.no_supervision = no_supervision
        if not np.isfinite(price_reconstruction_weight) or price_reconstruction_weight <= 0:
            raise ValueError("Price reconstruction weight must be finite and positive.")
        self.reconstruction_weights = None
        if price_reconstruction_weight != 1.:
            if model.config["features"] % 2:
                raise ValueError("Weighted reconstruction expects alternating price and volume features.")
            self.reconstruction_weights = next(model.parameters()).new_ones(model.config["features"])
            self.reconstruction_weights[0::2] = price_reconstruction_weight
        if not np.isfinite(structure_weight) or structure_weight < 0 or (structure_weight and (minimum is None or span is None)):
            raise ValueError("Structural loss needs a nonnegative weight and the training scaler.")
        self.structure_weight, self.minimum, self.span, self.margin = structure_weight, minimum, span, margin
        self.supervised_weight, self.moment_weight, self.gamma = supervised_weight, moment_weight, gamma
        groups = {"embedding_pretrain": ("embedder", "recovery"), "embedding": ("embedder", "recovery"), "supervised": ("supervisor",),
                  "generator": ("generator", "supervisor"), "discriminator": ("discriminator",)}
        self.groups = groups
        self.parameters = {key: [p for name in names for p in getattr(model, name).parameters()] for key, names in groups.items()}
        self.optimizers = {key: torch.optim.Adam(parameters, lr=learning_rate) for key, parameters in self.parameters.items()}

    def activate(self, group):
        self.model.train()
        self.model.zero_grad(set_to_none=True)
        for name, network in self.model.named_children():
            network.requires_grad_(name in self.groups[group])

    def update(self, loss, group, collect):
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite {group} loss.")
        loss.backward()
        # Norms are diagnostics, not clipping; avoid their GPU work on unlogged updates.
        norm = torch.linalg.vector_norm(torch.stack([p.grad.detach().norm() for p in self.parameters[group] if p.grad is not None])) if collect else None
        if collect and not torch.isfinite(norm):
            raise FloatingPointError(f"Non-finite {group} gradient.")
        self.optimizers[group].step()
        return norm

    def embedding_step(self, real, joint=False, collect=True, anchors=None, context=None):
        group = "embedding" if joint else "embedding_pretrain"
        self.activate(group)
        latent = self.model.embedder(real, context)
        recovered = self.model.recovery(latent, context)
        reconstruction = torch.nn.functional.mse_loss(recovered, real)
        supervision = temporal_loss(latent, self.model.supervisor(latent, context)) if joint and not self.no_supervision else reconstruction.new_zeros(())
        objective_mse = reconstruction if self.reconstruction_weights is None else ((recovered - real).square() * self.reconstruction_weights).mean()
        loss = 10 * objective_mse.sqrt() + (0.1 * supervision if joint else 0)
        if self.structure_weight:
            structure = order_penalty(recovered, anchors, self.minimum, self.span, self.margin)
            loss = loss + self.structure_weight * structure
        norm = self.update(loss, group, collect)
        result = dict(reconstruction_mse=reconstruction.detach(), embedding_temporal_mse=supervision.detach(), embedding_grad_norm=norm) if collect else {}
        if collect and self.reconstruction_weights is not None:
            result["weighted_reconstruction_mse"] = objective_mse.detach()
        if collect and self.structure_weight:
            result["reconstruction_order_penalty"] = structure.detach()
        return result

    def supervisor_step(self, real, collect=True, context=None):
        if self.no_supervision:
            raise ValueError("The no-supervision ablation must skip supervisor pretraining.")
        self.activate("supervised")
        with torch.no_grad():
            latent = self.model.embedder(real, context)
        loss = temporal_loss(latent, self.model.supervisor(latent, context))
        norm = self.update(loss, "supervised", collect)
        return dict(supervised_mse=loss.detach(), supervisor_grad_norm=norm) if collect else {}

    @staticmethod
    def classify(logits, real):
        targets = torch.ones_like(logits) if real else torch.zeros_like(logits)
        return torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)

    def generator_step(self, real, noise, collect=True, anchors=None, context=None):
        self.activate("generator")
        if not self.no_supervision:
            with torch.no_grad():
                latent = self.model.embedder(real, context)
        raw_fake = self.model.generator(noise, context)
        fake = self.model.supervisor(raw_fake, context)
        generated = self.model.recovery(fake, context)  # Frozen recovery still passes gradients to its input.
        adversarial = self.classify(self.model.discriminator(fake, context), True) + self.gamma * self.classify(self.model.discriminator(raw_fake, context), True)
        supervision = adversarial.new_zeros(())
        if not self.no_supervision:
            supervision = temporal_loss(latent, self.model.supervisor(latent, context))
        moments = moment_loss(real, generated)
        # Bypass the absent term entirely: zero times sqrt(0) can have undefined gradients.
        loss = adversarial + (0 if self.no_supervision else self.supervised_weight * supervision.sqrt()) + self.moment_weight * moments
        if self.structure_weight:
            structure = order_penalty(generated, anchors, self.minimum, self.span, self.margin)
            loss = loss + self.structure_weight * structure
        norm = self.update(loss, "generator", collect)
        result = dict(generator_loss=loss.detach(), generator_adversarial=adversarial.detach(),
                      supervised_mse=supervision.detach(), moment_error=moments.detach(), generator_grad_norm=norm) if collect else {}
        if collect and self.structure_weight:
            result["generator_order_penalty"] = structure.detach()
        return result

    def discriminator_step(self, real, noise, threshold=.15, collect=True, context=None):
        self.activate("discriminator")
        with torch.no_grad():
            latent = self.model.embedder(real, context)
            raw_fake = self.model.generator(noise, context)
            fake = self.model.supervisor(raw_fake, context)
        real_logits, fake_logits = self.model.discriminator(latent, context), self.model.discriminator(fake, context)
        loss = self.classify(real_logits, True) + self.classify(fake_logits, False) + self.gamma * self.classify(self.model.discriminator(raw_fake, context), False)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite discriminator loss.")
        updated = float(loss.detach()) > threshold
        norm = self.update(loss, "discriminator", collect) if updated else loss.new_zeros(())
        return dict(discriminator_loss=loss.detach(), discriminator_grad_norm=norm, discriminator_updated=loss.new_tensor(updated),
                    discriminator_real_probability=real_logits.detach().sigmoid().mean(),
                    discriminator_fake_probability=fake_logits.detach().sigmoid().mean()) if collect else {}


def timegan_code_hashes():
    names = ("modules.py", "train.py", "predict.py", "dataset.py", "evaluation.py", "protocol.py")
    return {name: hashlib.sha256((PROJECT_DIR / name).read_text(encoding="utf-8").encode()).hexdigest() for name in names}


def reconstruction_check(values, architecture, device, steps=600, contexts=None):
    """Independent eight-window learning check; these weights are discarded."""
    torch.manual_seed(9001)
    probe = TimeGAN(**architecture).to(device)
    trainer = TimeGANTrainer(probe)
    indices = torch.linspace(0, len(values) - 1, 8).long()
    batch = values[indices].to(device)
    context = None if contexts is None else contexts[indices].to(device)
    with torch.no_grad():
        before = float((probe.reconstruct(batch, context) - batch).square().mean())
        constant = float((batch - batch.mean(dim=(0, 1), keepdim=True)).square().mean())
    for _ in range(steps):
        trainer.embedding_step(batch, collect=False, context=context)
    with torch.no_grad():
        after = float((probe.reconstruct(batch, context) - batch).square().mean())
    return dict(windows=8, steps=steps, initial_mse=before, final_mse=after, constant_feature_mean_mse=constant,
                passed=bool(after < min(.5 * before, constant)), scope="Training-only reconstruction check; independent model discarded.")


def fit_timegan(data_dir, output_dir=None, report_path=None, embedding_steps=1000, supervised_steps=1000, joint_steps=1000,
                eval_every=250, seed=2026, batch_size=128, hidden=24, layers=3, learning_rate=1e-3, device="auto", sanity_steps=600, threads=4,
                structure_weight=0., balance_reconstruction=False, balance_power=1., no_supervision=False, anchor_conditioning="none"):
    """Reference-structure experiment; fixed validation selection, no test scoring."""
    if min(embedding_steps, supervised_steps, joint_steps, eval_every, batch_size, sanity_steps, threads) < 1 or not 0 <= seed < 2**63 - 1000:
        raise ValueError("Use positive training budgets and a valid seed.")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Learning rate must be finite and positive.")
    if anchor_conditioning not in ("none", "matched", "zero"):
        raise ValueError("Unknown anchor conditioning mode.")
    if not np.isfinite(balance_power) or not 0 <= balance_power <= 1 or (not balance_reconstruction and balance_power != 1.):
        raise ValueError("Use balance-reconstruction with a balance power between zero and one.")
    device = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
    if device not in ("cpu", "cuda") or (device == "cuda" and not torch.cuda.is_available()):
        raise ValueError("Requested training device is unavailable.")
    torch.set_num_threads(threads)
    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = False, True
    if device == "cuda":
        total_memory = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(7 * 1024**3 / total_memory, 1.0))
    output_dir = Path(output_dir or PROJECT_DIR / "outputs/timegan")
    report_path = Path(report_path or PROJECT_DIR / "reports/timegan/results.json")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    previous_runs = []
    if report_path.exists():
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        previous_runs = previous.get("previous_runs", []) + [{key: previous[key] for key in
                         ("config", "selected", "source_files", "total_seconds", "stage_training_seconds", "selected_checkpoint_sha256")}]
        checkpoint = output_dir / "selected.pt"
        if checkpoint.exists() and sha256_file(checkpoint) == previous["selected_checkpoint_sha256"]:
            archive = output_dir / "previous" / f"{previous['selected_checkpoint_sha256'][:16]}.pt"
            archive.parent.mkdir(exist_ok=True)
            if not archive.exists():
                archive.write_bytes(checkpoint.read_bytes())
    source_files = timegan_code_hashes()
    started = perf_counter()
    datasets, scaler, audit = prepare_datasets(data_dir, 32, 32)
    training, validation = datasets["train"], datasets["validation"]
    del datasets  # Full-day audit reads test rows; no test model outputs are produced.
    values = torch.stack([training[i]["features"] for i in range(len(training))])
    price_weight = reconstruction_price_weight(values, balance_power) if balance_reconstruction else 1.
    starts, anchors, real = validation_windows(validation, 4)
    training_anchors = torch.as_tensor(training.book[training.starts][:, PRICE_COLUMNS] / PRICE_DIVISOR)
    context_scaler = None if anchor_conditioning == "none" else fit_anchor_context(training_anchors.numpy())
    contexts = anchor_context(training_anchors.numpy(), context_scaler, anchor_conditioning)
    contexts = None if contexts is None else torch.from_numpy(contexts)
    validation_context = anchor_context(anchors, context_scaler, anchor_conditioning)
    spec, protocol = metric_specification(training), comparison_protocol(context_scaler=context_scaler)
    signature = comparison_signature(protocol, spec, starts + validation.raw_offset, audit["files"], scaler.to_dict())
    ar = json.loads((PROJECT_DIR / "reports/ar" / ("results.json" if context_scaler is None else "anchor_conditioned.json")).read_text(encoding="utf-8"))
    if signature != ar["comparison_signature"]:
        raise ValueError("AR and TimeGAN comparison contracts differ; resolve the discrepancy before training.")
    reference = ValidationEvaluator(real, anchors, spec)
    reference_feature_std = np.stack([validation.features[s:s + 32] for s in starts]).reshape(-1, 40).std(axis=0).tolist()
    loading_seconds = perf_counter() - started
    architecture = dict(features=40, hidden=hidden, layers=layers, noise_dim=40, cell="reference")
    if context_scaler is not None:
        architecture["context_dim"] = 19
    print("Checking reconstruction on eight training windows...", flush=True)
    sanity_started = perf_counter()
    sanity = reconstruction_check(values, architecture, device, sanity_steps, contexts)
    sanity["seconds"] = perf_counter() - sanity_started
    print(f"Reconstruction check: {sanity}", flush=True)
    if not sanity["passed"]:
        write_json(output_dir / "failed_reconstruction_check.json", sanity)
        raise RuntimeError("Reconstruction did not beat a constant-feature predictor. Inspect the check before adversarial training.")
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(seed)
    model = TimeGAN(**architecture).to(device)
    # Default weight zero preserves the reference objective. The extension uses training data only.
    asks, bids = training.book[:, 0::4], training.book[:, 2::4]
    margin = float(min(np.diff(asks, axis=1).min(), (-np.diff(bids, axis=1)).min(), (asks[:, 0] - bids[:, 0]).min()) / PRICE_DIVISOR)
    trainer = TimeGANTrainer(model, learning_rate, structure_weight=structure_weight,
                            minimum=torch.as_tensor(scaler.minimum, device=device), span=torch.as_tensor(scaler.span, device=device), margin=margin,
                            price_reconstruction_weight=price_weight, no_supervision=no_supervision)
    values = values.to(device)  # About 27 MiB: one transfer instead of copying every minibatch.
    sampler, noise_rng = torch.Generator(device=device).manual_seed(seed + 100), torch.Generator(device=device).manual_seed(seed + 200)
    anchor_values = training_anchors.to(device)
    context_values = None if contexts is None else contexts.to(device)
    config = dict(architecture=architecture, training_seed=seed, batch_size=batch_size, learning_rate=learning_rate,
                  embedding_steps=embedding_steps, supervised_steps=supervised_steps, joint_steps=joint_steps,
                  eval_every=eval_every, sequence_length=32, generation_seeds=protocol["generation_seeds"], dtype="float32",
                  supervised_weight=0. if no_supervision else 100., embedding_supervised_weight=0. if no_supervision else .1,
                  no_supervision=no_supervision, effective_supervised_steps=0 if no_supervision else supervised_steps,
                  moment_weight=100., gamma=1., gradient_clip_norm=None, square_root_epsilon=0.,
                  discriminator_threshold=.15, generator_updates_per_joint_step=2, memory_budget_gib=7., threads=threads,
                  log_every=100, training_tensor_device=device, reference="jsyoon0823/TimeGAN@8f6181cb9b9d2fa0c930cd902411d9ac8a308e07")
    config.update(structure_weight=structure_weight, structure_margin_usd=margin,
                  objective="reference + training-anchor order penalty" if structure_weight else "reference")
    config["anchor_conditioning"] = anchor_conditioning
    if context_scaler is not None:
        config["objective"] += " + static anchor context" if anchor_conditioning == "matched" else " + zero-context architecture control"
    if no_supervision:
        config["objective"] += " without temporal supervision"
        config["supervision_ablation"] = "No supervisor warm-up or temporal terms in E/R and G/S; same five networks, adversarial/moment terms and joint minibatch/noise streams."
    if balance_reconstruction:
        config.update(balance_reconstruction=True, balance_power=balance_power, price_reconstruction_weight=price_weight,
                      reconstruction_weight_scope="(Training-window volume variance / price variance) ** balance_power; volume weights remain one.")
        config["objective"] += " + variance-balanced reconstruction"
    history, validation_history, best, best_examples = [], [], None, None
    stage_seconds = {}

    def batch():
        indices = torch.randperm(len(values), generator=sampler, device=device)[:batch_size]
        return values[indices], anchor_values[indices], None if context_values is None else context_values[indices]

    def noise(count):
        return torch.rand(count, 32, architecture["noise_dim"], generator=noise_rng, device=device)

    def save_model(name, stage, step):
        checkpoint = {"format_version": 1, "model_type": "timegan", "architecture": architecture,
                      "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "feature_names": FEATURE_NAMES,
                      "scaler": {"minimum": torch.from_numpy(scaler.minimum), "maximum": torch.from_numpy(scaler.maximum)},
                      "training_price_anchors": training_anchors, "config": config, "stage": stage, "step": step,
                      "comparison_signature": signature, "source_files": source_files, "anchor_context_scaler": context_scaler}
        torch.save(checkpoint, output_dir / name)

    def evaluate(step):
        runs, samples, feature_summary = [], {}, None
        for generation_seed in protocol["generation_seeds"]:
            generated_at = perf_counter()
            features = model.generate(len(starts), 32, generation_seed, batch_size, context=validation_context)
            generation_seconds = perf_counter() - generated_at
            books = decode_batch(features, scaler, anchors)
            scores, details = reference.evaluate(books)
            runs.append(dict(seed=generation_seed, generation_seconds=generation_seconds, **scores))
            if generation_seed == protocol["generation_seeds"][0]:
                samples = dict(real=real[:3].tolist(), generated=books[:3].tolist(), anchors=anchors[:3].tolist(),
                               raw_anchor_indices=(starts[:3] + validation.raw_offset).tolist())
                feature_summary = dict(real_feature_std=reference_feature_std,
                                       generated_feature_std=features.flatten(0, 1).std(dim=0, unbiased=False).tolist(),
                                       mean_between_sequence_std=float(features.std(dim=0, unbiased=False).mean()),
                                       unique_sequences=int(len(np.unique(features.numpy().reshape(len(features), -1), axis=0))))
        summary = {f"{key}_{suffix}": float(operation([row[key] for row in runs])) for key in runs[0] if key != "seed"
                   for suffix, operation in (("mean", np.mean), ("sd", lambda a: np.std(a, ddof=1)))}
        return dict(step=step, runs=runs, **summary, diversity=feature_summary), samples

    for stage, steps in (("embedding", embedding_steps), ("supervised", supervised_steps), ("joint", joint_steps)):
        stage_started, evaluation_seconds = perf_counter(), 0.
        if stage == "supervised" and no_supervision:
            skip_supervisor_draws(sampler, len(values), steps, device)
            stage_seconds[stage] = 0.
            print(f"supervised: skipped {steps} scheduled updates; advanced sampler to preserve paired joint batches", flush=True)
            continue
        for step in range(1, steps + 1):
            collect = step == 1 or step % config["log_every"] == 0 or step == steps
            if stage == "embedding":
                real_batch, price_anchors, context = batch()
                losses = trainer.embedding_step(real_batch, collect=collect, anchors=price_anchors, context=context)
            elif stage == "supervised":
                real_batch, _, context = batch()
                losses = trainer.supervisor_step(real_batch, collect=collect, context=context)
            else:
                updates = []
                for _ in range(2):
                    real_batch, price_anchors, context = batch()
                    updates.append({**trainer.generator_step(real_batch, noise(len(real_batch)), collect=collect, anchors=price_anchors, context=context),
                                    **trainer.embedding_step(real_batch, joint=True, collect=collect, anchors=price_anchors, context=context)})
                losses = {key: torch.stack([row[key] for row in updates]).mean() for key in updates[0]}
                real_batch, _, context = batch()
                losses.update(trainer.discriminator_step(real_batch, noise(len(real_batch)), collect=collect, context=context))
            if collect:
                losses = dict(zip(losses, torch.stack(list(losses.values())).cpu().tolist()))
                if not all(np.isfinite(list(losses.values()))):
                    raise FloatingPointError(f"Non-finite {stage} diagnostics at step {step}.")
                history.append(dict(stage=stage, step=step, training_seconds=perf_counter() - stage_started - evaluation_seconds, **losses))
                print(f"{stage} {step}/{steps}: " + ", ".join(f"{k}={v:.5f}" for k, v in losses.items() if "grad" not in k), flush=True)
                write_json(output_dir / "progress.json", dict(config=config, history=history))
            if stage == "joint" and (step % eval_every == 0 or step == steps):
                if device == "cuda":
                    torch.cuda.synchronize()
                evaluation_started = perf_counter()
                row, examples = evaluate(step)
                validation_history.append(row)
                rank = lambda candidate: (candidate["invalid_rate_mean"], candidate["mean_kl_mean"], candidate["abs_return_acf_mae_mean"], candidate["step"])
                if best is None or rank(row) < rank(best):
                    best, best_examples = row, examples
                    save_model("selected.pt", stage, step)
                print(f"Validation {step}: invalid={row['invalid_rate_mean']:.2%}, SSIM={row['depth_ssim_mean']:.4f}", flush=True)
                evaluation_seconds += perf_counter() - evaluation_started
        if device == "cuda":
            torch.cuda.synchronize()
        stage_seconds[stage] = perf_counter() - stage_started - evaluation_seconds
    save_model("joint_last.pt", "joint", joint_steps)

    # Verify the selected checkpoint, not just the final in-memory model.
    checkpoint = torch.load(output_dir / "selected.pt", map_location=device, weights_only=True)
    reloaded = TimeGAN(**checkpoint["architecture"]).to(device)
    reloaded.load_state_dict(checkpoint["state_dict"])
    replay = decode_batch(reloaded.generate(len(starts), 32, protocol["generation_seeds"][0], batch_size, context=validation_context)[:3], scaler, anchors[:3])
    reload_difference = float(np.abs(replay - np.asarray(best_examples["generated"])).max())
    if not np.allclose(replay, best_examples["generated"], atol=1e-7, rtol=1e-7):
        raise RuntimeError("Selected checkpoint failed the saved-example reload check.")
    example_ssim = DepthSSIM(np.asarray(best_examples["real"]), spec["depth_ssim"]).score(np.asarray(best_examples["generated"]))[0]
    example_diagnostics = [dict(window=i + 1, model=label, **validity_rates(np.asarray(best_examples[key])[i:i + 1]),
                                depth_ssim=1. if key == "real" else float(example_ssim[i]))
                           for i in range(3) for label, key in (("Real", "real"), ("TimeGAN", "generated"))]
    np.savez_compressed(output_dir / "validation_examples.npz", **{k: np.asarray(v) for k, v in best_examples.items()})
    from predict import save_timegan_report_figures
    figures = save_timegan_report_figures(best_examples, report_path.parent)
    report = dict(format_version=1, status="single-training-seed reference-structure experiment; not final model selection", config=config,
                  previous_runs=previous_runs, source_files=source_files, comparison_protocol=protocol, comparison_signature=signature,
                  metric_specification=spec, input_files=audit["files"], splits=audit["splits"], fit_split="train",
                  selection_split="validation", test_evaluated=False, validation_windows=len(starts), real_validity=validity_rates(real),
                  history=history, validation_history=validation_history, selected=best, figures=figures, example_diagnostics=example_diagnostics,
                  example_raw_anchor_indices=best_examples["raw_anchor_indices"],
                  ar_baselines=[row for row in ar["candidates"] if row["candidate"] in (ar["original_candidate"], ar["selected_candidate"])],
                  ar_selected_candidate=ar["selected_candidate"], reconstruction_check=sanity, reload_max_abs_difference=reload_difference,
                  selected_checkpoint_sha256=sha256_file(output_dir / "selected.pt"), data_loading_seconds=loading_seconds,
                  stage_training_seconds=stage_seconds, total_seconds=perf_counter() - started,
                  device=device, device_name=torch.cuda.get_device_name(0) if device == "cuda" else "CPU",
                  peak_allocated_bytes=torch.cuda.max_memory_allocated() if device == "cuda" else None,
                  peak_reserved_bytes=torch.cuda.max_memory_reserved() if device == "cuda" else None,
                  parameter_count=sum(p.numel() for p in model.parameters()),
                  parameters_by_network={name: sum(p.numel() for p in network.parameters()) for name, network in model.named_children()},
                  torch_version=str(torch.__version__), numpy_version=np.__version__,
                  limitations=["One training seed; generation-seed SD does not measure retraining variability.",
                               "Short budget compared with the released 50,000 iterations per stage. Validation selects checkpoints; test remains untouched.",
                               "Reference GRU equations, topology, losses and schedule; PyTorch Adam epsilon placement, RNG and kernels differ from TensorFlow 1.",
                               "Training-only scaler keeps the shared AR contract: span without the reference scaler's added 1e-7, constant spans mapped to one.",
                               "Sigmoid recovery bounds generated features to training ranges, limiting tail extrapolation. This is a model choice, not post-processing.",
                               "No price-ladder or spread constraints are built in; no clipping, sorting, filtering or repair is applied.",
                               "Allocator peaks include main training, evaluation and reload, exclude the separate sanity check and CUDA context/driver allocations.",
                               "Feature-generation timing includes noise creation, batched inference and CPU transfer, but excludes book decoding and metrics.",
                               "Checkpoints support generation; exact interrupted-training resume is not implemented."])
    if structure_weight:
        report["status"] = "Structural-loss extension; not the original TimeGAN objective. Validation only."
        report["limitations"][5] = "Training penalises all 19 price gaps using training anchors; this encourages validity but does not guarantee it. No inference repair."
    if balance_reconstruction:
        report["status"] = "Variance-balanced reconstruction extension; not the original TimeGAN objective. Validation only."
        report["limitations"].append("Price reconstruction uses the training-window variance ratio raised to balance_power; generator moment losses remain unchanged.")
    if no_supervision:
        report["status"] = "Temporal-supervision ablation of the released-code reference; validation only."
        report["limitations"].append(config["supervision_ablation"] + " This is not an exact reproduction of the paper's four-component ablation.")
    if context_scaler is not None:
        report["status"] = "Anchor-conditioned architecture extension" if anchor_conditioning == "matched" else "Zero-context ablation of the conditional architecture"
        report["limitations"].append("All five networks have static context inputs. Only matched mode supplies observed anchor gaps; zero mode retains identical dimensions with zero values. This is not the authors' objective/architecture.")
        report["anchor_context_scaler"] = context_scaler
    write_json(report_path, report)
    return report


def make_checkpoint(model, training, scaler, report):
    """One checkpoint schema for standalone fits and validation-selected fits."""
    starts = training.starts[training.starts >= model.lags]
    if len(starts) == 0:
        raise ValueError("No training windows have enough conditioning history.")
    contexts = {"features": conditioning_history(training, starts, model.lags),
                "initial_prices": torch.as_tensor(training.book[starts][:, PRICE_COLUMNS] / PRICE_DIVISOR),
                "raw_start_indices": torch.as_tensor(starts + training.raw_offset, dtype=torch.int64)}
    return {"format_version": 2, "model_type": "vector_ar", "state_dict": model.state_dict(),
            "scaler": {"minimum": torch.from_numpy(scaler.minimum), "maximum": torch.from_numpy(scaler.maximum)},
            "feature_names": FEATURE_NAMES, "contexts": contexts, "fit_report": report}


def fit_baseline(data_dir, checkpoint_path, report_path, sequence_length=32, ridge=1e-4, threads=4, lags=1):
    if threads < 1:
        raise ValueError("threads must be positive.")
    torch.set_num_threads(threads)
    started = perf_counter()
    datasets, scaler, audit = prepare_datasets(data_dir, sequence_length=sequence_length, stride=sequence_length)
    training = datasets["train"]
    features = torch.from_numpy(training.features)
    data_seconds = perf_counter() - started

    # Need p observed features inside this split, ending at features[start - 1].
    # That final feature describes book[start], whose prices anchor the rollout.
    starts = training.starts[training.starts >= lags]
    if len(starts) == 0:
        raise ValueError("Need at least two training windows to supply valid generation contexts.")
    started = perf_counter()
    model = VectorAR.fit(features, ridge=ridge, lags=lags)
    fit_seconds = perf_counter() - started
    radius = model.spectral_radius
    report = {
        "model": f"vector_ar{lags}", "lags": lags, "device": "cpu", "torch_version": str(torch.__version__), "threads": threads,
        "session": audit["session"], "input_files": audit["files"], "fit_split": "train",
        "training_raw_row_interval": audit["splits"]["train"]["raw_row_interval_half_open"],
        "training_feature_rows": len(features), "training_pairs": len(features) - lags, "features": features.shape[1],
        "sequence_length": sequence_length, "ridge": ridge, "coefficient_count": model.weight.numel() + model.bias.numel(),
        "residual_vectors": len(model.residuals), "training_contexts": len(starts),
        "training_one_step_mse_scaled": float(model.residuals.square().mean()),
        "spectral_radius": radius, "data_loading_seconds": data_seconds, "fit_seconds": fit_seconds,
        "solver": "CPU float64 centred SVD ridge solution; unpenalised intercept",
        "ridge_objective": "mean over training pairs of squared vector error + ridge * sum(weight ** 2)",
        "fit_scope": "All within-training lag histories and next-row targets, including between-window pairs and trailing rows.",
        "context_scope": "Training window starts with p previous features and matching same-state price anchors.",
        "limitations": [
            "Training MSE is an in-sample fit statistic, not validation or simulation quality.",
            "Joint empirical residuals retain simultaneous dependencies but are sampled independently over time.",
            "No price-ladder, spread or non-negative-volume constraints are enforced.",
            "A spectral radius below one describes linear mean dynamics, not validity of generated order books.",
            "Validation/test files are loaded by the existing audit, but no held-out features enter fitting or contexts.",
        ],
    }
    checkpoint = make_checkpoint(model, training, scaler, report)
    checkpoint_path, report_path = Path(checkpoint_path), Path(report_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("ar", "timegan"), default="ar")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_DIR / "outputs/ar/model.pt")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--lags", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--embedding-steps", type=int, default=1000)
    parser.add_argument("--supervised-steps", type=int, default=1000)
    parser.add_argument("--joint-steps", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--sanity-steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=24)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--structure-weight", type=float, default=0., help="Optional engineering extension; zero keeps the reference objective.")
    parser.add_argument("--balance-reconstruction", action="store_true", help="Labelled extension: weight price reconstruction using training-only feature variance.")
    parser.add_argument("--balance-power", type=float, default=1., help="Exponent in [0, 1] for partial variance balance; requires --balance-reconstruction.")
    parser.add_argument("--no-supervision", action="store_true", help="Ablation: omit supervisor pretraining and both joint temporal-loss terms.")
    parser.add_argument("--anchor-conditioning", choices=("none","matched","zero"), default="none")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.model == "timegan":
        if args.sequence_length != 32:
            parser.error("The fixed AR/TimeGAN comparison requires sequence-length=32.")
        report = fit_timegan(args.data_dir, args.output_dir, args.report, args.embedding_steps, args.supervised_steps,
                            args.joint_steps, args.eval_every, args.seed, args.batch_size, args.hidden, args.layers,
                            args.learning_rate, args.device, args.sanity_steps, args.threads, args.structure_weight, args.balance_reconstruction, args.balance_power, args.no_supervision, args.anchor_conditioning)
        print(f"TimeGAN pilot complete. Selected joint step {report['selected']['step']}; validation only.")
        return
    report_path = args.report or PROJECT_DIR / "outputs/ar/fit.json"
    report = fit_baseline(args.data_dir, args.checkpoint, report_path, args.sequence_length, args.ridge, args.threads, args.lags)
    keys = ("training_pairs", "training_contexts", "training_one_step_mse_scaled", "spectral_radius", "fit_seconds")
    print(json.dumps({key: report[key] for key in keys}, indent=2))
    print(f"Saved checkpoint: {args.checkpoint}\nSaved fit report: {report_path}")


if __name__ == "__main__":
    main()
