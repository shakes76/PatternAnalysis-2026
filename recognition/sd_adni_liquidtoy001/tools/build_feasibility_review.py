"""Build the one-page evidence-based discussion draft; tutor check-off is pending."""

import argparse
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/checkoff"),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "feasibility-review-2026-10-14.pdf"
    doc = SimpleDocTemplate(
        str(path),
        pagesize=(210 * mm, 297 * mm),
        leftMargin=17 * mm,
        rightMargin=17 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
    )
    body = ParagraphStyle(
        "body",
        fontName="Helvetica",
        fontSize=9.2,
        leading=12,
        spaceAfter=5,
        alignment=TA_LEFT,
    )
    heading = ParagraphStyle(
        "heading",
        parent=body,
        fontName="Helvetica-Bold",
        fontSize=10.2,
        leading=13,
        textColor=colors.HexColor("#135866"),
        spaceBefore=7,
    )
    title = ParagraphStyle(
        "title", parent=heading, fontSize=16, leading=19, spaceBefore=0
    )
    story = []

    def add(text, style=body):
        story.append(Paragraph(text, style))

    add("COMP3710 | Feasibility review", title)
    add(
        "ADNI synthetic MRI with Stable Diffusion fine-tuning | Project: sd_adni_liquidtoy001"
    )
    add(
        "<b>Discussion draft - 14 October 2026 (revises 7 and 12 October). "
        "Student review and tutor check-off pending.</b><br/>"
        "Experiments below have already run; this is a current review, not a retrospective "
        "claim of prior approval. All results use validation data; the test split is sealed."
    )
    add("1. User need, scope and acceptance criteria", heading)
    add(
        "A research engineer needs to know whether synthetic 2D brain MRI slices are realistic "
        "and diverse enough to share or augment data, without reproducing training patients. "
        "This prototype measures that trade-off; it does not establish clinical utility, "
        "anonymization or safe public release."
    )
    add(
        "<b>Five testable criteria:</b> (1) zero cross-split subject/scan/file-hash overlap; "
        "(2) finite, reproducible training on one A100 40GB within job limits, with measured "
        "memory and time; (3) original SD, LoRA SD (E0, E3), fully fine-tuned SD (E1, E2) and an "
        "implemented simple VAE compared under one SSIM protocol, reporting the course's 0.6 "
        "target honestly; (4) uncurated grids, feature distances, t-SNE, nearest-neighbor "
        "audits and 3-5 failure cases; (5) pre-registered acceptance criteria for every "
        "training run, committed before it starts."
    )
    add("2. Model choice and course concepts", heading)
    add(
        "ADNI: 30,520 slices, 680 subjects. Filename numbers are scan IDs: the original "
        "train/test folders share no scan but 216 subjects, so the project splits by subject "
        "(476 / 102 / 102). Stable Diffusion 1.5 latent diffusion (Hard) at 256 px: frozen "
        "VAE and text encoder, noise-prediction MSE, own training loop. E0 trained rank 4 "
        "attention LoRA. E1 fine-tunes the full UNet with 10% empty-prompt dropout, offset "
        "noise and EMA weights. E2 repeats E1 without two defective NC scans (40 slices) "
        "flagged by a foreground rule fixed in advance. Concepts: latent compression, forward "
        "noising, denoising, classifier-free guidance, fine-tuning capacity. Reference: "
        "Rombach et al., <i>High-Resolution Image Synthesis with Latent Diffusion Models</i> "
        "(CVPR 2022)."
    )
    add("3. Measured evidence", heading)
    add(
        "E1 and E2: 10 epochs in 45 min each on one A100, 93 images/s, 23.8 GiB peak; "
        "together about 1.9 A100-hours including probes and evaluation. Mean feature-nearest "
        "SSIM over 64 fixed seeds: original SD 0.087, E0 0.377, E1 0.688, <b>E2 0.705</b> "
        "(64/64 above 0.6), simple VAE 0.727 (blurrier), real validation to training 0.710. "
        "The grey background of E0 is gone (border intensity 0.059 to 0.00007)."
    )
    add(
        "<b>Memorization.</b> E1 produced 7/32 blank NC samples whose nearest training slices "
        "all came from one defective scan; removing two such scans (E2) gave 0/32, so the "
        "model had amplified a rare training item. The pre-registered pixel criterion (mean "
        "nearest-training RMSE at least 0.0731) is <b>not met</b> by E1 (0.0655) or E2 "
        "(0.0673). The five closest E2 pairs are not copies, and with size-matched references "
        "the samples are no closer to training subjects (0.092) than to validation subjects "
        "(0.090). Pixel distance cannot prove the absence of memorization."
    )
    add("4. Risks, budget, next steps and fallback", heading)
    add(
        "Risks: pixel and VAE-feature metrics favour smooth images (the blurry VAE scores "
        "highest SSIM); faint colour tints and dim slices remain; one defective-scan rule may "
        "miss subtler artefacts. One more run is queued: E3 trains E0's rank-4 adapter with "
        "E2's recipe, so E0 to E3 isolates the recipe and E3 to E2 the capacity (protocol "
        "committed first; about 1.2 A100-hours). Then: freeze a test protocol with a size-matched "
        "memorization comparison and a loss-based membership test, evaluate all models once on "
        "the sealed test split, profile inference latency, write the README and open the pull "
        "request before the 26 October deadline. Fallback: submit the validation results."
    )
    add("Discussion points / completion record", heading)
    add(
        "Confirm (a) whether the 2.6 clarification (libraries may load models; training loop "
        "and evaluation must be own code) also applies to 2.4; (b) that reporting a "
        "pre-registered criterion as unmet, with the supporting analysis, is acceptable; "
        "(c) the subject-level re-split; (d) whether real ADNI slices may appear in the public "
        "README. Student must verify and defend this draft; Codex and Claude Code assisted code, "
        "experiments and wording. Consultation date: __________  Tutor/record: __________"
    )
    add(
        "Sources: task sheet v2.2 sections 2.4 and 3; docs/experiments.md, docs/e1-protocol.md, "
        "docs/e2-protocol.md and versioned aggregate results. This PDF is not evidence of "
        "completed check-off; the review will be included in the README."
    )
    doc.build(story)
    print(path)


if __name__ == "__main__":
    main()
