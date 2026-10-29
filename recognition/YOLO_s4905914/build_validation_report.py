"""Build the plain validation comparison PDF from the audited result artifacts."""
from pathlib import Path
import argparse
import json
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak


def build(root):
    evidence = root / "outputs/validation_comparison_20261017"
    data = json.loads((evidence / "comparison.json").read_text())
    assert data["status"] == "passed"
    n, s = (data["models"][key] for key in ("n", "s"))
    ns = (n, s)
    output = root / "output/pdf/COMP3710_Validation_Comparison.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    width = A4[0] - 84
    styles = {
        "title": ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=19, leading=23, spaceAfter=12),
        "heading": ParagraphStyle("heading", fontName="Helvetica-Bold", fontSize=13, leading=16, spaceAfter=9),
        "body": ParagraphStyle("body", fontName="Helvetica", fontSize=9.5, leading=13, spaceAfter=8),
        "small": ParagraphStyle("small", fontName="Helvetica", fontSize=8.4, leading=11, spaceAfter=7),
        "cell": ParagraphStyle("cell", fontName="Helvetica", fontSize=9, leading=11),
    }
    story = []
    def text(value, style="body"):
        story.append(Paragraph(value, styles[style]))
    def table(rows, widths=None):
        formatted = [[Paragraph(escape(str(v)), styles["cell"]) for v in row] for row in rows]
        obj = Table(formatted, colWidths=widths or [width*.56, width*.22, width*.22], hAlign="LEFT")
        obj.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#eeeeee")),
            ("LINEBELOW", (0,0), (-1,0), .7, colors.black),
            ("LINEBELOW", (0,-1), (-1,-1), .5, colors.black),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
            ("LEFTPADDING", (0,0), (-1,-1), 6), ("RIGHTPADDING", (0,0), (-1,-1), 6),
            ("TOPPADDING", (0,0), (-1,-1), 5), ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ]))
        story.extend([obj, Spacer(1, 10)])
    def figure(name, ratio):
        story.extend([Image(str(evidence / name), width=width, height=width*ratio), Spacer(1, 7)])
    def pct(value): return f"{100*value:.2f}%"
    def op(model): return model["operating_point"]
    def malignant(model): return op(model)["classes"]["Malignant"]

    text("YOLOv8n and YOLOv8s\nvalidation comparison".replace("\n", "<br/>"), "title")
    text("COMP3710 Project 3 | Restricted known-lesion study | 17 October 2026", "small")
    text("Both 60-epoch runs completed. YOLOv8n has higher macro mAP50:95 and higher malignant detection recall under the declared false-alert budget. YOLOv8s has higher malignant AP50. These are different advantages; the results do not establish an overall or statistically reliable winner.")
    text("The same 165 validation images contain 124 benign and 41 malignant lesions. Training uses 768 images. Both models start from their own original COCO weights, use Ultralytics 8.3.221 and the same seed, optimizer, augmentation and update budget. YOLOv8n remains the primary model under the policy written before training.")
    rows = [["Validation metric", "YOLOv8n", "YOLOv8s"]]
    for label, function in [
        ("Macro mAP50", lambda m: f'{m["mAP50"]:.4f}'),
        ("Macro mAP50:95", lambda m: f'{m["mAP50_95"]:.4f}'),
        ("Benign AP50", lambda m: f'{m["ap"]["0"]["AP50"]:.4f}'),
        ("Malignant AP50", lambda m: f'{m["ap"]["1"]["AP50"]:.4f}'),
        ("Malignant AP50:95", lambda m: f'{m["ap"]["1"]["AP50_95"]:.4f}'),
        ("Selected confidence threshold", lambda m: f'{op(m)["threshold"]:.2f}'),
        ("Malignant TP / FP / FN", lambda m: f'{malignant(m)["tp"]} / {malignant(m)["fp"]} / {malignant(m)["fn"]}'),
        ("Malignant detection precision", lambda m: pct(malignant(m)["precision"])),
        ("Malignant detection recall", lambda m: f'{malignant(m)["tp"]}/41 ({pct(malignant(m)["recall"])})'),
        ("Benign images with malignant false alerts", lambda m: f'{op(m)["image_alerts"]["benign_fp"]}/124 ({pct(op(m)["image_alerts"]["benign_false_alert_rate"])})'),
        ("Malignant images with any malignant alert", lambda m: f'{op(m)["image_alerts"]["malignant_tp"]}/41'),
        ("Benign detection precision / recall", lambda m: f'{pct(op(m)["classes"]["Benign"]["precision"])} / {pct(op(m)["classes"]["Benign"]["recall"])}'),
    ]:
        rows.append([label, *[function(m) for m in ns]])
    table(rows)
    text("AP sweeps scores from the 0.001 floor; it is not AP at the selected operating threshold. Detection TP requires the correct class and IoU &gt;= 0.5, with at most one matched prediction per GT. An image alert requires any malignant prediction, without an IoU condition. Detection FP counts and benign-image false-alert counts therefore differ.", "small")
    text("The 10% false-alert budget permits at most 12 of the 124 benign validation images. It is a predeclared classroom comparison rule, not a clinically validated acceptable rate. Confidence is not a probability of cancer.", "small")

    story.append(PageBreak())
    text("Threshold policy and training history", "heading")
    text("For each model, select the highest malignant detection recall among the same 14 predefined thresholds satisfying the benign-image false-alert budget. Break ties by precision, fewer benign alerts, then higher threshold. Filtering is strictly confidence &gt; threshold. The resulting thresholds are n=0.50 and s=0.80; the grid was not refined after inspecting these results.")
    figure("threshold_tradeoff.png", 3.2/9)
    text("Markers show evaluated thresholds. Connecting lines only guide the eye; intermediate thresholds were not tested for selection. The right panel enlarges the same observations.", "small")
    table([["Working point", "Malignant TP / 41", "Benign alerts / 124"],
           ["n at 0.50 (selected)", "13", "11 (within budget)"],
           ["s at 0.50 (same numerical threshold)", "22", "28 (over budget)"],
           ["s at 0.70", "14", "14 (over budget)"],
           ["s at 0.80 (selected)", "11", "7 (within budget)"]])
    text("At the same numerical threshold, s detects more malignant lesions but generates more benign false alerts. Under the same declared constraint, n detects two more. The selected-point difference is 4.88 percentage points on only 41 malignant images. A single seed and this sample size do not establish a general superiority claim.")
    figure("training_curves.png", 2.7/9)
    text("Validation classification loss uses a logarithmic axis to retain its early spikes and later detail; their cause has not been isolated. All logged scalar values are finite. Both runs completed 60 epochs and 2,880 optimizer.step calls. The checkpoint rule selected n at epoch 32 and s at epoch 42. An earlier best checkpoint does not alone prove overfitting, and 60 epochs does not prove convergence.", "small")

    story.append(PageBreak())
    text("What remains difficult", "heading")
    paired = data["paired_malignant_TP"]
    text(f'At their selected operating points, both models correctly detect {paired["both"]} malignant images, only n detects {paired["n_only"]}, only s detects {paired["s_only"]}, and neither detects {paired["neither"]}. The models do not succeed on an identical subset.')
    table([["Partition of malignant FN", "n: 28 FN", "s: 30 FN"],
           ["A qualified malignant candidate exists in the cache but falls below the operating threshold", "19", "21"],
           ["No qualified malignant candidate; a benign candidate reaches IoU >= 0.5", "7", "8"],
           ["No cached candidate of either class reaches IoU >= 0.5", "2", "1"]])
    text("The rows are mutually exclusive in the order shown. Candidates were already processed by single-label, class-aware NMS (IoU 0.7) and a score floor of 0.001. This describes stored predictions; it does not identify whether imbalance, training, NMS, label ambiguity or image artefacts caused the errors. Lower thresholds recover some candidates while increasing false alerts; they do not remove the trade-off.", "small")
    figure("validation_examples.png", 5.7/8)
    text("Dashed outline: malignant GT. Solid outline: prediction retained at that model's operating threshold. B/M labels indicate predicted Benign/Malignant. These are validation examples selected by the lowest image ID satisfying each stated category; they are not a representative error-rate estimate.", "small")
    text("Top: ISIC_0013839 is missed by both. n retains a well-located benign box; s retains no box at 0.80. At the cache floor, both have a localized benign candidate but no qualified malignant candidate. Bottom: ISIC_0013072 triggers alerts in both models. n's malignant box (IoU 0.422) fails. s retains two malignant boxes: one matches at IoU 0.539; its higher-score box has IoU 0.368 and is an extra FP.", "small")

    story.append(PageBreak())
    text("Evidence, limits and remaining work", "heading")
    table([["Recorded resource or localization measure", "YOLOv8n", "YOLOv8s"],
           ["Parameters", *[f'{m["parameters"]:,}' for m in ns]],
           ["Training-call elapsed time", *[f'{m["training_seconds"]/60:.2f} min' for m in ns]],
           ["Peak PyTorch CUDA allocation", *[f'{m["peak_cuda_allocated_GiB"]:.2f} GiB' for m in ns]],
           ["Malignant mean IoU: all 41 GT, correct class", *[f'{malignant(m)["iou"]["all_gt_class_correct"]["mean"]:.4f}' for m in ns]],
           ["Malignant mean IoU: all 41 GT, ignore class", *[f'{malignant(m)["iou"]["all_gt_class_agnostic"]["mean"]:.4f}' for m in ns]],
           ["Malignant mean IoU: matched TP only", *[f'{malignant(m)["iou"]["tp_only"]["mean"]:.4f} (n={malignant(m)["tp"]})' for m in ns]]])
    text("Resources were measured on the school A100 host. Training-call time is not inference latency; peak PyTorch allocation is not total device memory. IoU uses predictions retained at each selected threshold. Missing class-correct predictions contribute zero to the all-GT mean. High TP-only IoU cannot establish overall malignant detection success or compliance with the course's IoU target.", "small")
    text("The audit verified the locked configuration, all three manifest hashes, validation image IDs and GT boxes, original weight hashes, saved checkpoint hashes, actual training arguments, 60-epoch histories and 2,880-step traces. An independent NumPy matcher reproduced all 28 grid points, image-alert counts, IoU means and selected thresholds. AP remains the output of the pinned Ultralytics validator; it was not independently recomputed.")
    text("All recorded training/evaluation source files match the local copies except transfer.py, a packaging utility outside the training/evaluation import path. Both runs used the same recorded sources and image-content hash. The exact saved-checkpoint validation scores are reported here; training-time selection scores (n=0.40685, s=0.39275) are a different measurement. EMA serialization and differing validation batch sizes are among execution differences, not isolated causal explanations.", "small")
    text("Limitations", "heading")
    text("The 1,098-image cohort has recorded lesion IDs and no overlap of recorded lesion/patient links across splits. It still contains 708 images without patient_id, so complete patient independence is unproven. Restricting the cohort changes its composition and limits comparison with the old all-image experiment. The teacher has not yet confirmed acceptance of this restricted-cohort design.")
    text("Validation selected both the checkpoint and threshold; these are development results. The old test was previously inspected. Regrouping cannot make the supplementary reassessment a fresh blind test, and it has not been model-evaluated in this run. No retrospective model or threshold selection from that set is permitted.")
    text("Next steps", "heading")
    text("Retain n as the predeclared primary and s as its comparator. Preserve these validation locks. Prepare the final report around the observed trade-offs, then evaluate the locked supplementary set once under its declared retrospective role. Keep course acceptance, the required model-component packaging, the IoU target denominator, personal check-off and deadline as unresolved course items. No additional hyperparameter search is part of this comparison.")
    text("Evidence: outputs/provisional_20261017_669143/ (completed school job 669143); outputs/validation_comparison_20261017/ (audit, full grid and paired cases). Source: recognition/YOLO_s4905914/compare_validation.py. Fixed configuration: experiment_provisional.json. AI assistance: OpenAI ChatGPT/Codex assisted with analysis and this draft; the student must review it. YOLO, losses, training and AP code are Ultralytics implementations, not student-authored network components.", "small")
    SimpleDocTemplate(str(output), pagesize=A4, leftMargin=42, rightMargin=42,
                      topMargin=38, bottomMargin=38, title="COMP3710 validation comparison",
                      author="COMP3710 Project 3 working draft").build(story)
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    build(parser.parse_args().project_root.resolve())
