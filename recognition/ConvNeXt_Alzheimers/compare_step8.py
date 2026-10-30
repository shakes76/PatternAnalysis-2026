"""Compare aggregate validation reports; bundle an explicit allowlist for sharing.

Does NOT load medical images, checkpoints or patient-linked predictions. It also
never edits the student's README or selects a model using the final test set.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import sys
from zipfile import ZipFile, ZIP_DEFLATED
from typing import Any

from dataset import require
from training_utils import read_json, selection_key, utc_now, write_json


def validate_reports(initial: dict[str,Any], baseline: dict[str,Any], convnext: dict[str,Any]) -> None:
    require(initial.get("baseline_training_completed") is True, "Initial baseline incomplete.")
    for name,r in (("baseline",baseline),("convnext",convnext)):
        require(r.get("experiment_completed") is True and r.get("model_name") == name,
                "A Step 8 model report is missing or incomplete.")
    for r in (initial,baseline,convnext):
        require(r.get("final_test_model_evaluation_performed") is False
                and r.get("final_test_images_decoded") == 0, "Final-test evidence must not enter development selection.")
        require(r.get("manifest_sha256") == initial.get("manifest_sha256")
                and r.get("preprocessing") == initial.get("preprocessing")
                and r.get("split_counts") == initial.get("split_counts"),
                "The comparison uses different partitions, preprocessing or counts.")
    bc=dict(baseline["configuration"]);cc=dict(convnext["configuration"])
    bc.pop("model",None);cc.pop("model",None)
    require(bc == cc, "Step 8 training recipes differ; do not describe them as matched.")
    require(baseline.get("source_code_sha256") == convnext.get("source_code_sha256"),
            "Step 8 source files changed between runs.")


def compare(initial_dir: Path, baseline_dir: Path, convnext_dir: Path,
            smoke_dir: Path, output: Path) -> dict[str,Any]:
    initial=read_json(initial_dir/"baseline_summary.json")
    baseline=read_json(baseline_dir/"experiment_summary.json")
    convnext=read_json(convnext_dir/"experiment_summary.json")
    smoke=read_json(smoke_dir/"step8_smoke_summary.json")
    validate_reports(initial,baseline,convnext)
    require(smoke.get("step8_smoke_passed") is True
            and smoke.get("source_code_sha256") == baseline.get("source_code_sha256")
            and smoke.get("manifest_sha256") == baseline.get("manifest_sha256"), "Smoke report mismatch.")
    output=output.resolve()
    require(not output.exists(), "Choose a fresh comparison output directory.")
    output.mkdir(parents=True)
    names=("Initial CNN", "Regularized CNN", "ConvNeXt-Tiny")
    reports=(initial,baseline,convnext)
    preferred_baseline = ("Regularized CNN" if selection_key(baseline["selected_validation_metrics"]) >
                          selection_key(initial["selected_validation_metrics"]) else "Initial CNN")
    summary={"created_utc":utc_now(),"scope":"selected validation results only; final test untouched",
             "all_three_runs_preserved":True,"preferred_cnn_by_prespecified_validation_rule":preferred_baseline,
             "matched_step8_configuration_verified":True,"manifest_sha256":initial["manifest_sha256"],
             "models":{},"caution":"Two CNN configurations and one ConvNeXt configuration were examined. "
                "These development comparisons do not establish statistical superiority or clinical safety. "
                "Same training recipe is not proof each architecture is optimally tuned."}
    lines=["## Step 8: validation comparison", "",
        "The original CNN experiment was retained. One more strongly regularized CNN and one "
        "ConvNeXt-Tiny were trained from scratch using a shared, predeclared Step 8 recipe. "
        "The data splits, image preprocessing, validation aggregation, selection criterion and 0.5 "
        "decision threshold were unchanged. Final-test images were not evaluated.", "",
        "| Model | Patient accuracy | Macro-F1 | AUROC | AD recall | NC recall | Selected epoch |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for name,r in zip(names,reports):
        m=r["selected_validation_metrics"]["patient"]
        summary["models"][name]={"selected_epoch":r["selected_epoch"],"epochs_completed":r["epochs_completed"],
            "patient_metrics":m,"parameters":r["parameters"],"training_loop_seconds":r["training_loop_seconds"],
            "gpu_memory":r["training_loop_gpu_memory"],"inference":r["model_only_inference"],
            "stop_reason":r["stopping_reason"]}
        lines.append(f"| {name} | {m['accuracy']:.2%} | {m['macro_f1']:.4f} | {m['auroc']:.4f} | "
            f"{m['per_class']['AD']['recall']:.2%} | {m['per_class']['NC']['recall']:.2%} | {r['selected_epoch']} |")
    lines += ["",f"Preferred CNN under the predefined validation selection rule: **{preferred_baseline}**. "
        "Retain both CNN runs and report this selection; do not select a weaker baseline to favour ConvNeXt.","",
        "The initial-to-regularized CNN comparison changes multiple training settings at once; it does "
        "not isolate weight decay. The Step 8 CNN and ConvNeXt have matched optimizer settings and "
        "batch configuration, but their architectures and intrinsic regularizers differ. One seed, "
        "repeated validation use and a small validation cohort limit generalisation claims. Check stop "
        "reasons for any time-limited runs. No result here demonstrates attainment of the final-test "
        "accuracy target. Confidence calibration, referral analysis and the final test remain outstanding.","",
        "### Implementation and verification", "",
        "ConvNeXt-Tiny uses stage depths (3,3,9,3), widths (96,192,384,768), a one-channel "
        "stem, depthwise 7x7 convolutions, channel LayerNorm, 4x inverted bottlenecks, GELU, "
        "LayerScale and stochastic depth. It was implemented in PyTorch without downloading "
        "pretrained weights. Reference: Liu et al., *A ConvNet for the 2020s*, CVPR 2022, "
        "https://arxiv.org/abs/2201.03545 (assessment reference [7]).", "",
        "ChatGPT assisted with code, tests and this draft. Record the checks and code review actually "
        "performed in the report's Artificial Intelligence Usage Disclosure.", ""]
    for slug,folder in (("baseline",baseline_dir),("convnext",convnext_dir)):
        for kind in ("loss","validation"):
            dst=f"step8_{slug}_{kind}.png"
            shutil.copy2(folder/f"{kind}_curve.png",output/dst)
            lines.append(f"![{slug} {kind}](figures/{dst})\n")
    write_json(output/"comparison_summary.json",summary)
    (output/"README_step8_comparison.md").write_text("\n".join(lines),encoding="utf-8")
    with ZipFile(output/"COMP3710_Step8_Results.zip","w",ZIP_DEFLATED) as z:
        z.write(output/"comparison_summary.json","comparison_summary.json")
        z.write(smoke_dir/"step8_smoke_summary.json","step8_smoke_summary.json")
        z.write(initial_dir/"baseline_summary.json","initial_baseline/baseline_summary.json")
        z.write(initial_dir/"history.csv","initial_baseline/history.csv")
        for name,folder in (("regularized_baseline",baseline_dir),("convnext",convnext_dir)):
            for filename in ("experiment_summary.json","history.csv"):
                z.write(folder/filename,f"{name}/{filename}")
    print("STEP8_COMPARISON_READY")
    print(f"Preferred CNN by validation rule: {preferred_baseline}")
    print(f"Share only: {output / 'COMP3710_Step8_Results.zip'}")
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("initial-dir","baseline-dir","convnext-dir","smoke-dir","output-dir"):
        p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args()
    try: compare(a.initial_dir,a.baseline_dir,a.convnext_dir,a.smoke_dir,a.output_dir)
    except Exception as e:
        print(f"COMPARISON_STOPPED: {e}",file=sys.stderr);return 1
    return 0

if __name__ == '__main__':raise SystemExit(main())
