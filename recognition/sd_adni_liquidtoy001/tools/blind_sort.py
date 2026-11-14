"""Blinded real/synthetic sort by the student, a non-expert plausibility check
(local; docs/final-evaluation-protocol.md).

extract (on Rangpur): 16 AD and 16 NC slices drawn at random from the whole
test split (seed 3710), so the real images cover every slice position, as the
samples do; saved privately without identifiers.

    python tools/blind_sort.py extract --output outputs/blind-real.npz

reference: before sorting, a page of real slices from the training split only
(none can appear in the sort) with their class, so a non-expert knows what
real slices look like. No synthetic examples are shown.

    python tools/blind_sort.py extract --split train --per-class 8 \
        --output outputs/reference-real.npz
    python tools/blind_sort.py reference --real outputs/reference-real.npz \
        --output outputs/blind-sort/private-reference.html

build: those 32 slices plus 16 AD and 16 NC samples of the recommended model,
all shown in grayscale at 256 px without names, shuffled with seed 3710, in a
private HTML page; the key goes to a separate private file.

    python tools/blind_sort.py build --real outputs/blind-real.npz \
        --cohort outputs/final-e2 --output outputs/blind-sort

score: compares the answers saved from the page with the key.

    python tools/blind_sort.py score --key outputs/blind-sort/private-key.json \
        --answers answers.json
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

PER_CLASS = 16
SEED = 3710


def png_data(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def page(items: list[str]) -> str:
    cells = "\n".join(
        f'<figure><img src="data:image/png;base64,{data}" alt="image {i + 1}">'
        f"<figcaption>{i + 1}"
        f'<label><input type="radio" name="q{i}" value="real"> real</label>'
        f'<label><input type="radio" name="q{i}" value="synthetic"> synthetic</label>'
        "</figcaption></figure>"
        for i, data in enumerate(items)
    )
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Blinded sort</title><style>
body{{font-family:system-ui,sans-serif;margin:16px;background:#fff;color:#111}}
main{{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px}}
figure{{margin:0;border:1px solid #ccc;padding:6px}} img{{width:100%;image-rendering:auto}}
figcaption{{display:flex;gap:10px;align-items:center;font-size:14px}}
button{{margin:16px 0;padding:8px 16px;font-size:15px}}
</style></head><body>
<h1>Real or synthetic?</h1>
<p>{len(items)} brain MRI slices. Mark each one, then press Save. About half are real.</p>
<main>{cells}</main>
<button id="save">Save answers</button> <span id="status"></span>
<script>
document.getElementById('save').onclick = () => {{
  const answers = [];
  for (let i = 0; i < {len(items)}; i++) {{
    const chosen = document.querySelector('input[name="q' + i + '"]:checked');
    answers.push(chosen ? chosen.value : null);
  }}
  const missing = answers.filter(a => a === null).length;
  const blob = new Blob([JSON.stringify(answers)], {{type: 'application/json'}});
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob); link.download = 'answers.json'; link.click();
  document.getElementById('status').textContent =
    missing ? missing + ' unanswered; saved anyway' : 'saved answers.json';
}};
</script></body></html>
"""


def extract(args):
    """Random slices of one split (test for the sort); run where the data is."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from dataset import ADNISliceDataset
    from final_common import ADNI_ROOT, MANIFEST, balanced_sample

    data = ADNISliceDataset(ADNI_ROOT, MANIFEST, args.split, 256, verify_hash=True)
    chosen = balanced_sample([r["class_name"] for r in data.rows], args.per_class, SEED)
    images = [
        ((data[i]["pixel_values"][0] + 1) * 127.5).round().byte().numpy()
        for i in chosen
    ]
    is_nc = [data.rows[i]["class_name"] == "NC" for i in chosen]
    np.savez(args.output, images=np.stack(images), is_nc=np.array(is_nc, dtype=np.int8))
    print(f"saved {len(images)} {args.split} slices to", args.output)


def reference(args):
    """A labelled page of real training slices to look at before the sort."""
    real = np.load(args.real)
    sections = []
    for label, flag in (("AD", 0), ("NC", 1)):
        cells = "\n".join(
            f'<figure><img src="data:image/png;base64,{png_data(Image.fromarray(image))}"'
            f' alt="real {label} slice"><figcaption>real, {label}</figcaption></figure>'
            for image, is_nc in zip(real["images"], real["is_nc"])
            if is_nc == flag
        )
        sections.append(f"<h2>{label}</h2><main>{cells}</main>")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Real reference slices</title><style>
body{{font-family:system-ui,sans-serif;margin:16px;background:#fff;color:#111}}
main{{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px}}
figure{{margin:0;border:1px solid #ccc;padding:6px}} img{{width:100%}}
figcaption{{font-size:14px}}
</style></head><body>
<h1>Real slices for reference</h1>
<p>{len(real["images"])} real slices drawn at random from the training split. None of
them appear in the sort. Look at them, close this page, then open the sort.</p>
{"".join(sections)}
</body></html>
"""
    )
    print(f"wrote {len(real['images'])} reference slices to", args.output)


def build(args):
    args.output.mkdir(parents=True, exist_ok=False)
    random = np.random.default_rng(SEED)
    real = np.load(args.real)
    items = [
        {
            "truth": "real",
            "label": "NC" if is_nc else "AD",
            "image": Image.fromarray(image),
        }
        for image, is_nc in zip(real["images"], real["is_nc"])
    ]
    records = json.loads((args.cohort / "report.json").read_text())["records"]
    for label in ("AD", "NC"):
        pool = [r for r in records if r["label"] == label]
        for i in random.choice(len(pool), PER_CLASS, replace=False):
            with Image.open(args.cohort / pool[i]["file"]) as source:
                image = source.convert("L").resize(
                    (256, 256), Image.Resampling.BILINEAR
                )
            items.append(
                {
                    "truth": "synthetic",
                    "label": label,
                    "image": image,
                    "file": pool[i]["file"],
                }
            )
    order = random.permutation(len(items))
    shuffled = [items[i] for i in order]
    (args.output / "private-blind-sort.html").write_text(
        page([png_data(x["image"]) for x in shuffled])
    )
    key = [{k: v for k, v in x.items() if k != "image"} for x in shuffled]
    (args.output / "private-key.json").write_text(json.dumps(key, indent=2) + "\n")
    print(f"wrote {len(key)} items to", args.output / "private-blind-sort.html")


def score(args):
    from scipy import stats

    key = json.loads(args.key.read_text())
    answers = json.loads(args.answers.read_text())
    if len(answers) != len(key):
        raise SystemExit(f"{len(answers)} answers for {len(key)} items")
    answered = [(k, a) for k, a in zip(key, answers) if a is not None]
    correct = sum(k["truth"] == a for k, a in answered)
    result = {
        "items": len(key),
        "answered": len(answered),
        "correct": correct,
        "accuracy": correct / len(answered),
        "binomial_p_two_sided": stats.binomtest(correct, len(answered), 0.5).pvalue,
        "by_truth": {
            truth: {
                "n": sum(k["truth"] == truth for k, _ in answered),
                "called_real": sum(
                    k["truth"] == truth and a == "real" for k, a in answered
                ),
            }
            for truth in ("real", "synthetic")
        },
    }
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    e = commands.add_parser("extract")
    e.add_argument("--output", type=Path, required=True)
    e.add_argument("--split", choices=("train", "test"), default="test")
    e.add_argument("--per-class", type=int, default=PER_CLASS)
    r = commands.add_parser("reference")
    r.add_argument("--real", type=Path, required=True)
    r.add_argument("--output", type=Path, required=True)
    b = commands.add_parser("build")
    b.add_argument("--real", type=Path, required=True)
    b.add_argument("--cohort", type=Path, required=True)
    b.add_argument("--output", type=Path, required=True)
    s = commands.add_parser("score")
    s.add_argument("--key", type=Path, required=True)
    s.add_argument("--answers", type=Path, required=True)
    s.add_argument("--output", type=Path)
    args = parser.parse_args()
    handlers = {"extract": extract, "reference": reference, "build": build, "score": score}
    handlers[args.command](args)


if __name__ == "__main__":
    main()
