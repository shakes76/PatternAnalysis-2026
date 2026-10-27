"""
autopsy.py - evidence for the clinical safety autopsy: pre-registered case selection, keyword
checks for changed negation/hedging, data artefacts in the outputs, and reading grade level.

Author : Trung Bao Truong (Brian Truong), student ID 49846627 (s4984662)
Course : COMP3710 Pattern Recognition, UQ, Semester 2 2026 - Project 2.6
         (FLAN-T5 lay summarisation of radiology reports, LoRA vs full fine-tuning)

Description
    ROUGE cannot tell a safe summary from an unsafe one (spec 2.6), so a few cases are read
    by a person. To keep that from becoming cherry-picking, the rule that picks the cases is
    fixed in this file and uses a seeded random draw. The script reads the per-example
    predictions that train.py wrote (`val_predictions.jsonl`), so it needs no GPU and no model.

    1. Case selection. Pool = validation rows of the `clean` tier (report never seen in train
       and a different study), minus BIMCV rows whose session repeats (mismatched pairs),
       minus the rows already shown by predict.py, minus reports shorter than 8 words. One
       case is drawn from each of five strata:
         A  full-FT ROUGE-Lsum >= 0.70 but negation or hedging differs between source and output
         B  the source mentions an acute finding and the output is much shorter than the source
         C  the lowest 5% of full-FT ROUGE-Lsum (failure tail)
         D  the LoRA output copies the source (>= 0.9 of its words are in the source) while the
            reference does not (<= 0.5)
         E  random (control)
       If a stratum is empty its thresholds are relaxed once (0.70 -> 0.60, 0.6 -> 0.8,
       0.9 -> 0.8).
    2. Flag rates. For every output, did a negation or hedge keyword that is in the source
       disappear (or appear)? This is a crude keyword check: a paraphrase ("without" ->
       "does not show") also changes the flag, so a flag is a candidate, not an error. The
       reference summaries are scored the same way as a yardstick.
    3. Artefacts. Reference/outputs with Chinese characters, or with a second summary glued on
       after a '": "' (a JSON-like leak in some GPT-4o references).
    4. Reading grade. Flesch-Kincaid grade level of source, reference and outputs, with a
       simple syllable heuristic (approximate; the spec targets about grade 6).

Usage
    python autopsy.py --runs_dir runs --out_dir autopsy
    python autopsy.py --self_check
    The run folders are the ones slurm/submit_final.sh creates (RUN_DIRS below).
    Output: <out_dir>/cases.md, cases.json (the five cases with all three outputs) and
    summary.json (flag rates, artefacts, reading grade).

References
    Flesch, R. (1948) and Kincaid et al. (1975), reading-grade formula.
    Selection by strata with a fixed seed, so the cases can be regenerated exactly.

AI assistance: drafted with Claude (Anthropic) and reviewed by the author, see the README
section "Artificial Intelligence Usage Disclosure".
"""

import argparse
import json
import os
import random
import re
from typing import Callable, Dict, List, Optional, Sequence

from dataset import (SOURCE_COLUMN, TARGET_COLUMN, evaluation_tiers, load_laysumm,
                     session_duplicate_rows)
from utils import split_sentences

# Run folders written by slurm/submit_final.sh.
RUN_DIRS = {"zeroshot": "zeroshot", "lora_r8": "lora_r8_lr3e-3",
            "lora_r16": "lora_r16_lr3e-3", "full_ft": "full_lr3e-4"}
# Validation rows that predict.py draws by default (seed 0), left out of the pool.
DEMO_ROWS = (6412, 6998, 673, 4307)

NEGATION = ["no", "not", "without", "absent", "negative", "neither", "nor", "free of", "n't"]
HEDGE = ["possible", "possibly", "probable", "probably", "may", "might", "could", "likely",
         "suspected", "suspicious", "suggest", "suggests", "suggestive",
         "cannot be excluded", "cannot be ruled out", "can't be ruled out", "not excluded"]
ACUTE = ["pneumothorax", "effusion", "pneumonia", "consolidation", "edema", "oedema",
         "embolism", "fracture", "nodule", "nodules", "mass", "masses", "tumor", "tumour",
         "carcinoma", "malignant", "malignancy", "infiltrate", "infiltrates", "hemorrhage",
         "abscess"]
JUNK_TAIL = '": "'
CHINESE = re.compile(r"[一-鿿]")


def has_term(terms: Sequence[str], text: str) -> bool:
    """True if any term occurs in `text` as a whole word or phrase (case-insensitive).

    "n't" is matched inside words ("didn't", "isn't"), the other terms only as whole words.
    """
    lowered = text.lower()
    for term in terms:
        pattern = re.escape(term) if term == "n't" else r"(?<!\w)" + re.escape(term) + r"(?!\w)"
        if re.search(pattern, lowered):
            return True
    return False


def word_tokens(text: str) -> List[str]:
    """Lower-case word tokens (letters, digits, apostrophes)."""
    return re.findall(r"[a-z0-9']+", text.lower())


def copy_ratio(output: str, source: str) -> float:
    """Fraction of the words of `output` that also occur in `source`."""
    vocabulary = set(word_tokens(source))
    words = word_tokens(output)
    return sum(w in vocabulary for w in words) / max(len(words), 1)


def count_syllables(word: str) -> int:
    """Rough syllable count: groups of vowels, minus a silent final 'e', at least one."""
    word = word.lower()
    groups = re.findall(r"[aeiouy]+", word)
    count = len(groups)
    if word.endswith("e") and not word.endswith(("le", "ee")) and count > 1:
        count -= 1
    return max(count, 1)


def reading_grade(text: str) -> float:
    """Flesch-Kincaid grade level: 0.39 * words/sentences + 11.8 * syllables/words - 15.59."""
    words = word_tokens(text)
    if not words:
        return 0.0
    sentences = max(len(split_sentences(text).split("\n")), 1)
    syllables = sum(count_syllables(w) for w in words)
    return 0.39 * len(words) / sentences + 11.8 * syllables / len(words) - 15.59


def load_predictions(directory: str) -> Dict[int, Dict[str, object]]:
    """Per-example predictions of one run, keyed by validation row index."""
    path = os.path.join(directory, "val_predictions.jsonl")
    with open(path) as handle:
        rows = [json.loads(line) for line in handle]
    return {row["index"]: row for row in rows}


def build_pool(splits, exclude: Sequence[int]) -> List[int]:
    """Validation rows eligible for the autopsy (see the module docstring), sorted."""
    clean = set(evaluation_tiers(splits["train"], splits["validation"])["clean"])
    duplicated = set(session_duplicate_rows(splits)["validation"])
    sources = list(splits["validation"][SOURCE_COLUMN])
    return sorted(i for i in clean
                  if i not in duplicated and i not in set(exclude) and len(sources[i].split()) >= 8)


def select_cases(pool: Sequence[int], sources: Sequence[str], references: Sequence[str],
                 preds: Dict[str, Dict[int, Dict[str, object]]], seed: int = 0
                 ) -> Dict[str, int]:
    """Draw one case per stratum A-E (see the module docstring); returns {stratum: row}."""
    full, lora = preds["full_ft"], preds["lora_r16"]
    score = lambda i: full[i]["rougeLsum"]                              # noqa: E731
    cut = sorted(score(i) for i in pool)[int(0.05 * (len(pool) - 1))]

    def flags_differ(i: int) -> bool:
        output = full[i]["prediction"]
        return (has_term(NEGATION, sources[i]) != has_term(NEGATION, output)
                or has_term(HEDGE, sources[i]) != has_term(HEDGE, output))

    def a_rule(i: int, low: float = 0.70) -> bool:
        return score(i) >= low and flags_differ(i)

    def b_rule(i: int, ratio: float = 0.6) -> bool:
        shorter = len(full[i]["prediction"].split()) / len(sources[i].split()) < ratio
        return has_term(ACUTE, sources[i]) and shorter

    def d_rule(i: int, high: float = 0.9) -> bool:
        return (copy_ratio(lora[i]["prediction"], sources[i]) >= high
                and copy_ratio(references[i], sources[i]) <= 0.5)

    rules: Dict[str, Callable[[int], bool]] = {
        "A": a_rule, "B": b_rule, "C": lambda i: score(i) <= cut, "D": d_rule,
        "E": lambda i: True}
    relaxed: Dict[str, Callable[[int], bool]] = {
        "A": lambda i: a_rule(i, 0.60), "B": lambda i: b_rule(i, 0.8),
        "D": lambda i: d_rule(i, 0.8)}
    rng, chosen = random.Random(seed), {}
    for stratum, rule in rules.items():
        eligible = [i for i in pool if rule(i) and i not in chosen.values()]
        if not eligible and stratum in relaxed:
            eligible = [i for i in pool if relaxed[stratum](i) and i not in chosen.values()]
            print(f"stratum {stratum}: thresholds relaxed once")
        print(f"stratum {stratum}: {len(eligible)} eligible")
        chosen[stratum] = rng.choice(eligible)
    return chosen


def flag_rates(sources: Sequence[str], outputs: Sequence[str], rows: Sequence[int]
               ) -> Dict[str, float]:
    """Percent of `rows` whose negation / hedge keyword flag is lost or gained."""
    counts = dict(negation_lost=0, negation_gained=0, hedge_lost=0, hedge_gained=0)
    for i in rows:
        for name, terms in (("negation", NEGATION), ("hedge", HEDGE)):
            before, after = has_term(terms, sources[i]), has_term(terms, outputs[i])
            counts[f"{name}_lost"] += before and not after
            counts[f"{name}_gained"] += after and not before
    return {key: 100.0 * value / len(rows) for key, value in counts.items()}


def artefact_counts(texts: Sequence[str]) -> Dict[str, int]:
    """Texts with Chinese characters, and texts with a second summary after '": "'."""
    return {"chinese": sum(bool(CHINESE.search(t)) for t in texts),
            "junk_tail": sum(JUNK_TAIL in t for t in texts)}


def write_cases(chosen: Dict[str, int], sources: Sequence[str], references: Sequence[str],
                preds: Dict[str, Dict[int, Dict[str, object]]], out_dir: str) -> None:
    """Write the five cases (source, reference, three outputs) as cases.json and cases.md."""
    shown = {"zeroshot": "zeroshot", "lora_r16": "lora_r16", "full_ft": "full_ft"}
    cases = []
    for stratum, i in chosen.items():
        case = {"stratum": stratum, "index": i, "source": sources[i], "reference": references[i]}
        for key in shown:
            case[key] = {"output": preds[key][i]["prediction"],
                         "rougeLsum": round(preds[key][i]["rougeLsum"], 4)}
        cases.append(case)
    with open(os.path.join(out_dir, "cases.json"), "w") as handle:
        json.dump(cases, handle, indent=2)
    with open(os.path.join(out_dir, "cases.md"), "w") as handle:
        for case in cases:
            handle.write(f"## Stratum {case['stratum']} - validation row {case['index']}\n\n"
                         f"**SOURCE:** {case['source']}\n\n**REFERENCE:** {case['reference']}\n\n")
            for key in shown:
                handle.write(f"**{key}** (ROUGE-Lsum {case[key]['rougeLsum']}): "
                             f"{case[key]['output']}\n\n")


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Select the cases and write all evidence files."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs_dir", default="runs", help="folder with the run directories")
    parser.add_argument("--out_dir", default="autopsy")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exclude", type=int, nargs="*", default=list(DEMO_ROWS),
                        help="validation rows left out of the pool (rows shown by predict.py)")
    parser.add_argument("--self_check", action="store_true")
    args = parser.parse_args(argv)
    if args.self_check:
        _self_check()
        return

    os.makedirs(args.out_dir, exist_ok=True)
    splits = load_laysumm()
    sources = list(splits["validation"][SOURCE_COLUMN])
    references = list(splits["validation"][TARGET_COLUMN])
    preds = {name: load_predictions(os.path.join(args.runs_dir, folder))
             for name, folder in RUN_DIRS.items()}
    for name, rows in preds.items():
        assert sorted(rows) == list(range(len(sources))), f"{name}: need all validation rows"

    pool = build_pool(splits, args.exclude)
    print(f"pool: {len(pool)} validation rows")
    chosen = select_cases(pool, sources, references, preds, args.seed)
    print("cases:", chosen)
    write_cases(chosen, sources, references, preds, args.out_dir)

    tiers = evaluation_tiers(splits["train"], splits["validation"])
    texts = {"reference": references}
    texts.update({name: [rows[i]["prediction"] for i in range(len(sources))]
                  for name, rows in preds.items()})
    summary: Dict[str, object] = {"pool": len(pool), "cases": chosen, "flag_rates": {},
                                  "artefacts": {}, "reading_grade": {}}
    print(f"\n{'':10s} {'tier':5s} {'neg lost':>9s} {'neg new':>8s} {'hedge lost':>11s} "
          f"{'hedge new':>10s}  (% of rows)")
    for name, outputs in texts.items():
        for tier in ("full", "clean"):
            rates = flag_rates(sources, outputs, tiers[tier])
            summary["flag_rates"][f"{name}/{tier}"] = rates
            print(f"{name:10s} {tier:5s} {rates['negation_lost']:9.1f} "
                  f"{rates['negation_gained']:8.1f} {rates['hedge_lost']:11.1f} "
                  f"{rates['hedge_gained']:10.1f}")
        summary["artefacts"][name] = artefact_counts(outputs)
    texts["source"] = sources
    for name, items in texts.items():
        grades = [reading_grade(t) for t in items]
        summary["reading_grade"][name] = sum(grades) / len(grades)
    print("\nartefacts (count of validation texts):", summary["artefacts"])
    print("mean Flesch-Kincaid grade:",
          {k: round(v, 2) for k, v in summary["reading_grade"].items()})
    with open(os.path.join(args.out_dir, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)


def _self_check() -> None:
    """Hand-checkable examples for the keyword matcher, copy ratio and reading grade."""
    assert has_term(NEGATION, "There is no pneumothorax.")
    assert has_term(NEGATION, "The lungs didn't collapse.")        # contraction inside a word
    assert not has_term(NEGATION, "A node is seen.")               # 'no' inside 'node' ignored
    assert has_term(HEDGE, "Pneumonia cannot be excluded.")
    assert not has_term(HEDGE, "The heart is enlarged.")
    assert has_term(ACUTE, "Small left Pneumothorax.") and not has_term(ACUTE, "Normal study.")
    assert copy_ratio("a b c d", "a b x y") == 0.5
    assert copy_ratio("", "a b") == 0.0
    # "The cat sat on the mat." = 6 words, 6 syllables, 1 sentence: 0.39*6 + 11.8*1 - 15.59.
    assert abs(reading_grade("The cat sat on the mat.") - (-1.45)) < 1e-9
    assert reading_grade("") == 0.0
    sources = ["No fluid. Heart is normal.", "Pneumonia is possible.", "Lungs are clear."]
    outputs = ["There is fluid. Heart is normal.", "There is pneumonia.", "Lungs are clear."]
    rates = flag_rates(sources, outputs, [0, 1, 2])
    third = 100 / 3
    assert abs(rates["negation_lost"] - third) < 1e-9 and abs(rates["hedge_lost"] - third) < 1e-9
    assert rates["negation_gained"] == 0 and rates["hedge_gained"] == 0
    assert artefact_counts(["a", '1": "2', "中文"]) == {"chinese": 1, "junk_tail": 1}
    print("autopsy self-check passed")


if __name__ == "__main__":
    main()
