"""Paired 2D vs 3D comparison on the same test volumes (helper for the README).

Run from the project folder after copying the result JSONs from Rangpur:
    python compare_2d_3d.py --dir outputs_rangpur
"""
import argparse, json, math
from collections import defaultdict

# (metric, higher_is_better)
METRICS = [("dice_prostate", True), ("dice_prostate_base", True), ("dice_prostate_apex", True),
           ("dice_bladder", True), ("dice_rectum", True),
           ("leak_into_bladder", False), ("leak_into_rectum", False)]


def load(path):
    with open(path) as f:
        return {r["scan"]: r for r in json.load(f)["per_volume"]}


def valid(x):
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="outputs_rangpur")
    args = p.parse_args()
    r2, r3 = load(f"{args.dir}/test_results_2d.json"), load(f"{args.dir}/test_results_3d.json")
    scans = sorted(set(r2) & set(r3))
    print(f"{len(scans)} test volumes in both files\n")

    print("3D vs 2D: better / same (|diff| < 0.001) / worse")
    print(f"{'metric':<20}{'volumes':>16}{'patients':>16}")
    for k, higher in METRICS:
        pairs = [(r2[s][k], r3[s][k], r2[s]["case"]) for s in scans if valid(r2[s][k]) and valid(r3[s][k])]
        def outcome(a, b):                      # +1 = 3D better, 0 = same, -1 = 3D worse
            if abs(b - a) < 1e-3:
                return 0
            return 1 if (b > a) == higher else -1
        def tally(results):
            return f"{results.count(1)}/{results.count(0)}/{results.count(-1)}"
        vol = [outcome(a, b) for a, b, _ in pairs]
        by_case = defaultdict(list)
        for a, b, c in pairs:
            by_case[c].append((a, b))
        pat = [outcome(sum(a for a, _ in v) / len(v), sum(b for _, b in v) / len(v))
               for v in by_case.values()]
        print(f"{k:<20}{tally(vol):>16}{tally(pat):>16}")

    print("\nPer-volume prostate results, sorted by 3D prostate Dice (worst first):")
    print(f"{'scan':<16}{'2D Dice':>9}{'3D Dice':>9}{'2D base':>9}{'3D base':>9}"
          f"{'2D apex':>9}{'3D apex':>9}{'2D leakR':>10}{'3D leakR':>10}{'3D leakB':>10}")
    f = lambda x: f"{x:.3f}" if valid(x) else "  nan"
    for s in sorted(scans, key=lambda s: r3[s]["dice_prostate"] if valid(r3[s]["dice_prostate"]) else 1):
        a, b = r2[s], r3[s]
        print(f"{s:<16}{f(a['dice_prostate']):>9}{f(b['dice_prostate']):>9}"
              f"{f(a['dice_prostate_base']):>9}{f(b['dice_prostate_base']):>9}"
              f"{f(a['dice_prostate_apex']):>9}{f(b['dice_prostate_apex']):>9}"
              f"{f(a['leak_into_rectum']):>10}{f(b['leak_into_rectum']):>10}"
              f"{f(b['leak_into_bladder']):>10}")


if __name__ == "__main__":
    main()