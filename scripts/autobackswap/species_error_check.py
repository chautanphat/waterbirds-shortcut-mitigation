"""Which species account for the errors, and how do the numbers change without three species whose whole
folder is labelled waterbird in metadata.csv? Exploratory check, not part of the method.

Inputs: per-example predictions written by dump_all_predictions.py (<pred_dir>/<run>/predictions.csv)
and the Waterbirds metadata.csv. Prints, per run: errors from the listed species, and Acc / WGA on the
full test set versus the test set without those species.
"""
import argparse
import collections
import csv
import glob
import os

SPECIES = ["021.Eastern_Towhee", "088.Western_Meadowlark", "102.Western_Wood_Pewee"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--species", nargs="*", default=SPECIES)
    a = ap.parse_args()

    labels = collections.defaultdict(set)  # species folder -> set of y values in metadata.csv
    with open(a.metadata) as f:
        for r in csv.DictReader(f):
            labels[r["img_filename"].split("/")[0]].add(r["y"])
    for s in a.species:
        print(f"{s}: y values in metadata.csv = {sorted(labels[s])} (1 = waterbird)")
    print()

    for p in sorted(glob.glob(f"{a.pred_dir}/*/predictions.csv")):
        run = os.path.basename(os.path.dirname(p))
        n, e, ns, es = (collections.Counter() for _ in range(4))
        with open(p) as f:
            for r in csv.DictReader(f):
                g, sp = int(r["group"]), r["image_path"].split("/")[0] in a.species
                wrong = r["correct"] == "0"
                n[g] += 1
                e[g] += wrong
                if sp:
                    ns[g] += 1
                    es[g] += wrong
        wga = min(100 * (1 - e[g] / n[g]) for g in (1, 2, 3, 4))
        wga_x = min(100 * (1 - (e[g] - es[g]) / (n[g] - ns[g])) for g in (1, 2, 3, 4))
        N, E, NS, ES = sum(n.values()), sum(e.values()), sum(ns.values()), sum(es.values())
        print(f"{run:28s} errors {E:4d}, from listed species {ES:3d}/{NS} images | "
              f"Acc {100 * (1 - E / N):.2f} -> {100 * (1 - (E - ES) / (N - NS)):.2f} | WGA {wga:.2f} -> {wga_x:.2f}")


if __name__ == "__main__":
    main()
