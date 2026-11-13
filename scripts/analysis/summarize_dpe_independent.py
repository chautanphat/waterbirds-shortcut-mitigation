"""Build the complete local report after the independent CSV audit passes."""
import argparse
import csv
import json
from pathlib import Path
import shutil


def build(study, output):
    study, output = Path(study), Path(output)
    audit = json.loads((study/"independent_audit.json").read_text())
    completion = json.loads((study/"completion.json").read_text())
    if audit["status"] != "passed" or completion["failures"] or not completion["results_complete"]:
        raise ValueError("Complete audited results are required")
    output.mkdir(parents=True, exist_ok=True)
    with (study/"summary.csv").open() as stream:
        summary = list(csv.DictReader(stream))
    with (study/"per_configuration.csv").open() as stream:
        per_seed = list(csv.DictReader(stream))
    index = {(r["recipe"], r["arm"], r["configuration"]): r for r in summary}

    def score(recipe, arm, configuration):
        return 100*float(index[(recipe, arm, configuration)]["mean_wga"])

    def delta(recipe, arm, configuration):
        return float(index[(recipe, arm, configuration)]["mean_delta_matched_uniform_pp"])

    def extent(recipe, arm, predicate):
        values = [100*float(r["mean_wga"]) for r in summary if r["recipe"] == recipe and r["arm"] == arm and predicate(r["configuration"])]
        return f"{min(values):.4f}–{max(values):.4f}"

    (output/"summary.csv").write_text((study/"summary.csv").read_text())
    shutil.copy2(study/"completion.json", output/"completion.json")
    shutil.copy2(study/"independent_audit.json", output/"independent_audit.json")
    # Keep portable tables in the repository; full generated paths stay with outputs.
    fields = [k for k in per_seed[0] if k != "experiment"]
    with (output/"per_seed.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader(); writer.writerows(per_seed)
    numerical = json.loads((study/"numerical_recovery.json").read_text())
    (output/"numerical_checks.json").write_text(json.dumps(dict(
        earlier_attempt_retained=True, changed_method_grid=False, tolerances=numerical["preserved_tolerances"],
        precision_checks=numerical["precision_checks"], brier_checks=numerical["brier_checks"]), indent=2)+"\n")

    importance = "prototype-importance-shared-T0.05-maximum-l0.01"
    linear = "residual-linear-average-l0.1"
    affine = "residual-affine-average-l0.1"
    lines = ["# Independent DPE methods: completed frozen-backbone study", "",
        "Executed 2026-11-13. All **5,880 declared rows** completed: **5,222 distinct predictors and 658 algebraic aliases**, with no failed or missing configurations. Every row has its own test scores and canonical prediction export. No method or setting was selected for final testing.", "",
        "The strongest primary-allocation prototype-weighting result does not survive every allocation. A fixed linear residual gives positive mean changes on all three normal-ERM allocations, but does not transfer that pattern to ERM*. Distinct H3 subsets and local evidence do not establish a broad improvement over original DPE.", "",
        "## Protocol and evidence", "",
        "Waterbirds official splits are 4,795/1,199/5,794; test groups 00/01/10/11 contain 2,255/2,255/642/642 examples. Normal ERM has five reused independent sources (seeds 0–4); ERM* has two (0–1). Sources, validation allocations, methods and folds are not pooled as independent repetitions. Means below average each source's minimum group accuracy.", "",
        "Each source contributes 85 full-validation geometry rows, 240 rows at each of three 720/479 allocations, and 35 five-fold crossfit rows. Fitted methods use only their declared validation fitting labels. Test labels do not select settings, memberships, checkpoints or methods. Full-validation DPE is a separate complete-procedure reference.", "",
        "The study ran in the managed Codex cloud execution workspace using four CPU worker processes. Earlier references to local execution mean that attached workspace. No backbone was trained. Saved features/heads were recovered from the two existing public Kaggle notebooks. New lightweight heads were fitted only for missing ERM* allocations and crossfit folds. The fresh Kaggle launcher was prepared and syntax-checked; no new Kaggle run was submitted.", "",
        "Base source revision: `22b65ec5fc8f00b33938fe2af6dc32ad2f34ecfd`, plus the retained local implementation snapshot. New differentiable fits use the declared 400 CPU float64 Adam steps. Existing neural anchors preserve their AdamW recipes. All settings remain in the [configuration](../../../configs/dpe_independent/study.yaml).", "",
        "See [execution, result counts and artifact availability](execution.md) for the 5,880-row/658-alias derivation, the CPU environment, public input sources, retained-output layout and reproduction commands.", "", "## Baselines and fixed-procedure allocation sensitivity", "",
        "WGA is in percent. P0/P1/P2 are nested validation allocations, not additional backbone sources. These are the **same fixed procedures** across allocations.", "",
        "| Recipe / procedure | P0 | P1 | P2 |", "| --- | ---: | ---: | ---: |"]
    names = [("Uniform", "uniform"), ("Affine calibration, average loss, λ=.1", affine),
             ("Linear residual, average loss, λ=.1", linear),
             ("Prototype importance, shared scale, T=.05, maximum loss, λ=.01", importance),
             ("Existing MLP, wd=.01", "anchor-mlp-wd0.01"),
             ("Existing dropout MLP, wd=.01", "anchor-mlp_dropout-wd0.01")]
    for recipe in ("normal_erm", "ermstar"):
        label = "Normal ERM (5 sources)" if recipe == "normal_erm" else "ERM* (2 sources)"
        for name, identifier in names:
            cells = " | ".join(f"{score(recipe, 'partition'+str(i), identifier):.4f}" for i in range(3))
            lines.append(f"| {label}: {name} | {cells} |")
    lines += ["", f"Full-validation original DPE means are **{score('normal_erm', 'full_validation', 'uniform'):.4f}%** (normal ERM) and **{score('ermstar', 'full_validation', 'uniform'):.4f}%** (ERM*). These are our reproduced references, not paper-reported numbers.", "",
        "### Prototype importance", "",
        f"The fixed shared-scale/T=.05/maximum-loss/λ=.01 procedure changes primary WGA by **{delta('normal_erm', 'partition0', importance):+.4f} pp** on normal ERM and **{delta('ermstar', 'partition0', importance):+.4f} pp** on ERM*. It improves all five primary normal-ERM sources. On ERM* its paired changes are −0.6231 and +2.1729 pp: the mean gain does not mean both sources improve.", "",
        "The same procedure's paired mean changes across P0/P1/P2 are:", "",
        "| Recipe | P0 Δ (pp) | P1 Δ (pp) | P2 Δ (pp) |", "| --- | ---: | ---: | ---: |"]
    for recipe in ("normal_erm", "ermstar"):
        lines.append("| "+recipe+" | "+" | ".join(f"{delta(recipe, 'partition'+str(i), importance):+.4f}" for i in range(3))+" |")
    lines += ["", "The normal-ERM P2 reversal rejects an allocation-independent gain for this procedure. Calibration controls also remain competitive on the other allocations. Prototype importance is a useful exploratory signal, with substantial fitting sensitivity.", "",
        "### Linear residual and matched calibration", "",
        "The average-loss/λ=.1 linear residual has normal-ERM mean gains of +1.1141, +0.4018 and +0.5567 pp. Its gains over the corresponding affine-only control are:", "",
        "| Recipe | P0 extra Δ (pp) | P1 extra Δ (pp) | P2 extra Δ (pp) |", "| --- | ---: | ---: | ---: |"]
    for recipe in ("normal_erm", "ermstar"):
        lines.append("| "+recipe+" | "+" | ".join(f"{score(recipe, 'partition'+str(i), linear)-score(recipe, 'partition'+str(i), affine):+.4f}" for i in range(3))+" |")
    lines += ["", "For normal ERM the linear residual wins/loses/ties against uniform on 4/1/0, 4/0/1 and 4/0/1 sources across the three allocations. On ERM* it underperforms the matched affine control on all three allocation means. This is a recipe-specific finding, not a universal neural-aggregation advance.", "",
        "## Every family remains in the results", "",
        "The following are **ranges of source-mean WGA over fixed primary-allocation settings**. Extrema describe the completed grid; they are not selected predictors. Full per-source settings and controls remain in [per_seed.csv](per_seed.csv), and all 1,680 recipe/allocation/configuration summaries are in [summary.csv](summary.csv).", "",
        "| Family | Normal ERM WGA range (%) | ERM* WGA range (%) |", "| --- | ---: | ---: |"]
    groups = [
        ("Tempered original pairs", lambda k: k.startswith("h3-") and "-k1-" in k and k.endswith("-cyclic")),
        ("Distinct H3 subsets", lambda k: k.startswith("h3-") and any(f"-k{i}-" in k for i in (2, 3, 5))),
        ("Global prototype pooling", lambda k: k.startswith("h3-") and "-k15-" in k and k.endswith("-cyclic")),
        ("Local top-k evidence", lambda k: k.startswith("local-") and "-k15-" not in k),
        ("Learned prototype importance", lambda k: k.startswith("prototype-importance")),
        ("Prototype temperature/bias controls", lambda k: k.startswith("prototype-") and not k.startswith("prototype-importance")),
        ("Affine-only residual control", lambda k: k.startswith("residual-affine")),
        ("Linear residual", lambda k: k.startswith("residual-linear")),
        ("Nonlinear residual", lambda k: k.startswith("residual-nonlinear")),
        ("Average/maximum-group Brier", lambda k: k.startswith("brier-")),
        ("Existing logistic/MLP/dropout anchors", lambda k: k.startswith("anchor-")),
        ("Existing maximum-group NLL weights", lambda k: k.startswith("joint-")),
        ("Fitted H3+H2 combinations", lambda k: k.startswith("combined-") and not k.endswith("-uniform"))]
    for label, predicate in groups:
        lines.append(f"| {label} | {extent('normal_erm', 'partition0', predicate)} | {extent('ermstar', 'partition0', predicate)} |")
    lines += ["", "Distinct H3 subset gains are small and inconsistent across allocations. The best observed full-validation local-evidence mean is 89.8058% on normal ERM, versus 89.5950% original DPE; its best ERM* counterpart is 94.0031%, below the 94.0810% reference. Sharpening and original-pair controls are retained, so these changes are not automatically attributed to subset competition.", "",
        "Certified maximum-group Brier fits do not improve the primary original-head uniform mean in either recipe. Some combinations improve particular allocations, without establishing a stable overall improvement. The tested nonlinear residual does not provide a consistent advantage over its simpler controls.", "",
        "## Cross-fitting", "",
        "Each procedure uses five 15-head ensembles. Calibration is fitted on out-of-fold predictions and applied to each fold ensemble before averaging test probabilities. The additional heads and different per-fold training pools change computation as well as data usage.", "",
        "| Recipe | Uncalibrated original fold ensemble | Range of all 35 crossfit procedures | Full-validation original DPE |",
        "| --- | ---: | ---: | ---: |"]
    for recipe in ("normal_erm", "ermstar"):
        lines.append(f"| {recipe} | {score(recipe, 'crossfit', 'crossfit-original-uniform'):.4f} | {extent(recipe, 'crossfit', lambda k: True)} | {score(recipe, 'full_validation', 'uniform'):.4f} |")
    lines += ["", "The highest observed crossfit means are 90.2736% and 94.0483%. Normal ERM benefits from this fold ensemble; ERM* does not exceed its full-validation reference in mean. Uncalibrated fold controls prevent attributing the entire difference to calibration.", "",
        "![Fixed procedures across allocations](allocation_sensitivity.png)", "",
        "Each line shows one fixed procedure's source-mean WGA; no line switches settings between allocations. The plot is descriptive and does not show confidence intervals.", "",
        "## Numerical recovery, verification and limits", "",
        "The initial float32 CPU distance path differed from historical GPU probabilities by up to 7.75e-7, with unchanged decisions. That preflight and its outputs were retained. Float64 normalized distances reduce the maximum difference in all nine historical ensemble replays to 1.1921e-7, with zero changed head or uniform decisions. The 2e-7 historical tolerance was not relaxed.", "",
        "Some initial Brier solves had a successful solver flag but failed the independent certificate. The final fitter retains quadratic shrinkage in its lower bound and refines rejected solves at tighter numerical tolerance. The objective and 1e-7 certificate acceptance threshold are unchanged; all final fits passed.", "",
        "Final checks: **219 tests passed, 9 skipped**. The [independent audit](independent_audit.json) verified all 5,880 exports, 5,794 reference images per export, integer accuracy/group arithmetic, source summaries, frozen-state/head/snapshot hashes, Brier certificate evidence, and retained NumPy replay checks. Frozen-model replay tolerance is 1e-10 with no label differences. No independent CNN image replay was performed.", "",
        "All results are exploratory. The same sources and test set informed earlier work, the grid is broad, and ERM* has only two sources. Positive source means, grid maxima or repeated allocations do not establish formal confirmation, population-level superiority, novel methods, or causal removal of background shortcuts. Historical ERM* feature markers lack a backbone digest and selected head epochs were not exported; those facts remain unavailable.", "",
        "Full raw features, native heads, fitted parameters, canonical CSVs, per-predictor run/model records, source snapshots, solver histories, recovery evidence and logs remain outside Git. The source implementation, lightweight tables and audit evidence are included in this GitHub publication. Large generated artifacts remain outside Git.", "",
        "Recreate the table/report from retained outputs with `scripts/analysis/summarize_dpe_independent.py`; rerun the read-only audit with `scripts/analysis/audit_dpe_independent.py`. The [method README](../../../src/methods/dpe_independent/README.md) and [fresh Kaggle notebook](../../../notebooks/dpe_independent_methods_kaggle.ipynb) document execution and recovery."]
    (output/"README.md").write_text("\n".join(lines)+"\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    curves = [("Uniform", "uniform", "#777777"), ("Affine control", affine, "#2c7fb8"),
              ("Linear residual", linear, "#31a354"), ("Prototype importance", importance, "#e6550d")]
    for ax, recipe, label in zip(axes, ("normal_erm", "ermstar"), ("Normal ERM · 5 sources", "ERM* · 2 sources")):
        for name, identifier, color in curves:
            ax.plot(range(3), [score(recipe, "partition"+str(i), identifier) for i in range(3)],
                    marker="o", linewidth=2, label=name, color=color)
        ax.axhline(score(recipe, "full_validation", "uniform"), color="#5e3c99", linestyle="--", label="Full-validation DPE")
        ax.set(xticks=range(3), xticklabels=["P0", "P1", "P2"], xlabel="Validation allocation", ylabel="Mean worst-group accuracy (%)", title=label)
        ax.grid(axis="y", alpha=.2); ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=8, loc="lower left")
    fig.savefig(output/"allocation_sensitivity.png", dpi=180)
    plt.close(fig)
    return output/"README.md"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(build(args.study, args.output))
