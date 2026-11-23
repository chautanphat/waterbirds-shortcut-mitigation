"""Audit recovered binary DPE heads using held-out validation features and predictions."""

import argparse
import csv
import itertools
import json
import platform
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from src.methods.dpe_head_reweighting import ensemble
from src.methods.dpe_head_reweighting.aggregation_source import load_cache, partition
from src.methods.dpe_head_reweighting.experiment import digest
from src.methods.dpe_head_reweighting.reweighting import GROUPS, metrics, probabilities


ROOT = Path(__file__).resolve().parents[2]
FIT = "head_training"
HELD_OUT = "combiner_training"
HEADS, DIMENSION = 15, 2048
GROUP_COUNTS = (187, 186, 53, 53)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def source_file(directory, relative_path, name):
    path = (directory / relative_path).resolve(strict=True)
    if not path.is_relative_to(directory) or path.name != name:
        raise ValueError(f"Unexpected source file: {path}")
    return path


def load_source(kind, seed, directory, feature_path):
    """Validate row identities, source provenance and held-out split ancestry."""
    if kind == "standard_erm":
        model = source_file(directory, "checkpoints/ensemble.pt", "ensemble.pt")
        split = source_file(directory, "experiment/manifest.json", "manifest.json")
        record = source_file(directory, "experiment/heads.json", "heads.json")
        cache = source_file(directory, f"experiment/predictions/{HELD_OUT}.npz", f"{HELD_OUT}.npz")
    elif kind == "erm_star":
        model = source_file(directory, "neural_split/ensemble.pt", "ensemble.pt")
        split = source_file(directory, "neural_split/partition.json", "partition.json")
        cache = source_file(directory, f"neural_split/{HELD_OUT}.npz", f"{HELD_OUT}.npz")
    else:
        raise ValueError(f"Unknown cohort: {kind}")

    state = ensemble.load(model, dim=DIMENSION, count=HEADS)
    manifest = read_json(split)
    model_hash = digest(model)
    if kind == "standard_erm":
        record_data = read_json(record)
        split_hash = digest(split)
        if (
            record_data["lock"] != {"ensemble": model_hash, "manifest": split_hash}
            or record_data["head_ids"] != state["head_ids"]
            or record_data["selected_epochs"] != state["selected_epochs"]
            or state["provenance"]["manifest"] != split_hash
            or manifest["provenance"]["seed"] != seed
            or manifest["provenance"]["track"] != "standard"
            or manifest["parts"] != partition(manifest["validation"])
            or manifest["val_features_hash"] != digest(feature_path)
        ):
            raise ValueError(f"Standard ERM source identity mismatch: {directory}")
        parts = manifest["parts"]
        scores, labels, groups, ids = load_cache(directory, HELD_OUT, manifest, state["head_ids"], model_hash)
        inputs = [model, split, record, cache, feature_path]
    else:
        parts = manifest
        if (
            state["provenance"]["seed"] != seed
            or state["provenance"]["training_pool_ids"] != [row["id"] for row in parts[FIT]]
        ):
            raise ValueError(f"ERM* source identity mismatch: {directory}")
        with np.load(cache, allow_pickle=False) as saved:
            scores = probabilities(saved["probabilities"])
            labels, groups, ids = saved["labels"], saved["groups"], saved["ids"]
        inputs = [model, split, cache, feature_path]

    fit_rows, held_out = parts[FIT], parts[HELD_OUT]
    all_rows = fit_rows + held_out
    by_id = {row["id"]: row for row in all_rows}
    if (
        len(fit_rows) != 720 or len(held_out) != 479 or len(by_id) != 1199
        or any(row["split"] != 1 for row in all_rows)
        or scores.shape != (HEADS, len(held_out), 2)
        or ids.tolist() != [row["id"] for row in held_out]
        or labels.tolist() != [row["y"] for row in held_out]
        or groups.tolist() != [row["group"] for row in held_out]
        or tuple(int(sum(groups == group)) for group in GROUPS) != GROUP_COUNTS
    ):
        raise ValueError(f"Invalid validation split or cache: {directory}")

    with np.load(feature_path, allow_pickle=False) as data:
        feature_ids = data["ids"].tolist()
        raw = data["features"]
        if (
            len(feature_ids) != len(by_id) or set(feature_ids) != by_id.keys()
            or raw.shape != (len(by_id), DIMENSION)
        ):
            raise ValueError(f"Misaligned validation rows: {feature_path}")
        if kind == "erm_star" and (
            data["y"].tolist() != [by_id[key]["y"] for key in feature_ids]
            or data["g"].tolist() != [by_id[key]["group"] for key in feature_ids]
        ):
            raise ValueError(f"ERM* validation labels disagree with the partition: {feature_path}")
    index = {key: i for i, key in enumerate(feature_ids)}
    return state, raw[[index[key] for key in ids]], scores, labels, groups, ids, inputs


def unit_rows(values):
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if not np.all(np.isfinite(norms)) or np.any(norms <= 1e-12):
        raise ValueError("Degenerate prototype, direction or feature vector")
    return values / norms


def decision_directions(prototypes):
    normalized = unit_rows(prototypes)
    nominal = normalized[:, 1] - normalized[:, 0]
    # Row standardization constrains features to the subspace orthogonal to all ones.
    effective = nominal - nominal.mean(axis=1, keepdims=True)
    return normalized, nominal, effective


def stage_penalties(prototypes):
    """Replay the unweighted loss on saved stages, excluding norm penalty."""
    by_class = np.einsum("icd,jcd->ijc", prototypes, prototypes) / (prototypes.shape[-1] - 1)
    stages = []
    for index in range(len(prototypes)):
        terms = by_class[index, :index]
        denominator = prototypes.shape[1] * (index + 1)
        signed = float(abs(terms.sum()) / denominator)
        absolute = float(np.abs(terms).sum() / denominator)
        stages.append((signed, absolute, 1 - signed / absolute if absolute else None))
    return stages


def verify_replay(state, raw, cached, tolerance):
    replay = ensemble.predict(state, raw, device="cpu")
    cached_decisions = cached.argmax(axis=-1)
    replay_difference = float(np.max(np.abs(replay - cached)))
    if replay_difference > tolerance or not np.array_equal(replay.argmax(axis=-1), cached_decisions):
        raise ValueError(f"CPU replay does not reproduce the validation cache (max error {replay_difference:.3g})")

    prototypes = state["prototypes"].numpy().astype(np.float64)
    _, _, effective = decision_directions(prototypes)
    features = unit_rows(ensemble.standardize(raw).numpy().astype(np.float64))
    if np.any(state["distance_scales"].numpy() == 0):
        raise ValueError("A zero distance scale makes the decision direction undefined")
    margins = effective @ features.T
    # The independent float64 geometry can resolve near-ties differently from float32 inference.
    disagreements = (margins > 0) != cached_decisions
    if np.any(np.abs(margins[disagreements]) > 1e-6):
        raise ValueError("Analytical directions disagree with validation head decisions away from ties")

    # Powers of two deliberately change raw geometry without introducing roundoff on rescaling.
    factors = torch.tensor([0.5, 2.0], dtype=state["prototypes"].dtype)
    rescaled_state = dict(state, prototypes=state["prototypes"] * factors[None, :, None])
    rescaled = ensemble.predict(rescaled_state, raw, device="cpu")
    rescaled_difference = float(np.max(np.abs(rescaled - replay)))
    if rescaled_difference > tolerance or not np.array_equal(rescaled.argmax(axis=-1), cached_decisions):
        raise ValueError("Positive prototype rescaling changed predictions")
    scaled_prototypes = rescaled_state["prototypes"].numpy().astype(np.float64)
    return prototypes, scaled_prototypes, dict(
        replay_max_abs_error=replay_difference,
        analytic_near_tie_disagreements=int(disagreements.sum()),
        rescale_max_abs_error=rescaled_difference,
    )


def cosine(x, y):
    return float(np.clip(x @ y, -1, 1))


def pair_behavior(correct_a, correct_b):
    both_correct = int(np.count_nonzero(correct_a & correct_b))
    only_a = int(np.count_nonzero(correct_a & ~correct_b))
    only_b = int(np.count_nonzero(~correct_a & correct_b))
    count = len(correct_a)
    return dict(
        count=count, both_correct=both_correct, only_a_correct=only_a, only_b_correct=only_b,
        both_wrong=count - both_correct - only_a - only_b, disagreement=only_a + only_b,
        oracle_extra_correct=min(only_a, only_b),
    )


def describe_pairs(prototypes, state, correct, groups, identity):
    normalized, nominal, effective = decision_directions(prototypes)
    nominal = unit_rows(nominal)
    effective = unit_rows(effective)
    totals = {"all": np.ones(len(groups), dtype=bool)}
    totals.update({group: groups == group for group in GROUPS})
    rows = []
    for a, b in itertools.combinations(range(len(prototypes)), 2):
        raw_dot = np.einsum("cd,cd->c", prototypes[a], prototypes[b])
        sim = [cosine(x, y) for x, y in zip(normalized[a], normalized[b])]
        geometry = dict(
            head_a=state["head_ids"][a], head_b=state["head_ids"][b],
            raw_dot_class_0=float(raw_dot[0]), raw_dot_class_1=float(raw_dot[1]),
            prototype_cos_class_0=sim[0], prototype_cos_class_1=sim[1],
            mean_abs_prototype_cos=float(np.mean(np.abs(sim))),
            nominal_direction_cos=cosine(nominal[a], nominal[b]),
            effective_direction_cos=cosine(effective[a], effective[b]),
        )
        for group, mask in totals.items():
            rows.append(dict(**identity, **geometry, group=group, **pair_behavior(correct[a, mask], correct[b, mask])))
    return rows


def gate_for_seed(pair_rows, thresholds, config):
    """Require redundant pairs and a functional contrast within the same group."""
    decisions = {}
    for name, limits in thresholds.items():
        affected = [
            row for row in pair_rows if row["mean_abs_prototype_cos"] <= limits["max_prototype_similarity"]
            and row["effective_direction_cos"] >= limits["min_boundary_similarity"]
        ]
        controls = [row for row in pair_rows if row["effective_direction_cos"] <= limits["max_control_boundary_similarity"]]
        evidence = {}
        for group in GROUPS:
            observed = [row for row in affected if row["group"] == group]
            comparison = [row for row in controls if row["group"] == group]
            median_shared = float(np.median([row["both_wrong"] for row in observed])) if observed else None
            gain_gap = (
                float(np.median([row["oracle_extra_correct"] for row in comparison])
                      - np.median([row["oracle_extra_correct"] for row in observed]))
                if observed and comparison else None
            )
            evidence[group] = dict(
                affected_median_shared_error_rows=median_shared, control_minus_affected_median_oracle_rows=gain_gap,
                qualifies=(
                    min(len(observed), len(comparison)) >= config["min_pairs_per_comparison"]
                    and median_shared >= config["min_shared_error_rows"]
                    and gain_gap >= config["min_oracle_gain_gap_rows"]
                ),
            )
        decisions[name] = dict(
            affected_pairs=sum(row["group"] == "all" for row in affected),
            control_pairs=sum(row["group"] == "all" for row in controls),
            groups=evidence,
        )
    return decisions


def run_source(kind, seed, state, prototypes, rescaled, scores, labels, groups, replay, config):
    identity = dict(cohort=kind, seed=seed)
    pred = scores.argmax(axis=-1)
    correct = pred == labels
    pairs = describe_pairs(prototypes, state, correct, groups, identity)
    penalties = stage_penalties(prototypes)
    scaled_penalties = stage_penalties(rescaled)
    heads = []
    for index, key in enumerate(state["head_ids"]):
        signed, absolute, canceled = penalties[index]
        accuracy = metrics(scores[index], labels, groups)
        heads.append(dict(
            **identity, head_id=key, stage=index + 1, selected_epoch=state["selected_epochs"][index],
            norm_class_0=float(np.linalg.norm(prototypes[index, 0])),
            norm_class_1=float(np.linalg.norm(prototypes[index, 1])),
            abs_distance_scale=float(abs(state["distance_scales"][index])),
            signed_penalty=signed, termwise_abs_penalty=absolute, canceled_fraction=canceled,
            rescaled_signed_penalty=scaled_penalties[index][0],
            acc=accuracy["overall"], wga=accuracy["wga"],
            **{f"acc_{group}": accuracy["groups"][group] for group in GROUPS},
        ))
    changed_penalties = sum(
        not np.isclose(before[0], after[0], atol=1e-15, rtol=1e-10)
        for before, after in zip(penalties, scaled_penalties)
    )
    if not changed_penalties:
        raise ValueError("Rescaling did not change any stage's raw penalty")
    all_pairs = [row for row in pairs if row["group"] == "all"]
    uniform = metrics(scores.mean(axis=0), labels, groups)
    gate = gate_for_seed(pairs, config["thresholds"], config)
    summary = dict(
        **identity, validation_rows=len(labels), head_pairs=len(all_pairs),
        **{f"rows_{g}": count for g, count in zip(GROUPS, GROUP_COUNTS)},
        median_canceled_fraction=float(np.median([p[2] for p in penalties if p[2] is not None])),
        rescale_changed_stage_penalties=changed_penalties, **replay,
        median_abs_prototype_cos=float(np.median([row["mean_abs_prototype_cos"] for row in all_pairs])),
        median_effective_direction_cos=float(np.median([row["effective_direction_cos"] for row in all_pairs])),
        max_effective_direction_cos=max(row["effective_direction_cos"] for row in all_pairs),
        median_disagreement=float(np.median([row["disagreement"] for row in all_pairs])) / len(labels),
        median_oracle_extra_correct=float(np.median([row["oracle_extra_correct"] for row in all_pairs])),
        affected_pairs=gate["primary"]["affected_pairs"], control_pairs=gate["primary"]["control_pairs"],
        qualifying_groups=" ".join(g for g, v in gate["primary"]["groups"].items() if v["qualifies"]),
        uniform_acc=uniform["overall"], uniform_wga=uniform["wga"],
        **{f"uniform_acc_{g}": uniform["groups"][g] for g in GROUPS},
        **{f"oracle_acc_{g}": float(correct[:, groups == g].any(axis=0).mean()) for g in GROUPS},
    )
    return summary, heads, pairs, gate


def plot_pairs(rows, limits, output):
    fig, (geometry, behavior) = plt.subplots(1, 2, figsize=(10.4, 4.6), layout="constrained")
    symbols = [("standard_erm", "Standard ERM", "#245da6", "o"), ("erm_star", "ERM*", "#bc5100", "^")]
    for cohort, label, color, marker in symbols:
        points = [row for row in rows if row["cohort"] == cohort and row["group"] == "all"]
        boundary = [row["effective_direction_cos"] for row in points]
        style = dict(color=color, label=label, s=13, marker=marker, alpha=0.42, linewidths=0)
        geometry.scatter([row["mean_abs_prototype_cos"] for row in points], boundary, **style)
        behavior.scatter(boundary, [row["disagreement"] / row["count"] for row in points], **style)
    geometry.fill_between(
        [0, limits["max_prototype_similarity"]], limits["min_boundary_similarity"], 1,
        color="#dedfe3", zorder=0, label="Primary mismatch region",
    )
    geometry.axhline(limits["min_boundary_similarity"], color="#555555", ls="--", lw=1)
    geometry.set(
        xlabel="Mean absolute same-class prototype cosine", ylabel="Effective decision-direction cosine",
        title="Prototype geometry and decision alignment",
    )
    behavior.axvline(limits["min_boundary_similarity"], color="#555555", ls="--", lw=1)
    behavior.set(xlabel="Effective decision-direction cosine", ylabel="Head prediction disagreement",
                 title="Decision alignment and validation disagreement")
    behavior.yaxis.set_major_formatter(PercentFormatter(1))
    for axis in (geometry, behavior):
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.18)
    fig.legend(*geometry.get_legend_handles_labels(), loc="outside lower center", ncol=3,
               frameon=False, markerscale=1.6)
    fig.suptitle("Saved Waterbirds DPE heads · 105 pairs per backbone", fontsize=13)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config = read_json(args.config)
    torch.set_num_threads(1)
    summaries, heads, pairs, sources = [], [], [], []
    standard_counts = {name: dict.fromkeys(GROUPS, 0) for name in config["thresholds"]}
    reference = None
    seen = set()

    for cohort in config["cohorts"]:
        kind = cohort["name"]
        root = (ROOT / cohort["root"]).resolve(strict=True)
        feature_name = "val_raw.npz" if kind == "standard_erm" else "val.npz"
        for feature_set in cohort["feature_sets"]:
            for seed in feature_set["seeds"]:
                if (kind, seed) in seen:
                    raise ValueError(f"Duplicate source: {kind} seed {seed}")
                seen.add((kind, seed))
                directory = (root / cohort["source_pattern"].format(seed=seed)).resolve(strict=True)
                if not directory.is_relative_to(root):
                    raise ValueError(f"Source directory escapes its configured root: {directory}")
                features = source_file(root, feature_set["pattern"].format(seed=seed), feature_name)
                state, raw, scores, labels, groups, ids, inputs = load_source(kind, seed, directory, features)
                identity = list(zip(ids.tolist(), labels.tolist(), groups.tolist()))
                if reference is not None and identity != reference:
                    raise ValueError("Validation rows differ across sources")
                reference = identity
                prototype, scaled, replay = verify_replay(state, raw, scores, config["replay_probability_atol"])
                summary, head_rows, pair_rows, gate = run_source(
                    kind, seed, state, prototype, scaled, scores, labels, groups, replay, config,
                )
                summaries.append(summary)
                heads.extend(head_rows)
                pairs.extend(pair_rows)
                sources.append(dict(cohort=kind, seed=seed, gate=gate, inputs=[
                    dict(path=path.relative_to(ROOT).as_posix(), sha256=digest(path)) for path in inputs
                ]))
                if kind == "standard_erm":
                    for name, counts in standard_counts.items():
                        for group in GROUPS:
                            counts[group] += int(gate[name]["groups"][group]["qualifies"])
                print(f"{kind} seed {seed}: verified {len(labels)} validation images, {summary['affected_pairs']} affected pairs", flush=True)

    if seen != {("standard_erm", s) for s in range(5)} | {("erm_star", s) for s in range(2)}:
        raise ValueError("Expected all five standard ERM and both ERM* sources")
    decisions = {
        name: "go" if max(counts.values()) >= config["min_standard_seeds"] else "no-go"
        for name, counts in standard_counts.items()
    }
    output = args.output.resolve()
    if any(output.is_relative_to(ROOT / cohort["root"]) for cohort in config["cohorts"]):
        raise ValueError("Output must be outside recovered source directories")
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "summary.csv", summaries)
    write_csv(output / "heads.csv", heads)
    write_csv(output / "head_pairs.csv", pairs)
    plot_pairs(pairs, config["thresholds"]["primary"], output / "head_diversity.png")
    provenance = dict(
        study_id=config["study_id"], device="cpu", torch_threads=torch.get_num_threads(), split=HELD_OUT,
        working_directory=ROOT.as_posix(),
        command=["python", "-B", "-m", "scripts.analysis.dpe_head_diversity", *(argv if argv is not None else sys.argv[1:])],
        environment=dict(platform=platform.platform(), python=platform.python_version(), numpy=np.__version__,
                         torch=str(torch.__version__), matplotlib=matplotlib.__version__),
        configuration=dict(path=args.config.as_posix(), sha256=digest(args.config), settings=config),
        implementation_sha256={
            path.relative_to(ROOT).as_posix(): digest(path)
            for path in (Path(__file__).resolve(), Path(ensemble.__file__).resolve(),
                         ROOT / "src/methods/dpe_head_reweighting/aggregation_source.py",
                         ROOT / "src/methods/dpe_head_reweighting/reweighting.py")
        },
        sources=sources, qualifying_standard_seeds_by_group=standard_counts, decisions=decisions,
        output_sha256={name: digest(output / name) for name in
                       ("summary.csv", "heads.csv", "head_pairs.csv", "head_diversity.png")},
    )
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Primary decision: {decisions['primary'].upper()} ({standard_counts['primary']})")


if __name__ == "__main__":
    main()
