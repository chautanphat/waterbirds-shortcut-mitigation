"""Read-only CSV/record audit; no model fitting, PyTorch, or test-based selection.

python scripts/analysis/audit_dpe_independent.py --study /path/to/study --output /path/to/audit.json
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
from collections import defaultdict


def read(path):
    return json.loads(Path(path).read_text())


def audit(study):
    study = Path(study)
    plan = read(study/"study.json")
    completion = read(study/"completion.json")
    hashes, metadata_cache, checked = {}, {}, []

    def digest(path):
        path = Path(path)
        if str(path) not in hashes:
            value = hashlib.sha256()
            with path.open("rb") as stream:
                while block := stream.read(1024*1024):
                    value.update(block)
            hashes[str(path)] = value.hexdigest()
        return hashes[str(path)]

    for source in plan["sources"]:
        path = source["metadata"]
        if path not in metadata_cache:
            with Path(path).open(newline="", encoding="utf-8-sig") as stream:
                records = list(csv.DictReader(stream))
            reference = {}
            for row in records:
                if row["split"] != "2":
                    continue
                name = row.get("img_filename") or row.get("filename")
                prefix = "waterbird_complete95_forest2water2/"
                if name.startswith(prefix):
                    name = name[len(prefix):]
                assert name not in reference
                reference[name] = (int(row["y"]), int(row.get("place", row.get("a"))))
            assert len(reference) == 5794
            metadata_cache[path] = reference
        reference = metadata_cache[path]
        worker = Path(plan.get("worker_outputs", {}).get(source["id"], str(study)))
        for ledger_path in (worker/source["id"]).glob("*/ledger.json"):
            for identifier, state in read(ledger_path).items():
                if state["status"] not in ("completed", "alias"):
                    continue
                root = Path(state["experiment"])
                provenance = read(root/"predictions/test/provenance.json")
                prediction_path = root/"predictions/test/predictions.csv"
                assert digest(prediction_path) == provenance["csv_sha256"], root
                assert provenance["seed"] == source["seed"]
                assert provenance["split"] == "test" and provenance["run_label"] == root.name
                counts, correct, seen = [0]*4, [0]*4, set()
                with prediction_path.open(newline="") as stream:
                    for row in csv.DictReader(stream):
                        name = row["image_path"]
                        assert name not in seen and name in reference, root
                        seen.add(name)
                        y, a = reference[name]
                        g = 2*y+a
                        prediction = {"landbird": 0, "waterbird": 1}[row["predicted_label"]]
                        assert row["true_label"] == ("landbird", "waterbird")[y]
                        assert int(row["group"]) == g+1
                        assert row["spurious_attribute"] == ("land", "water")[a]
                        assert row["split"] == "test" and int(row["seed"]) == source["seed"]
                        assert row["run_label"] == root.name
                        assert int(row["correct"]) == int(prediction == y)
                        assert .5 <= float(row["confidence"]) <= 1
                        counts[g] += 1; correct[g] += int(prediction == y)
                assert seen == set(reference) and counts == [2255, 2255, 642, 642]
                group_accuracy = [c/n for c, n in zip(correct, counts)]
                m = read(root/"metrics.json")["metrics"]
                assert abs(m["wga"]-min(group_accuracy)) < 1e-12
                assert abs(m["overall"]-sum(correct)/sum(counts)) < 1e-12
                assert m == state["metrics"]
                for g, accuracy in zip(("00", "01", "10", "11"), group_accuracy):
                    assert abs(m["groups"][g]-accuracy) < 1e-12
                run, model = read(root/"run.json"), read(root/"model.json")
                assert run["experiment_id"] == model["experiment_id"] == root.name
                assert run["execution"]["status"] == "completed"
                assert run["seed"] == model["seed"] == source["seed"]
                assert model["input"]["kind"] == "features"
                assert model["verification"]["changed_labels"] == 0
                assert model["verification"]["maximum_probability_difference"] <= 1e-10
                for artifact in model["artifacts"]:
                    assert digest(artifact["location"]) == artifact["sha256"], artifact["location"]
                implementation = model["implementation"]
                assert digest(implementation["snapshot"]) == implementation["sha256"]
                frozen_path = Path(run["artifacts"]["fitted_state"])
                frozen = read(frozen_path)
                arm = frozen_path.parent.parent
                lock = read(arm/"fit_complete.json")[identifier]
                assert lock["sha256"] == digest(frozen_path)
                identity = hashlib.sha256(json.dumps(read(arm/"inputs.json"), sort_keys=True, allow_nan=False).encode()).hexdigest()
                assert frozen["identity"] == identity
                if frozen["configuration"]["family"] == "brier":
                    fitted = frozen["model"]
                    assert fitted["objective_gap_bound"] <= 1e-7
                    assert min(fitted["weights"]) >= 0 and abs(sum(fitted["weights"])-1) < 1e-12
                checked.append(dict(source=source["id"], arm=ledger_path.parent.name, configuration=identifier,
                                    status=state["status"], wga=m["wga"], delta=m["delta_matched_uniform_pp"]))
                if len(checked) % 500 == 0:
                    print("Audited", len(checked), "predictors", flush=True)
    expected = completion["statuses"].get("completed", 0)+completion["statuses"].get("alias", 0)
    assert len(checked) == expected
    with (study/"per_configuration.csv").open(newline="") as stream:
        table = list(csv.DictReader(stream))
    assert len(table) == completion["expected"]
    table_index = {(r["source_id"], r["arm"], r["configuration"]): r for r in table}
    assert len(table_index) == len(table)
    for row in checked:
        saved = table_index[(row["source"], row["arm"], row["configuration"])]
        assert abs(float(saved["wga"])-row["wga"]) < 1e-12
    grouped = defaultdict(list)
    for row in table:
        grouped[(row["recipe"], row["arm"], row["configuration"])].append(row)
    with (study/"summary.csv").open(newline="") as stream:
        summaries = list(csv.DictReader(stream))
    assert len(summaries) == len(grouped)
    for row in summaries:
        group = grouped[(row["recipe"], row["arm"], row["configuration"])]
        complete = [r for r in group if r["status"] in ("completed", "alias")]
        assert int(row["sources_expected"]) == len(group)
        assert int(row["sources_completed"]) == len(complete)
        assert len({r["seed"] for r in complete}) == len(complete)
        if complete:
            assert abs(float(row["mean_wga"])-statistics.mean(float(r["wga"]) for r in complete)) < 1e-12
            assert abs(float(row["mean_delta_matched_uniform_pp"])-statistics.mean(float(r["delta_matched_uniform_pp"]) for r in complete)) < 1e-10
        if len(complete)>1:
            assert abs(float(row["source_sd_wga"])-statistics.stdev(float(r["wga"]) for r in complete)) < 1e-12
    return dict(status="passed", predictors_audited=len(checked), rows_per_predictor=5794,
                auditor_sha256=digest(__file__), study_plan_sha256=digest(study/"study.json"),
                expected_configurations=completion["expected"], execution_statuses=completion["statuses"],
                source_counts=completion["independent_backbone_sources"], selected_predictor=None,
                checks=["canonical CSV hashes", "full metadata identities, labels, groups and counts", "independent integer accuracy arithmetic",
                        "ledger/table/record agreement", "source-wise mean/SD and paired-delta summaries",
                        "fitted-state and head-artifact hashes", "frozen-fit locks and convex Brier certificates", "implementation snapshot hashes",
                        "retained independent NumPy replay evidence"],
                limitations=["Does not rerun CNN image extraction or optimizer training", "Reused sources and test set remain exploratory"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.study)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))
