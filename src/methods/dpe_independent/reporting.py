"""Every configuration, including missing/failed rows; sources stay separate."""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from .grid import candidates, crossfit_candidates
from .storage import atomic_json, read_json


def report(output):
    output = Path(output)
    plan = read_json(output/"study.json")
    workers = read_json(output/"workers.json") if (output/"workers.json").exists() else {}
    latest_attempt = {attempt["source"]: attempt for attempt in workers.get("attempts", [])}
    rows, grouped = [], defaultdict(list)
    for source in plan["sources"]:
        source_root = Path(plan.get("worker_outputs", {}).get(source["id"], str(output))) / source["id"]
        arms = ["full_validation", *[f"partition{p}" for p in plan["config"]["partitions"]]]
        if plan["crossfit"]:
            arms.append("crossfit")
        for arm in arms:
            specs = crossfit_candidates(plan["config"]) if arm == "crossfit" else candidates(plan["config"], arm != "full_validation")
            ledger_path = source_root/arm/"ledger.json"
            ledger = read_json(ledger_path) if ledger_path.exists() else {}
            for spec in specs:
                state = ledger.get(spec["id"], dict(status="planned", error="Not executed in this session"))
                attempt = latest_attempt.get(source["id"], {})
                if state["status"] not in ("completed", "alias", "failed", "blocked") and attempt.get("returncode", 0) != 0:
                    state = dict(state, status="blocked", error=f"Source worker exited {attempt['returncode']}; see {source['id']}.log")
                m = state.get("metrics", {})
                row = dict(source_id=source["id"], recipe=source["recipe"], seed=source["seed"], arm=arm,
                           configuration=spec["id"], family=spec["family"], stage=spec["stage"], status=state["status"],
                           alias_of=spec.get("alias_of", ""), wga=m.get("wga"), overall=m.get("overall"),
                           delta_matched_uniform_pp=m.get("delta_matched_uniform_pp"),
                           group_balanced_nll=m.get("group_balanced_nll"), group_balanced_brier=m.get("group_balanced_brier"),
                           error=state.get("error", ""), experiment=state.get("experiment", ""))
                for g in ("00", "01", "10", "11"):
                    row[f"group_{g}"] = m.get("groups", {}).get(g)
                    row[f"count_{g}"] = m.get("counts", {}).get(g)
                rows.append(row)
                grouped[(row["recipe"], arm, spec["id"])].append(row)
    summary = []
    for (recipe, arm, configuration), entries in grouped.items():
        complete = [r for r in entries if r["status"] in ("completed", "alias")]
        values = [r["wga"] for r in complete]
        deltas = [r["delta_matched_uniform_pp"] for r in complete]
        summary.append(dict(recipe=recipe, arm=arm, configuration=configuration, family=entries[0]["family"],
                            sources_expected=len(entries), sources_completed=len(complete),
                            source_seeds=",".join(str(r["seed"]) for r in complete),
                            mean_wga=float(np.mean(values)) if values else None,
                            source_sd_wga=float(np.std(values, ddof=1)) if len(values)>1 else None,
                            mean_delta_matched_uniform_pp=float(np.mean(deltas)) if deltas else None,
                            incomplete=len(complete) != len(entries)))
    for name, values in (("per_configuration.csv", rows), ("summary.csv", summary)):
        with (output/name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]) if values else [])
            writer.writeheader(); writer.writerows(values)
    statuses = {s: sum(r["status"] == s for r in rows) for s in sorted({r["status"] for r in rows})}
    final = dict(expected=len(rows), statuses=statuses, results_complete=all(r["status"] in ("completed", "alias", "failed", "blocked") for r in rows),
                 selected_configuration=None, independent_backbone_sources={recipe: sum(s["recipe"] == recipe for s in plan["sources"]) for recipe in ("normal_erm", "ermstar")},
                 failures=[{k: r[k] for k in ("source_id", "arm", "configuration", "status", "error")} for r in rows if r["status"] in ("failed", "blocked")])
    atomic_json(output/"completion.json", final)
    text = ["# Independent DPE methods", "", "Every declared configuration is retained. No predictor is selected using test scores.", "",
            f"Declared rows: {len(rows)}. Status counts: {statuses}.", "",
            "`per_configuration.csv` contains all source/allocation/configuration results and failures. `summary.csv` averages each source's WGA within the same recipe and allocation; SD is sample SD across sources. Accuracies are fractions, deltas are percentage points.", "",
            "Normal ERM and ERM* remain separate. Validation partitions and crossfit folds are nested within source backbones. These are exploratory results on a repeatedly inspected test set, not independent formal confirmation.", "",
            "Every completed predictor retains run/model records, data-only fitted parameters, an independent NumPy replay, and canonical test predictions. No image-level replay was performed."]
    (output/"README.md").write_text("\n".join(text)+"\n")
    return final
