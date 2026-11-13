"""Predeclared independent predictors; aliases are algebraic, never score-based."""
from itertools import product
from pathlib import Path

import yaml

DEFAULT = Path(__file__).resolve().parents[3] / "configs/dpe_independent/study.yaml"
ORIGINAL = dict(family="original", scale="native")


def load_config(path=DEFAULT):
    c = yaml.safe_load(Path(path).read_text())
    if c["version"] != 1 or c["selection"] != "none_report_every_configuration":
        raise ValueError("Independent reporting is mandatory")
    for name, allowed in (("scales", {"native", "shared"}), ("memberships", {"cyclic", "permuted"}),
                          ("losses", {"average", "maximum"})):
        if not c[name] or len(set(c[name])) != len(c[name]) or not set(c[name]) <= allowed:
            raise ValueError(f"Invalid {name}")
    for name in ("temperatures", "penalties", "anchor_weight_decays", "joint_nll_penalties"):
        if not c[name] or any(type(v) not in (int, float) or not 0 < v < float("inf") for v in c[name]):
            raise ValueError(f"Invalid {name}")
    if c["steps"] < 1 or not 0 < c["learning_rate"] < 1 or c["crossfit"]["folds"] != 5:
        raise ValueError("Invalid fitting/crossfit settings")
    return c


def number(x):
    return format(x, "g")


def h3_id(scale, k, temperature, membership="cyclic"):
    return f"h3-{scale}-k{k}-T{number(temperature)}-{membership}"


def combination_inputs(c):
    return [dict(family="h3", scale="shared", k=k, temperature=t, membership="cyclic")
            for k, t in product(c["crossfit"]["subset_sizes"], c["crossfit"]["temperatures"])]


def candidates(c, fitted=True):
    rows = [dict(id="uniform", family="uniform", stage="geometry", input=ORIGINAL)]
    for scale, t, k, membership in product(c["scales"], c["temperatures"], c["subset_sizes"], c["memberships"]):
        row = dict(id=h3_id(scale, k, t, membership), family="h3", stage="geometry",
                   scale=scale, temperature=t, k=k, membership=membership)
        if k in (1, 15) and membership == "permuted":
            row["alias_of"] = h3_id(scale, k, t)
        if k == 1 and scale == "native" and t == 1. and membership == "cyclic":
            row["alias_of"] = "uniform"
        rows.append(row)
    for scale, t, k in product(c["scales"], c["temperatures"], c["local_sizes"]):
        row = dict(id=f"local-{scale}-k{k}-T{number(t)}", family="local", stage="geometry",
                   scale=scale, temperature=t, k=k)
        if k == 15:
            row["alias_of"] = h3_id(scale, 15, t)
        rows.append(row)
    if fitted:
        for scale, t, loss, penalty, kind in product(c["scales"], c["temperatures"], c["losses"], c["penalties"], ("importance", "bias")):
            rows.append(dict(id=f"prototype-{kind}-{scale}-T{number(t)}-{loss}-l{number(penalty)}",
                             family="prototype", stage="prototype", kind=kind, scale=scale,
                             temperature=t, loss=loss, penalty=penalty))
        for scale, loss, penalty, kind in product(c["scales"], c["losses"], c["penalties"], ("temperature", "temperature_bias")):
            rows.append(dict(id=f"prototype-{kind}-{scale}-{loss}-l{number(penalty)}",
                             family="prototype", stage="prototype", kind=kind, scale=scale,
                             temperature=1., loss=loss, penalty=penalty))
        for loss, penalty, kind in product(c["losses"], c["penalties"], ("affine", "linear", "nonlinear")):
            rows.append(dict(id=f"residual-{kind}-{loss}-l{number(penalty)}", family="residual", stage="h2",
                             kind=kind, input=ORIGINAL, loss=loss, penalty=penalty))
        for loss, penalty in product(c["losses"], c["penalties"]):
            rows.append(dict(id=f"brier-{loss}-l{number(penalty)}", family="brier", stage="h2",
                             input=ORIGINAL, loss=loss, penalty=penalty))
        for kind, wd in product(("logistic", "mlp", "mlp_dropout"), c["anchor_weight_decays"]):
            rows.append(dict(id=f"anchor-{kind}-wd{number(wd)}", family="anchor", stage="anchors",
                             input=ORIGINAL, kind=kind, weight_decay=wd))
        for penalty in c["joint_nll_penalties"]:
            rows.append(dict(id=f"joint-nll-maximum-l{number(penalty)}", family="joint", stage="anchors",
                             input=ORIGINAL, loss="maximum", penalty=penalty))
        for rep in combination_inputs(c):
            prefix = f"combined-k{rep['k']}-T{number(rep['temperature'])}"
            rows.append(dict(id=prefix+"-uniform", family="uniform", stage="combination", input=rep,
                             alias_of=h3_id("shared", rep["k"], rep["temperature"])))
            for kind, wd in product(("logistic", "mlp"), c["anchor_weight_decays"]):
                rows.append(dict(id=f"{prefix}-{kind}-wd{number(wd)}", family="anchor", stage="combination",
                                 input=rep, kind=kind, weight_decay=wd))
            for penalty in c["penalties"]:
                rows.append(dict(id=f"{prefix}-brier-maximum-l{number(penalty)}", family="brier", stage="combination",
                                 input=rep, loss="maximum", penalty=penalty))
            for loss, penalty in product(c["losses"], c["penalties"]):
                rows.append(dict(id=f"{prefix}-affine-{loss}-l{number(penalty)}", family="residual", stage="combination",
                                 input=rep, kind="affine", loss=loss, penalty=penalty))
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate configuration IDs")
    if any(r.get("alias_of", "uniform") not in ids for r in rows):
        raise ValueError("Missing alias target")
    return rows


def crossfit_candidates(c):
    rows = []
    for rep in [ORIGINAL, *combination_inputs(c)]:
        name = "original" if rep["family"] == "original" else f"h3-k{rep['k']}-T{number(rep['temperature'])}"
        rows.append(dict(id=f"crossfit-{name}-uniform", family="uniform", input=rep, stage="crossfit"))
        for loss, penalty in product(c["losses"], c["penalties"]):
            rows.append(dict(id=f"crossfit-{name}-affine-{loss}-l{number(penalty)}", family="residual",
                             input=rep, kind="affine", loss=loss, penalty=penalty, stage="crossfit"))
    return rows
