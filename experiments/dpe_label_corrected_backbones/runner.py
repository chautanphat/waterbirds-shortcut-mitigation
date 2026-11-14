#!/usr/bin/env python3
"""Audited DPE orchestration with a label-set switch (official | corrected Waterbirds labels).

Derived from the runner of the public notebook noerula/dpe-reproduction. Changes, all marked
"LABEL-SET" or "SMOKE": expected group counts per label set; --steps / --num-stages for a smoke
test only; head-count checks follow --num-stages. Upstream training code is unchanged.

Training functions are loaded from pinned upstream sources, not reimplemented here.
See docs/dpe_reproduction_audit.md for protocol limitations.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import importlib.util
from types import SimpleNamespace

DPE_COMMIT = "68b811461e238d44cced6ae6c8ce75b66cad5449"
SUBPOP_COMMIT = "4d3dbbe21029666ef19d040e110ec22908640c5b"
# LABEL-SET: (y, a) counts per split, order (0,0), (0,1), (1,0), (1,1). "corrected" relabels the
# three species that generate_waterbirds.py marks waterbird through the substring "tern"
# (021 Eastern Towhee, 088 Western Meadowlark, 102 Western Wood-Pewee) as landbirds.
GROUP_COUNTS_BY_LABEL_SET = {
    "official": {0: [3498, 184, 56, 1057], 1: [467, 466, 133, 133], 2: [2255, 2255, 642, 642]},
    "corrected": {0: [3502, 252, 52, 989], 1: [478, 473, 122, 126], 2: [2298, 2302, 599, 595]},
}
GROUP_COUNTS = GROUP_COUNTS_BY_LABEL_SET["official"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def verify_repo(path, commit):
    actual = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if actual != commit:
        raise RuntimeError(f"Expected {commit}, found {actual}: {path}")
    if subprocess.check_output(["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"], text=True):
        raise RuntimeError(f"Upstream tracked files have been modified: {path}")


def verify_data(data_dir, label_set="official"):
    import pandas as pd
    GROUP_COUNTS = GROUP_COUNTS_BY_LABEL_SET[label_set]  # LABEL-SET
    root = Path(data_dir) / "waterbirds"
    meta = root / "metadata_waterbirds.csv"
    df = pd.read_csv(meta)
    if (len(df) != sum(map(sum, GROUP_COUNTS.values())) or
            df[["filename", "y", "a", "split"]].isna().any().any() or
            set(df.y) != {0, 1} or set(df.a) != {0, 1}):
        raise ValueError("Expected the complete original Waterbirds dataset with binary labels/attributes")
    if set(df.split) != {0, 1, 2} or df.filename.duplicated().any():
        raise ValueError("Unexpected splits or duplicate image paths")
    for s, expected in GROUP_COUNTS.items():
        observed = df[df.split == s].groupby(["y", "a"]).size()
        counts = [int(observed.get((y, a), 0)) for y in (0, 1) for a in (0, 1)]
        if counts != expected:
            raise ValueError(f"Split {s}: expected {expected}, got {counts}")
    image_root = root / "waterbird_complete95_forest2water2"
    image_hash = hashlib.sha256()
    for name in df.filename:
        p = image_root / name
        if not p.is_file():
            raise FileNotFoundError(p)
        image_hash.update(name.encode() + b"\0" + bytes.fromhex(sha256(p)))
    return {"metadata_sha256": sha256(meta), "images_sha256": image_hash.hexdigest(),
            "group_counts": GROUP_COUNTS, "rows": len(df), "label_set": label_set}


def load_functions(dpe_root, names, namespace):
    """Compile only named definitions; never execute notebook cells or magics."""
    path = Path(dpe_root) / "notebooks/01_waterbirds_with_attribute_annotation.ipynb"
    nb = json.loads(path.read_text())
    found = {}
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        source = "\n".join(line for line in source.splitlines() if not line.startswith(("%", "!")))
        for node in ast.parse(source).body:
            if isinstance(node, ast.FunctionDef) and node.name in names:
                found[node.name] = node
    if set(found) != set(names):
        raise ValueError(f"Missing upstream definitions: {set(names) - set(found)}")
    module = ast.Module(body=list(found.values()), type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def load_official_main(dpe_root):
    spec = importlib.util.spec_from_file_location("dpe_official_main", Path(dpe_root) / "main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stage1_command(args, out):
    """The Waterbirds train_pe.sh + train_all_pe.sh invocation, without Slurm/W&B."""
    return [sys.executable, "main.py", "--dataset_name", "Waterbirds", "--model_name", "resnet50",
            "--data_dir", str(args.data_dir), "--pretrained_path", str(out / "ckpt_last.pt"),
            "--ckpt_dir", str(out / "stage1"), "--seed", str(args.seed), "--device", "cuda",
            "--loss_name", "isomax", "--stage", "1", "--num_stages", str(args.num_stages), "--epochs", "20",
            "--train_mode", "freeze", "--train_attr", args.train_attr, "--subsample_type", "group",
            "--batch_size", "256", "--batch_size_eval", "512", "--workers", str(args.workers),
            "--optim", "sgd", "--lr", "0.001", "--momentum", "0.9", "--optim_weight_decay", "0",
            "--scheduler", "none", "--wd_weight", "10", "--cov_reg", "500000", "--entropic_scale", "30",
            "--norm_emb", "yes", "--ensemble_criterion", "wga_val", "--wdb_group", "dpe",
            "-ncbt", "-sit", "--no_wandb"]


def official_environment(dpe_root):
    """Avoid jsonargparse's implicit environment arguments overriding defaults."""
    env = dict(os.environ)
    for node in ast.walk(ast.parse((Path(dpe_root) / "main.py").read_text())):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            for argument in node.args:
                if isinstance(argument, ast.Constant) and isinstance(argument.value, str) and argument.value.startswith("--"):
                    env.pop(argument.value[2:].upper(), None)
    return env


def run_official_cli(args, out, datasets, device):
    """Train with the original CLI, then evaluate its saved final ensemble."""
    import torch
    from torch.utils.data import DataLoader
    from utils.misc import fix_random_seed
    from utils.models import get_model

    stage_dir = out / "stage1"
    stage_dir.mkdir()
    cmd = stage1_command(args, out)
    write_json(out / "stage1_command.json", {"argv": cmd, "cwd": str(args.dpe_root)})
    with open(out / "stage1.log", "w") as log:
        subprocess.run(cmd, cwd=args.dpe_root, env=official_environment(args.dpe_root),
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    prototypes = torch.load(stage_dir / "prototype_ensemble_wga_val.pt", map_location="cpu", weights_only=True)
    scales = torch.load(stage_dir / "dist_scales_wga_val.pt", map_location="cpu", weights_only=True)
    heads = args.num_stages - 1  # SMOKE: 15 heads unless --num-stages is lowered
    if prototypes.shape != (2, heads, 2048) or len(scales) != heads:
        raise ValueError(f"Official CLI did not produce a {heads}-head, 2048-dimensional ensemble")
    if not torch.isfinite(prototypes).all() or not all(s.shape == (1,) and torch.isfinite(s).all() for s in scales):
        raise ValueError("Nonfinite prototypes or invalid distance scales in the official output")
    ensemble = [[prototypes[:, i:i+1].to(device), s.to(device)] for i, s in enumerate(scales)]
    torch.save([[p.cpu(), s.cpu()] for p, s in ensemble], out / "ensemble.pt")

    # Replay the original NumPy subsampler solely to save exact membership/order.
    # Only dataset construction consumes NumPy randomness in the original CLI;
    # torch initializers/DataLoaders affect the separate torch RNG stream.
    fix_random_seed(args.seed, True, False)
    from utils import datasets as dsets
    subsets = []
    for stage in range(1, args.num_stages):
        ds = dsets.Waterbirds(str(args.data_dir), "va", None, train_attr=args.train_attr,
                             subsample_type="group", stage=stage, pre_extracted_feats=datasets["val"].feats)
        subsets.append({"stage": stage, "files": ds.x, "labels": ds.y, "groups": ds.g,
                        "cov_reg": 5e5, "batches_per_epoch": len(ds) // 256})
    write_json(out / "subsets.json", subsets)
    model = get_model("Waterbirds", 2, "freeze", str(out / "ckpt_last.pt"),
                      loss_name="isomax", pretrained_imgnet=False).to(device)
    test_loader = DataLoader(datasets["test"], batch_size=512, shuffle=False, num_workers=args.workers)
    # Use the released evaluator directly; no custom ensemble arithmetic.
    logits = load_official_main(args.dpe_root).evaluate_ensemble_fixed_backbone(
        prototypes.to(device), test_loader, model, distance_scales=[s.to(device) for s in scales], return_preds=True)
    probabilities = logits.softmax(2).mean(0).numpy()
    return probabilities


def metric_summary(probabilities, dataset):
    import numpy as np
    labels, groups = np.asarray(dataset.y), np.asarray(dataset.g)
    probabilities = np.asarray(probabilities)
    if (probabilities.shape != (len(labels), 2) or not np.isfinite(probabilities).all() or
            (probabilities < 0).any() or (probabilities > 1).any() or
            not np.allclose(probabilities.sum(1), 1, atol=1e-6) or
            set(groups) != {"00", "01", "10", "11"}):
        raise ValueError("Expected finite normalized binary probabilities and all four Waterbirds groups")
    prediction = probabilities.argmax(1)
    correct = prediction == labels
    per_group = {g: {"n": int((groups == g).sum()),
                     "correct": int(correct[groups == g].sum()),
                     "accuracy": float(correct[groups == g].mean())} for g in sorted(set(groups))}
    return {"overall_accuracy": float(correct.mean()),
            "wga": min(v["accuracy"] for v in per_group.values()), "per_group": per_group}


def convert_checkpoint(source, target):
    """Strictly map known DPE or SubpopBench ResNet50/linear checkpoints."""
    import torch
    import torchvision
    from torch import nn
    # Only load checkpoints the user trusts; SubpopBench includes argparse objects.
    ckpt = torch.load(source, map_location="cpu", weights_only=False)
    if "model_dict" in ckpt:
        state = ckpt["model_dict"]
        prefix = "featurizer.network."
        children = {"conv1": "0", "bn1": "1", "layer1": "4", "layer2": "5", "layer3": "6", "layer4": "7"}
        mapped = {}
        for k, v in state.items():
            if k.startswith(prefix):
                name, rest = k[len(prefix):].split(".", 1)
                mapped[f"0.{children[name]}.{rest}"] = v
        for k in ("weight", "bias"):
            mapped["2." + k] = state["classifier." + k]
    else:
        mapped = ckpt
    net = torchvision.models.resnet50(weights=None)
    model = nn.Sequential(nn.Sequential(*list(net.children())[:-1]), nn.Flatten(), nn.Linear(2048, 2))
    model.load_state_dict(mapped, strict=True)
    torch.save(model.state_dict(), target)
    return model


def run_stage0(args, out):
    target = out / "ckpt_last.pt"
    if args.protocol == "checkpoint":
        if not args.checkpoint or not args.checkpoint_provenance:
            raise ValueError("checkpoint mode requires a trusted --checkpoint and --checkpoint-provenance")
        model = convert_checkpoint(args.checkpoint, target)
        return model, {"source": str(args.checkpoint), "sha256": sha256(args.checkpoint),
                       "provenance": args.checkpoint_provenance, "selection": "supplied checkpoint"}
    if args.protocol == "subpopbench":
        # Original entry point, default hyperparameters, fixed 5001 steps; no ES.
        # This is a new default-hparam run, NOT the paper authors' recovered checkpoint.
        bootstrap = ("import runpy; from subpopbench.dataset.datasets import Waterbirds; "
                     f"Waterbirds.N_WORKERS={args.workers}; "
                     "runpy.run_module('subpopbench.train', run_name='__main__')")
        cmd = [sys.executable, "-c", bootstrap, "--algorithm", "ERM", "--dataset", "Waterbirds",
               "--train_attr", "no", "--data_dir", str(args.data_dir), "--output_dir", str(out / "subpopbench"),
               "--output_folder_name", "standard", "--seed", str(args.seed), "--hparams_seed", "0"]
        if args.steps:  # SMOKE only; the real run keeps SubpopBench's default 5,001 steps
            cmd += ["--steps", str(args.steps)]
        cwd = args.subpop_root
    else:
        epochs = 100 if args.protocol == "legacy100" else 400
        cmd = [sys.executable, "main.py", "--dataset_name", "Waterbirds", "--data_dir", str(args.data_dir),
               "--ckpt_dir", str(out / "native"), "--pretrained_imgnet", "--loss_name", "ce",
               "--epochs", str(epochs), "--lr", "0.001", "--batch_size", "128", "--batch_size_eval", "256" if args.protocol == "legacy100" else "512",
               "--workers", str(args.workers), "--no_wandb", "--seed", str(args.seed)]
        if args.protocol == "legacy100":
            cmd.append("--no_augmentation")
        cwd = args.dpe_root
        (out / "native").mkdir()
    write_json(out / "stage0_command.json", {"argv": cmd, "cwd": str(cwd)})
    with open(out / "stage0.log", "w") as f:
        subprocess.run(cmd, cwd=cwd, env=official_environment(args.dpe_root), stdout=f,
                       stderr=subprocess.STDOUT, check=True)
    if args.protocol == "subpopbench":
        source = out / "subpopbench/standard_attrNo" / f"Waterbirds_ERM_hparams0_seed{args.seed}/model.pkl"
        if not (source.parent / "done").exists():
            raise RuntimeError("SubpopBench did not finish")
    else:
        source = out / "native/ckpt_last.pt"
    model = convert_checkpoint(source, target)
    return model, {"source": str(source), "sha256": sha256(source), "selection": "fixed final checkpoint",
                   "note": "Fresh run; paper's selected ERM artifact/hparams not recovered"}


def extract_features(model, dataset, device, workers):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    model.eval()
    features = []
    with torch.no_grad():
        for _, inputs, *_ in DataLoader(dataset, batch_size=256, shuffle=False, num_workers=workers):
            features.append(model[:-1](inputs.to(device)).cpu().numpy())
    raw = np.concatenate(features)
    if raw.shape != (len(dataset), 2048) or not np.isfinite(raw).all() or (raw.std(1) == 0).any():
        raise ValueError("Invalid raw features")
    return raw


def fit_dpe(args, out, model, device):
    import numpy as np
    import torch
    import torchvision
    from torch import nn
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from utils import datasets as dsets
    from utils.misc import fix_random_seed
    from utils.eval_helpers import eval_metrics
    from utils.isomaxplus import IsoMaxPlusLossFirstPart, IsoMaxPlusLossSecondPart

    datasets, raw_features = {}, {}
    fix_random_seed(args.seed, True, False)  # Deterministic convolution for raw feature extraction.
    model = model.to(device)
    for name, split in (("val", "va"), ("test", "te")):
        datasets[name] = dsets.Waterbirds(str(args.data_dir), split, None)
        if args.head_implementation == "official_cli":
            extraction_args = SimpleNamespace(dataset_name="Waterbirds", train_mode="freeze",
                                              pretrained_path=str(out / "ckpt_last.pt"), device=device,
                                              model_name="resnet50", workers=args.workers, batch_size_eval=512)
            raw = load_official_main(args.dpe_root).extract_features(extraction_args, datasets[name])
        else:
            raw = extract_features(model, datasets[name], device, args.workers)
        if raw.shape != (len(datasets[name]), 2048) or not np.isfinite(raw).all() or (raw.std(1) == 0).any():
            raise ValueError("Invalid raw features")
        raw_features[name] = raw
        np.save(out / f"feats_{name}.npy", raw)
        # NumPy ddof=0, per image. This is the upstream DPE normalization.
        datasets[name].feats = (raw - raw.mean(1, keepdims=True)) / raw.std(1, keepdims=True)

    with torch.no_grad():
        raw_logits = model[-1](torch.from_numpy(raw_features["test"]).to(device))
        erm = metric_summary(raw_logits.softmax(1).cpu().numpy(), datasets["test"])
    write_json(out / "erm_metrics_raw_features.json", erm)

    if args.head_implementation == "official_cli":
        probs = run_official_cli(args, out, datasets, device)
        return save_predictions(out, probs, datasets["test"])

    namespace = dict(torch=torch, torchvision=torchvision, nn=nn, np=np, DataLoader=DataLoader, tqdm=tqdm,
                     dsets=dsets, eval_metrics=eval_metrics, IsoMaxPlusLossFirstPart=IsoMaxPlusLossFirstPart,
                     IsoMaxPlusLossSecondPart=IsoMaxPlusLossSecondPart, data_dir=str(args.data_dir),
                     workers=args.workers, batch_size_train=256)
    load_functions(args.dpe_root, ["evaluate", "get_subsampled_train_set", "get_train_loader",
                                  "init_model", "train_prototypes", "cov_reg_scheduler_inv"], namespace)
    # Upstream evaluate/train_prototypes default to cuda. Isolation maps the process
    # to one GPU. CPU is supported only in unit smoke tests, with this default bound.
    upstream_evaluate = namespace["evaluate"]
    namespace["evaluate"] = lambda m, loader: upstream_evaluate(m, loader, device=device)
    val_loader = DataLoader(datasets["val"], batch_size=256, shuffle=False, num_workers=args.workers)

    fix_random_seed(args.seed)  # Same reset and cudnn flags as the upstream notebook.
    train_loader = namespace["get_train_loader"](datasets, args.train_attr)
    scheduler = namespace["cov_reg_scheduler_inv"](5e5, 15)
    ensemble, full_model = [], None
    subsets = []
    for stage in range(1, 16):
        ds = train_loader.dataset
        # Preserve membership AND order, since minibatch/drop_last behavior matters.
        subsets.append({"stage": stage, "files": ds.x, "labels": ds.y, "groups": ds.g,
                        "cov_reg": scheduler(stage), "batches_per_epoch": len(train_loader)})
        write_json(out / "subsets.json", subsets)
        full_model = namespace["init_model"](str(out / "ckpt_last.pt"), _model=full_model, device=device)
        chosen = namespace["train_prototypes"](
            train_loader, val_loader, full_model, _prototype_ensemble=ensemble,
            epochs=20, cov_reg=scheduler(stage), wd_weight=10, entropic=30, lr=1e-3,
            _stage=stage, verbose=False, device=device)
        if chosen is None or not all(torch.isfinite(t).all() for t in chosen):
            raise RuntimeError(f"No finite selected head at stage {stage}")
        ensemble.append(chosen)
        torch.save([[p.cpu(), s.cpu()] for p, s in ensemble], out / "ensemble.pt")
        # The upstream demo evaluates on test after every stage; replace this with
        # an equally shaped VAL inference to preserve DataLoader RNG consumption.
        # Its model-head assignment is preserved before initializing the next head.
        ensemble_probabilities(ensemble, val_loader, full_model, device, args.dpe_root)
        train_loader = namespace["get_train_loader"](datasets, args.train_attr)

    # Fixed 15 heads, then a single final test inference. No best-test-head count.
    test_loader = DataLoader(datasets["test"], batch_size=256, shuffle=False, num_workers=args.workers)
    probs = ensemble_probabilities(ensemble, test_loader, full_model, device, args.dpe_root)
    return save_predictions(out, probs, datasets["test"])


def save_predictions(out, probs, dataset):
    import numpy as np
    result = metric_summary(probs, dataset)
    np.savez_compressed(out / "test_predictions.npz", probabilities=probs,
                        y=np.asarray(dataset.y), groups=np.asarray(dataset.g), filenames=np.asarray(dataset.x))
    return result


def ensemble_probabilities(ensemble, loader, model, device, dpe_root):
    """Run the original notebook evaluator; capture its probability array unchanged."""
    import torch
    import numpy as np
    from torch import nn
    from tqdm import tqdm
    from utils.eval_helpers import eval_metrics, get_acc
    captured = {}
    def capture(probabilities, *labels):
        captured["probabilities"] = probabilities
        return eval_metrics(probabilities, *labels)
    namespace = dict(torch=torch, nn=nn, np=np, tqdm=tqdm, eval_metrics=capture, get_acc=get_acc)
    load_functions(dpe_root, ["evaluate_ensemble"], namespace)
    namespace["evaluate_ensemble"](ensemble, loader, model, device=device, verbose=False)
    return captured["probabilities"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dpe-root", type=Path, required=True)
    p.add_argument("--subpop-root", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--protocol", choices=["subpopbench", "checkpoint", "legacy100", "ermstar400"], default="subpopbench")
    p.add_argument("--head-implementation", choices=["official_cli", "official_notebook"], default="official_cli")
    p.add_argument("--train-attr", choices=["yes", "no"], default="yes")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--checkpoint-provenance", default="")
    p.add_argument("--label-set", choices=["official", "corrected"], required=True)  # LABEL-SET
    p.add_argument("--steps", type=int, default=0)  # SMOKE only
    p.add_argument("--num-stages", type=int, default=16)  # SMOKE only
    args = p.parse_args()
    for name in ("dpe_root", "subpop_root", "data_dir", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    verify_repo(args.dpe_root, DPE_COMMIT)
    verify_repo(args.subpop_root, SUBPOP_COMMIT)
    args.output_dir.mkdir(parents=True, exist_ok=False)  # Never silently reuse stale seed artifacts.
    import torch
    import torchvision
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Launch with CUDA_VISIBLE_DEVICES=<one GPU>; expected exactly one visible CUDA device")
    manifest = {"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "source": {"dpe": DPE_COMMIT, "subpopbench": SUBPOP_COMMIT, "runner_sha256": sha256(__file__)},
                "environment": {"python": sys.version, "torch": torch.__version__, "torchvision": torchvision.__version__,
                                "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                                "gpu": torch.cuda.get_device_name(0), "visible_devices": os.getenv("CUDA_VISIBLE_DEVICES")},
                "dataset": verify_data(args.data_dir, args.label_set), "status": "running"}
    write_json(args.output_dir / "manifest.json", manifest)
    freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (args.output_dir / "pip-freeze.txt").write_text(freeze)
    # Native SubpopBench checkpoints pickle their EarlyStopping class alongside
    # model_dict, even without early stopping. Make that trusted module available.
    sys.path.insert(0, str(args.subpop_root))
    try:
        model, provenance = run_stage0(args, args.output_dir)
        manifest["backbone"] = provenance
        manifest["checkpoint_sha256"] = sha256(args.output_dir / "ckpt_last.pt")
        write_json(args.output_dir / "manifest.json", manifest)
        sys.path.insert(0, str(args.dpe_root))
        result = fit_dpe(args, args.output_dir, model, "cuda")
        write_json(args.output_dir / "metrics.json", result)
        manifest["artifacts"] = {name: sha256(args.output_dir / name) for name in
                                 ("ensemble.pt", "feats_val.npy", "feats_test.npy", "subsets.json", "test_predictions.npz")}
        if args.head_implementation == "official_cli":
            for name in ("stage1/prototype_ensemble_wga_val.pt", "stage1/dist_scales_wga_val.pt", "stage1_command.json"):
                manifest["artifacts"][name] = sha256(args.output_dir / name)
        manifest["status"] = "complete"
        print(json.dumps(result), flush=True)
    except BaseException as e:
        manifest["status"] = "failed"
        manifest["error"] = repr(e)
        raise
    finally:
        write_json(args.output_dir / "manifest.json", manifest)


if __name__ == "__main__":
    main()