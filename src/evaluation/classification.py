"""Waterbirds metrics; all accuracies are fractions in [0, 1]."""

import torch
import torch.nn.functional as F


def group_metrics(predictions, targets, attributes):
    predictions, targets, attributes = [torch.as_tensor(x).cpu()
                                        for x in (predictions, targets, attributes)]
    if predictions.ndim != 1 or targets.shape != predictions.shape or attributes.shape != targets.shape:
        raise ValueError("Predictions, targets and attributes must be matching 1-D arrays")
    if not len(targets):
        raise ValueError("Cannot evaluate an empty split")
    if any(not torch.all((x == 0) | (x == 1)) for x in (predictions, targets, attributes)):
        raise ValueError("Waterbirds targets, attributes and predictions must be binary")
    predictions, targets, attributes = [x.long() for x in (predictions, targets, attributes)]
    correct = predictions == targets
    groups = 2 * targets + attributes
    per_group = {}
    for group in range(4):
        mask = groups == group
        count = int(mask.sum())
        per_group[str(group)] = {
            "target": group // 2, "background": group % 2, "count": count,
            "accuracy": float(correct[mask].float().mean()) if count else None,
        }
    # Do not present a three-group minimum as the four-group Waterbirds WGA.
    complete = all(group["count"] for group in per_group.values())
    return {"accuracy": float(correct.float().mean()), "count": len(targets),
            "worst_group_accuracy": min(g["accuracy"] for g in per_group.values()) if complete else None,
            "per_group": per_group}


@torch.inference_mode()
def evaluate(model, loader, device="cpu"):
    model.eval()
    predictions, targets, attributes = [], [], []
    loss_sum = 0.0
    for _, inputs, labels, attrs in loader:
        logits = model(inputs.to(device))
        loss_sum += F.cross_entropy(logits, labels.to(device), reduction="sum").item()
        predictions.append(logits.argmax(1).cpu())
        targets.append(labels)
        attributes.append(attrs)
    if not targets:
        raise ValueError("Cannot evaluate an empty loader")
    metrics = group_metrics(torch.cat(predictions), torch.cat(targets), torch.cat(attributes))
    metrics["loss"] = loss_sum / metrics["count"]
    return metrics
