"""Preserved DPE routines, copied without changing their math.

Source: baselines/dpe4subpop/utils/{isomaxplus,datasets}.py and
baselines/dpe4subpop/notebooks/src/utils.py. See upstream MIT LICENSE.
Only imports and the feature-only evaluation callback are adapted here.
"""
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from src.methods.dpe_head_reweighting.reweighting import metrics


def evaluate(model, loader):
    device = next(model.parameters()).device
    predictions = []
    with torch.no_grad():
        for _, _, _, _, _, features in loader:
            predictions.append(model[-1](features.to(device)).softmax(-1).cpu().numpy())
    labels = loader.dataset.tensors[2].numpy()
    groups = loader.dataset.groups
    return {'min_group': {'accuracy': metrics(np.concatenate(predictions), labels, groups)['wga']}}


class IsoMaxPlusLossFirstPart(nn.Module):
    """This part replaces the model classifier output layer nn.Linear()"""

    def __init__(self, num_features, num_classes, temperature=1.0, const_init=0):
        super(IsoMaxPlusLossFirstPart, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.temperature = temperature
        self.prototypes = nn.Parameter(torch.Tensor(num_classes, num_features))
        self.distance_scale = nn.Parameter(torch.Tensor(1))
        nn.init.normal_(self.prototypes, mean=const_init, std=1e-2)
        nn.init.constant_(self.distance_scale, 1.0)

    def forward(self, features):
        distances = torch.abs(self.distance_scale) * torch.cdist(F.normalize(features), F.normalize(self.prototypes),
                                                                 p=2.0, compute_mode="donot_use_mm_for_euclid_dist")
        logits = -distances
        # The temperature may be calibrated after training to improve uncertainty estimation.
        return logits / self.temperature

class IsoMaxPlusLossSecondPart(nn.Module):
    """This part replaces the nn.CrossEntropyLoss()"""

    def __init__(self, entropic_scale=20.0, reduction='none'):
        super(IsoMaxPlusLossSecondPart, self).__init__()
        self.entropic_scale = entropic_scale
        self.reduction = reduction

    def forward(self, logits, targets, reduction=None, debug=False):
        #############################################################################
        #############################################################################
        """Probabilities and logarithms are calculated separately and sequentially"""
        """Therefore, nn.CrossEntropyLoss() must not be used to calculate the loss"""
        #############################################################################
        #############################################################################
        self.reduction = reduction if reduction is not None else self.reduction
        targets = targets.argmax(1) if targets.ndim == 2 else targets  # added
        distances = -logits
        probabilities_for_training = nn.Softmax(dim=1)(-self.entropic_scale * distances)
        probabilities_at_targets = probabilities_for_training[range(distances.size(0)), targets]
        loss = -torch.log(probabilities_at_targets)

        if not debug:
            if reduction == 'none':
                return loss
            return loss.mean()
        else:
            targets_one_hot = torch.eye(distances.size(1))[targets].long().cuda()
            intra_inter_distances = torch.where(targets_one_hot != 0, distances, torch.Tensor([float('Inf')]).cuda())
            inter_intra_distances = torch.where(targets_one_hot != 0, torch.Tensor([float('Inf')]).cuda(), distances)
            intra_distances = intra_inter_distances[intra_inter_distances != float('Inf')]
            inter_distances = inter_intra_distances[inter_intra_distances != float('Inf')]
            return loss, 1.0, intra_distances, inter_distances

def subsample(df: pd.DataFrame,
              target='y',
              target_split: int = 0,
              verbose=False,
              sort_idx=False,
              filter_mask=None,
              num_samples=None,
              dynamic_num_samples=False,
              stage=None) -> pd.DataFrame:
    """

    :param stage:
    :param filter_mask:
    :param sort_idx: sort the output df by their index
    :param verbose: print counts per target-group
    :param target: sub-sampling column, 'y' for class for 'subgroup' for subgroup
    :param df: metadata created by subpop repo
    :param target_split: 0: train, 1: val, 2: test
    :param num_samples: a fixed number of samples in each class
    :return: sub-sampled metadata
    """
    if target is None:
        return df

    if 'train' in set(df.split):  # special case with ImagenetBG
        splits = {0: 'train', 1: 'val', 2: 'test'}
        target_split = splits[target_split]

    all_num_samples = (df.groupby('split').get_group(target_split)[target].value_counts())
    if num_samples is None:
        num_samples = all_num_samples.min()

    indices = df[df.split != target_split].index.tolist()
    target_indices = []
    for i, subgroup in enumerate(all_num_samples.index):

        idx = (df.groupby('split')
               .get_group(target_split)
               .groupby(target)
               .get_group(subgroup)
               .index.tolist())

        if (
                target == 'subgroup') and dynamic_num_samples:  # equal number of samples in each subgroup, could be different among classes
            df_split = df.groupby('split').get_group(target_split)
            cls = df_split.loc[df_split.subgroup == subgroup].iloc[0].y
            num_samples = df_split.groupby('y').get_group(cls)[target].value_counts().min()

        if filter_mask is not None:
            idx_by_subgroup = (df[df.split == target_split].reset_index()
                               .groupby(target)
                               .get_group(subgroup)
                               .index.tolist())
            sampling_weights = filter_mask[idx_by_subgroup]
            idx = np.array(idx)[np.argsort(sampling_weights)[::-1]]  # [::-1]
        tmp = list(np.random.permutation(idx)[:num_samples])
        target_indices.append(tmp)

    target_indices = np.concatenate(target_indices).reshape((len(target_indices), -1)).T.flatten().tolist()
    indices.extend(target_indices)
    if sort_idx:
        indices = sorted(indices)

    assert len(np.unique(indices)) == len(indices)
    df = df.iloc[indices]

    if verbose:
        print(df[df.split == target_split][target].value_counts())
    return df

def train_prototypes(train_loader, val_loader, model, prototype_ensemble=(),
                     epochs=20, cov_reg=5e5, wd_weight=10, device='cuda',
                     entropic=30, lr=1e-3, stage=1, verbose=True, loss_name='isomax'):
    """
    Train a single prototype head using supervised data and optionally diversify from previous heads.

    Parameters
    ----------
    train_loader : DataLoader
        Loader for training samples (features only).
    val_loader : DataLoader
        Loader for validation data (used to select best prototype).
    model : torch.nn.Module
        Backbone + prototype classifier head.
    prototype_ensemble : list of tuple
        List of previous prototype tensors and distance scales.
    epochs : int
        Number of training epochs.
    cov_reg : float
        Weight on the covariance regularization term for inter-prototype diversity.
    wd_weight : float
        Weight decay multiplier for L2 penalty on prototypes.
    device : str
        CUDA or CPU device identifier.
    entropic : float
        Entropic scale parameter for IsoMax loss.
    lr : float
        Learning rate for SGD.
    stage : int
        Current stage in ensemble construction.
    verbose : bool
        Whether to show tqdm and metrics.
    loss_name : str
        Either 'isomax' or 'ce'.

    Returns
    -------
    tuple
        Best prototype head: (prototypes, distance_scale) for IsoMax or (weights, bias) for CE.
    """
    best_val_wga, val_wga = 0.0, 0.0
    best_val_wga_prototype = None

    if loss_name == 'isomax':
        criterion = IsoMaxPlusLossSecondPart(entropic_scale=entropic, reduction='mean')
    else:
        criterion = CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    if len(prototype_ensemble) > 0:
        prototype_ensemble = torch.concat([_[0] for _ in prototype_ensemble], dim=1).detach()

    for epoch in range(epochs):
        model.train()
        running_loss, running_clf, running_cov, running_correct, total = 0.0, 0.0, 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"[Stage {stage}] Epoch {epoch + 1}",
                    leave=(epoch == epochs - 1) and verbose)

        for _, _, labels, _, _, feats in pbar:
            feats = feats.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model[-1](feats)
            clf_loss = criterion(outputs, labels)

            head = model[-1]
            cov_loss = torch.tensor(0.0, device=device)

            if isinstance(criterion, IsoMaxPlusLossSecondPart):
                wd = torch.einsum('ijk,ilk->ijl', [head.prototypes[:, None], head.prototypes[:, None]]) * wd_weight
                wd = wd.squeeze().mean()
                loss = clf_loss + wd

                if len(prototype_ensemble) and (cov_reg > 0):
                    _prototypes = torch.cat([head.prototypes[:, None], prototype_ensemble], dim=1)
                    n_pro, n_dim = _prototypes.shape[1:]
                    cov = torch.einsum('ijk,ilk->ijl', [_prototypes, _prototypes]) / (n_dim - 1)
                    cov_loss = torch.abs(cov[:, 0, 1:].sum(1).div(n_pro).mean())
                    loss += cov_loss * cov_reg
            else:
                weight = head.weight  # if loss_name == 'ce'
                loss = clf_loss + 0.1 * torch.norm(weight, 1)

            loss.backward()
            optimizer.step()

            preds = outputs.argmax(dim=1)
            correct = (preds == labels).sum().item()
            running_loss += loss.item()
            running_clf += clf_loss.item()
            running_cov += cov_loss.item()
            running_correct += correct
            total += labels.size(0)

            pbar.set_postfix({
                'loss': running_loss / (total // labels.size(0)),
                'clf': running_clf / (total // labels.size(0)),
                'cov': running_cov / (total // labels.size(0)),
                'acc': f"{running_correct / total:.2%}",
                'val wga': f"{val_wga:.2%}"
            })

        val_wga = evaluate(model, val_loader)['min_group']['accuracy']
        if val_wga >= best_val_wga:
            best_val_wga = val_wga
            if loss_name == 'isomax':
                best_val_wga_prototype = [
                    model[-1].prototypes[:, None].detach().clone(),
                    model[-1].distance_scale.detach().clone()
                ]
            else:
                best_val_wga_prototype = [
                    model[-1].weight[:, None].detach().clone(),
                    model[-1].bias.detach().clone(),
                ]

    return best_val_wga_prototype
