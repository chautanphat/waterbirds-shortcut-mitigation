"""Post-lock evaluation only. Groups never cross the model/selection boundary."""
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


def summary(p, y, group=None):
    p = np.asarray(p); pred = p.argmax(1); correct = pred == y
    loss = -np.log(p[np.arange(len(y)), y].clip(1e-15))
    pc = [float(correct[y == k].mean()) if (y == k).any() else None for k in (0, 1)]
    nllc = [float(loss[y == k].mean()) for k in (0, 1) if (y == k).any()]
    conf = p.max(1); ece = 0.
    for mask in [(conf >= x/15) & (conf < (x+1)/15 if x < 14 else conf <= 1) for x in range(15)]:
        if mask.any(): ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    available = [v for v in pc if v is not None]
    r = dict(AA=float(correct.mean()), BA=float(np.mean(available)), worst_class_accuracy=min(available),
             worst_class_NLL=max(nllc), NLL=float(loss.mean()), ECE=float(ece),
             Brier=float(((p-np.eye(2)[y])**2).sum(1).mean()), count=len(y),
             class0_accuracy=pc[0], class1_accuracy=pc[1])
    if group is not None:
        for g in range(4):
            mask = group == g
            r[f'group{g}_count'] = int(mask.sum())
            r[f'group{g}_accuracy'] = float(correct[mask].mean()) if mask.any() else None
        r['WGA'] = min(r[f'group{g}_accuracy'] for g in range(4) if r[f'group{g}_count'])
    return r


def ranking(scores, correct):
    mixed = correct.any(1) & (~correct).any(1)
    values = []
    for s, c in zip(scores[mixed], correct[mixed]):
        gap = s[c][:, None] - s[~c][None]
        values.append(float(((gap > 0) + .5*(gap == 0)).mean()))
    return float(np.mean(values)) if values else None


def expert_stats(scores, weights, p, y):
    correct = p.argmax(-1) == y[:, None]
    c = correct.ravel(); s = scores.ravel()
    auc = float(roc_auc_score(c, s)) if len(np.unique(c)) == 2 and np.ptp(s) else None
    rho = float(spearmanr(s, c).statistic) if np.ptp(s) and len(np.unique(c)) == 2 else None
    return dict(pairwise_image_ranking=ranking(scores, correct), mixed_images=int((correct.any(1)&(~correct).any(1)).sum()),
                expert_AUROC=auc, expert_Spearman=rho,
                mean_weight_correct=float(weights[correct].mean()) if correct.any() else None,
                mean_weight_wrong=float(weights[~correct].mean()) if (~correct).any() else None,
                wrong_head_weight=float((weights*~correct).sum(1).mean()),
                mean_weight=float(weights.mean()), max_weight=float(weights.max()), mean_max_weight=float(weights.max(1).mean()),
                entropy=float(-(weights*np.log(weights.clip(1e-15))).sum(1).mean()),
                effective_members=float((1/(weights**2).sum(1)).mean()))


def paired_ci(p, base, y, groups, ids, repeats=500, seed=24019):
    assert len(set(ids)) == len(ids)
    c = p.argmax(1) == y; b = base.argmax(1) == y
    pools = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    rng = np.random.default_rng(seed)
    aa = []; wga = []
    for _ in range(repeats):
        takes = [rng.choice(pool, len(pool), replace=True) for pool in pools]
        idx = np.concatenate(takes)
        aa.append(float(c[idx].mean()-b[idx].mean())*100)
        wga.append(float(min(c[t].mean() for t in takes)-min(b[t].mean() for t in takes))*100)
    return dict(delta_AA_CI95_pp=np.quantile(aa, [.025,.975]).tolist(), delta_WGA_CI95_pp=np.quantile(wga,[.025,.975]).tolist(),
                CI_method='paired stratified distinct-image bootstrap, fixed models', bootstrap_repeats=repeats)


def evaluate(p, base, raw_base, member_p, weights, scores, y, groups, ids, repeats=500):
    r = summary(p,y,groups); u = summary(base,y,groups); raw = summary(raw_base,y,groups)
    r.update(uniform_AA=u['AA'], uniform_WGA=u['WGA'], raw_uniform_AA=raw['AA'], raw_uniform_WGA=raw['WGA'],
             delta_AA_pp=(r['AA']-u['AA'])*100, delta_WGA_pp=(r['WGA']-u['WGA'])*100,
             **expert_stats(scores,weights,member_p,y), **paired_ci(p,base,y,groups,ids,repeats))
    correct = p.argmax(1) == y; original = base.argmax(1) == y
    rows = []
    for kind, arr, keys in [('group',groups,range(4)),('class',y,range(2))]:
        for k in keys:
            mask = arr == k
            rows.append(dict(kind=kind, value=k, count=int(mask.sum()),
                             corrected=int((mask & correct & ~original).sum()), broken=int((mask & ~correct & original).sum())))
    r['corrected'] = int((correct & ~original).sum()); r['broken'] = int((~correct & original).sum())
    for a in (0,1):
        for b in (0,1): r[f'flip_{a}_to_{b}'] = int(((base.argmax(1)==a)&(p.argmax(1)==b)&(a!=b)).sum())
    return r, rows
