"""Small neural combiners; a project adaptation, not a paper reproduction."""
from contextlib import contextmanager

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .reweighting import GROUPS, combine, metrics, probabilities
from .weighting_rules import best_candidate, fit_all_rules, validate_config as validate_rules

FAMILIES = ('logistic', 'mlp', 'mlp_dropout', 'gate', 'gate_dropout')


def validate_config(c, heads):
    expected = {'version', 'heads', 'hidden_units', 'steps', 'learning_rate',
                'weight_decays', 'prediction_dropout', 'seed_offset', 'loss'}
    if not isinstance(c, dict) or set(c) not in (expected, expected | {'protocol'}, expected | {'protocol', 'weighting_rules'}):
        raise ValueError('Invalid neural-combiner configuration fields')
    if c.get('protocol', 'b_select_c') not in ('b_select_c', 'bc_all'):
        raise ValueError('Invalid neural protocol')
    if 'weighting_rules' in c:
        if c.get('protocol') != 'bc_all' or not isinstance(c['weighting_rules'], dict):
            raise ValueError('Global rules require the bc_all protocol and a rule configuration')
        validate_rules(c['weighting_rules'], heads)
    if (type(c['version']) is not int or c['version'] != 1 or
            type(c['heads']) is not int or c['heads'] != heads or heads != 15 or
            type(c['hidden_units']) is not int or not 1 <= c['hidden_units'] <= 32 or
            type(c['steps']) is not int or c['steps'] < 1 or
            type(c['seed_offset']) is not int or c['seed_offset'] < 0 or
            c['loss'] != 'group_balanced_bce'):
        raise ValueError('Unsupported neural-combiner contract')
    for name in ('learning_rate', 'prediction_dropout'):
        v = c[name]
        if type(v) not in (int, float) or not np.isfinite(v) or not 0 < v < 1:
            raise ValueError(f'Invalid {name}')
    values = c['weight_decays']
    if (not isinstance(values, list) or not values or
            any(type(v) not in (int, float) or not np.isfinite(v) or v < 0 for v in values) or
            len(set(values)) != len(values)):
        raise ValueError('Invalid weight_decays')


@contextmanager
def cpu_randomness(seed):
    """Isolate initialization/dropout RNG and avoid threaded tiny-matrix overhead."""
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            yield
    finally:
        torch.set_num_threads(threads)


def prediction_mask(q, rate, generator):
    """Drop an entire head prediction per example; always retain at least one."""
    keep = torch.rand(q.shape, generator=generator) >= rate
    empty = torch.where(~keep.any(dim=1))[0]
    if len(empty):
        keep[empty, torch.randint(q.shape[1], (len(empty),), generator=generator)] = True
    return keep


class Combiner(nn.Module):
    def __init__(self, kind, heads, hidden):
        super().__init__()
        if kind not in ('logistic', 'mlp', 'gate'):
            raise ValueError('Unknown combiner kind')
        self.kind, self.heads, self.hidden = kind, heads, hidden
        width = heads if kind == 'gate' else 1
        self.network = (nn.Linear(heads, 1) if kind == 'logistic' else
                        nn.Sequential(nn.Linear(heads, hidden), nn.Tanh(), nn.Linear(hidden, width)))
        self.double()
        # Gates begin exactly uniform; stackers begin at probability 0.5.
        last = self.network if kind == 'logistic' else self.network[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, q, keep=None):
        # q: N x H waterbird probabilities. Neutral replacement, no inverted scaling.
        inputs = 2 * q - 1
        if keep is not None:
            inputs = inputs.masked_fill(~keep, 0)
        logits = self.network(inputs)
        if self.kind == 'gate':
            if keep is not None:
                logits = logits.masked_fill(~keep, -torch.inf)
            w = logits.softmax(dim=1)
            return (w * q).sum(dim=1), w
        return logits.squeeze(1).sigmoid(), None


def tensor_inputs(p):
    p = probabilities(p)
    if p.ndim != 3:
        raise ValueError('Expected head predictions [H,N,2]')
    return torch.from_numpy(np.ascontiguousarray(p[:, :, 1].T))


def group_loss(q, y, groups):
    loss = F.binary_cross_entropy(q.clamp(1e-12, 1 - 1e-12), y, reduction='none')
    return torch.stack([loss[groups == i].mean() for i in range(4)]).mean()


def model_record(model):
    return dict(kind=model.kind, heads=model.heads, hidden_units=model.hidden,
                state={name: value.detach().tolist() for name, value in model.state_dict().items()})


def predict(record, p):
    """Data-only checkpoint; never unpickle code or train at inference."""
    q = tensor_inputs(p)
    if q.shape[1] != record['heads']:
        raise ValueError('Combiner/head count mismatch')
    with cpu_randomness(0):
        model = Combiner(record['kind'], record['heads'], record['hidden_units'])
        state = {k: torch.tensor(v, dtype=torch.float64) for k, v in record['state'].items()}
        if any(not torch.isfinite(v).all() for v in state.values()):
            raise ValueError('Nonfinite combiner checkpoint')
        model.load_state_dict(state, strict=True)
        model.eval()
        with torch.no_grad():
            result, w = model(q)
    result = result.numpy()
    return probabilities(np.stack([1 - result, result], axis=-1)), None if w is None else w.numpy()


def fit(p_b, y_b, g_b, config, kind, weight_decay, dropout, seed):
    """Fixed-step, full-batch CPU float64 AdamW on supplied training data only."""
    q = tensor_inputs(p_b)
    for head in p_b:
        metrics(head, y_b, g_b)
    y = torch.tensor(y_b, dtype=torch.float64)
    groups = torch.tensor([GROUPS.index(g) for g in g_b])
    with cpu_randomness(seed):
        generator = torch.Generator().manual_seed(seed + 1)
        model = Combiner(kind, q.shape[1], config['hidden_units'])
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],
                                      weight_decay=weight_decay)
        history = []
        for step in range(config['steps'] + 1):
            if step % 100 == 0 or step == config['steps']:
                with torch.no_grad():
                    history.append(dict(step=step, b_group_balanced_bce=float(group_loss(model(q)[0], y, groups))))
            if step == config['steps']:
                break
            model.train()
            keep = prediction_mask(q, dropout, generator) if dropout else None
            predicted, _ = model(q, keep)
            loss = group_loss(predicted, y, groups)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite neural training loss; no candidate accepted')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        record = model_record(model)
    return record, dict(seed=seed, steps=config['steps'], history=history,
                        parameter_count=sum(p.numel() for p in model.parameters()))


def diagnostics(p, y, groups, w=None):
    """NLL alongside WGA; effective head counts apply only to convex mixtures."""
    p = probabilities(p)
    m = metrics(p, y, groups)
    nll = -np.log(np.maximum(p[np.arange(len(y)), np.asarray(y, dtype=int)], 1e-12))
    result = dict(metrics=m, group_nll={g: float(nll[np.asarray(groups) == g].mean()) for g in GROUPS})
    if w is not None:
        effective = 1 / np.square(w).sum(axis=1)
        result['gating'] = dict(mean_weights=w.mean(axis=0).tolist(),
                              mean_effective_heads=float(effective.mean()),
                              min_effective_heads=float(effective.min()),
                              max_effective_heads=float(effective.max()),
                              mean_max_weight=float(w.max(axis=1).mean()),
                              group_mean_weights={g: w[np.asarray(groups) == g].mean(axis=0).tolist() for g in GROUPS})
    return result


def complementarity(p, y, groups):
    """An unattainable label-aware head oracle diagnoses room for convex gating."""
    correct = p.argmax(axis=-1) == np.asarray(y)[None, :]
    uniform = combine(p, np.full(len(p), 1 / len(p))).argmax(axis=-1) == y
    oracle = correct.any(axis=0)
    return dict(oracle_is_not_a_predictor=True,
                oracle_wga=min(float(oracle[np.asarray(groups) == g].mean()) for g in GROUPS),
                uniform_errors_with_correct_head=int((~uniform & oracle).sum()),
                all_heads_wrong=int((~oracle).sum()))


def candidate_prediction(row, p):
    if 'weights' in row:
        w = np.asarray(row['weights'], dtype=float)
        return combine(p, w), np.broadcast_to(w, (p.shape[1], len(w)))
    if row['family'] == 'uniform':
        return combine(p, np.full(len(p), 1 / len(p))), np.full((p.shape[1], len(p)), 1 / len(p))
    return predict(row['model'], p)


def fit_select(p_b, y_b, g_b, p_c, y_c, g_c, config, source_seed):
    p_b, p_c = probabilities(p_b), probabilities(p_c)
    if p_b.ndim != 3 or p_c.ndim != 3 or len(p_b) != len(p_c):
        raise ValueError('Expected matching head axes')
    validate_config(config, len(p_b))
    if type(source_seed) is not int or source_seed not in (0, 1, 2):
        raise ValueError('Expected source training seed 0, 1 or 2')
    for p, y, g in ((p_b, y_b, g_b), (p_c, y_c, g_c)):
        for head in p:
            metrics(head, y, g)
    uniform = combine(p_c, np.full(len(p_c), 1 / len(p_c)))
    rows = []

    def add(row):
        b, wb = candidate_prediction(row, p_b)
        c, wc = candidate_prediction(row, p_c)
        row.update(b=diagnostics(b, y_b, g_b, wb), c=diagnostics(c, y_c, g_c, wc),
                   metrics=metrics(c, y_c, g_c),
                   c_prediction_changes=int((c.argmax(axis=1) != uniform.argmax(axis=1)).sum()),
                   c_probability_max_abs_delta=float(np.abs(c - uniform).max()))
        rows.append(row)

    add(dict(id='uniform', family='uniform', parameters={}, model=None, fit=None))
    for family in FAMILIES:
        kind = family.split('_')[0]
        dropout = config['prediction_dropout'] if family.endswith('_dropout') else 0.
        for wd in config['weight_decays']:
            model, fitted = fit(p_b, y_b, g_b, config, kind, wd, dropout, source_seed + config['seed_offset'])
            add(dict(id=f'{family}-wd{wd}', family=family,
                     parameters=dict(weight_decay=wd, prediction_dropout=dropout), model=model, fit=fitted))
    return dict(candidates=rows, selected=best_candidate(rows)['id'],
                per_family_selected={family: best_candidate([r for r in rows if r['family'] in ('uniform', family)])['id']
                                     for family in FAMILIES},
                complementarity={role: complementarity(p, y, g) for role, p, y, g in
                                 [('B', p_b, y_b, g_b), ('C', p_c, y_c, g_c)]})


def fit_all(p, y, g, config, source_seed):
    """Fit every declared configuration on pooled B+C; no model selection."""
    validate_config(config, len(p))
    if type(source_seed) is not int or source_seed not in (0, 1, 2):
        raise ValueError('Expected source training seed 0, 1 or 2')
    rows = []

    def add(row):
        prediction, weights = candidate_prediction(row, p)
        row['training'] = diagnostics(prediction, y, g, weights)
        rows.append(row)

    add(dict(id='uniform', family='uniform', parameters={}, model=None, fit=None))
    for family in FAMILIES:
        dropout = config['prediction_dropout'] if family.endswith('_dropout') else 0.
        for wd in config['weight_decays']:
            model, fitted = fit(p, y, g, config, family.split('_')[0], wd, dropout,
                                source_seed + config['seed_offset'])
            for item in fitted['history']:
                item['training_group_balanced_bce'] = item.pop('b_group_balanced_bce')
            add(dict(id=f'{family}-wd{wd}', family=family,
                     parameters=dict(weight_decay=wd, prediction_dropout=dropout), model=model, fit=fitted))
    if 'weighting_rules' in config:
        for row in fit_all_rules(p, y, g, config['weighting_rules']):
            if row['id'] != 'uniform':  # Share one matched uniform reference.
                add(row)
    return dict(protocol='bc_all', training_role='B+C', training_count=len(y), candidates=rows,
                complementarity={'B+C': complementarity(p, y, g)})
