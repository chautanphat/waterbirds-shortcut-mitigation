"""Fitting-only entry points. No test inputs, early stopping or candidate selection."""
import numpy as np
import torch
from scipy.optimize import linprog, minimize
from torch import nn
from torch.nn import functional as F

from src.methods.dpe_head_reweighting.neural_combiner import cpu_randomness
from src.methods.dpe_head_reweighting.reweighting import GROUPS, metrics, probabilities


def group_losses(values, groups):
    return torch.stack([values[groups == i].mean() for i in range(4)])


def reduce_groups(values, groups, objective):
    losses = group_losses(values, groups)
    if objective == "average":
        return losses.mean()
    if objective == "maximum":
        return losses.max()
    raise ValueError("Unknown group objective")


class PrototypeModel(nn.Module):
    def __init__(self, kind, temperature, config):
        super().__init__()
        if kind not in ("importance", "temperature", "bias", "temperature_bias"):
            raise ValueError("Unknown prototype model")
        self.kind, self.base_temperature = kind, temperature
        self.temperature_bounds = config["temperature_bounds"]
        self.bias_bounds = config["bias_bounds"]
        if kind == "importance":
            self.theta = nn.Parameter(torch.zeros(15, 2, dtype=torch.float64))
        if "temperature" in kind:
            self.log_temperature = nn.Parameter(torch.tensor(np.log(temperature), dtype=torch.float64))
        if "bias" in kind:
            self.bias = nn.Parameter(torch.zeros((), dtype=torch.float64))

    def logits(self, z):
        temperature = self.log_temperature.exp() if hasattr(self, "log_temperature") else self.base_temperature
        log_alpha = self.theta.log_softmax(0) if hasattr(self, "theta") else -np.log(15)
        evidence = torch.logsumexp(z / temperature + (log_alpha[:, None, :] if hasattr(self, "theta") else log_alpha), dim=0)
        logit = evidence[:, 1] - evidence[:, 0]
        return logit + (self.bias if hasattr(self, "bias") else 0.)

    def regularizer(self):
        if hasattr(self, "theta"):
            log_alpha = self.theta.log_softmax(0)
            return (log_alpha.exp() * (log_alpha + np.log(15))).sum(0).mean()
        result = torch.zeros((), dtype=torch.float64)
        if hasattr(self, "log_temperature"):
            result = result + (self.log_temperature - np.log(self.base_temperature)).square()
        if hasattr(self, "bias"):
            result = result + self.bias.square()
        return result

    def constrain(self):
        with torch.no_grad():
            if hasattr(self, "log_temperature"):
                self.log_temperature.clamp_(*np.log(self.temperature_bounds))
            if hasattr(self, "bias"):
                self.bias.clamp_(*self.bias_bounds)


class ResidualModel(nn.Module):
    def __init__(self, kind, heads=15, hidden=8):
        super().__init__()
        if kind not in ("affine", "linear", "nonlinear"):
            raise ValueError("Unknown residual model")
        self.kind, self.heads, self.hidden = kind, heads, hidden
        self.a = nn.Parameter(torch.ones((), dtype=torch.float64))
        self.b = nn.Parameter(torch.zeros((), dtype=torch.float64))
        if kind == "linear":
            self.correction = nn.Linear(heads, 1, bias=False).double()
            nn.init.zeros_(self.correction.weight)
        elif kind == "nonlinear":
            self.correction = nn.Sequential(nn.Linear(heads, hidden), nn.Tanh(), nn.Linear(hidden, 1, bias=False)).double()
            nn.init.zeros_(self.correction[-1].weight)

    def logits(self, q):
        mean = q.mean(1).clamp(1e-12, 1-1e-12)
        result = self.a * torch.logit(mean) + self.b
        if hasattr(self, "correction"):
            disagreement = q - q.mean(1, keepdim=True)
            result = result + (self.correction(disagreement) - self.correction(torch.zeros_like(disagreement))).squeeze(1)
        return result

    def regularizer(self):
        result = (self.a-1).square() + self.b.square()
        if hasattr(self, "correction"):
            result = result + torch.cat([p.reshape(-1) for p in self.correction.parameters()]).square().mean()
        return result

    def constrain(self):
        pass


def fit_adam(inputs, y, groups, spec, config, seed):
    """Fixed 400-step final fit, with data-only state and full-group loss traces."""
    y, groups = np.asarray(y), np.asarray(groups)
    metrics(np.full((len(y), 2), .5), y, groups)
    x = torch.tensor(np.asarray(inputs), dtype=torch.float64)
    target = torch.tensor(y, dtype=torch.float64)
    g = torch.tensor([GROUPS.index(value) for value in groups])
    if not torch.isfinite(x).all() or (spec["family"] == "prototype" and x.shape != (15, len(y), 2)):
        raise ValueError("Invalid fitting inputs")
    with cpu_randomness(seed):
        model = (PrototypeModel(spec["kind"], spec["temperature"], config) if spec["family"] == "prototype"
                 else ResidualModel(spec["kind"], x.shape[1], config["hidden_units"]))
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
        history = []
        for step in range(config["steps"] + 1):
            logits = model.logits(x)
            per_example = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
            data_loss = reduce_groups(per_example, g, spec["loss"])
            objective = data_loss + spec["penalty"] * model.regularizer()
            if not torch.isfinite(objective) or any(not torch.isfinite(p).all() for p in model.parameters()):
                raise ValueError("Nonfinite fit; no candidate accepted")
            if step % 100 == 0 or step == config["steps"]:
                history.append(dict(step=step, objective=float(objective.detach()),
                                    group_bce=group_losses(per_example, g).detach().tolist()))
            if step == config["steps"]:
                break
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            optimizer.step()
            model.constrain()
        state = {name: value.detach().tolist() for name, value in model.state_dict().items()}
    return dict(family=spec["family"], kind=spec["kind"], state=state,
                temperature=spec.get("temperature"), heads=15, hidden=config["hidden_units"],
                seed=seed, steps=config["steps"], history=history)


def predict_adam(record, inputs, config):
    with cpu_randomness(0):
        model = (PrototypeModel(record["kind"], record["temperature"], config) if record["family"] == "prototype"
                 else ResidualModel(record["kind"], record["heads"], record["hidden"]))
        model.load_state_dict({k: torch.tensor(v, dtype=torch.float64) for k, v in record["state"].items()}, strict=True)
        with torch.inference_mode():
            q = model.logits(torch.tensor(np.asarray(inputs), dtype=torch.float64)).sigmoid().numpy()
    return probabilities(np.stack([1-q, q], axis=-1))


def brier_certificate(w, losses, gradients, penalty, objective):
    """Global lower bound from a supporting hyperplane on the simplex."""
    regularizer_gradient = 2 * penalty * (w - 1/len(w))
    if objective == "average":
        mixture = np.full(len(losses), 1/len(losses))
    else:
        # Find the strongest supporting-plane bound over group multipliers.
        costs = np.r_[gradients @ w - losses, -1.]
        fit = linprog(costs, A_ub=np.c_[-gradients.T, np.ones(len(w))],
                      b_ub=regularizer_gradient, A_eq=np.array([[1., 1., 1., 1., 0.]]),
                      b_eq=[1.], bounds=[(0, None)]*4+[(None, None)], method="highs",
                      options=dict(dual_feasibility_tolerance=1e-9, primal_feasibility_tolerance=1e-9))
        if not fit.success:
            raise ValueError("Cannot certify Brier fit")
        mixture = fit.x[:4]
    gradient = mixture @ gradients + regularizer_gradient
    value = losses.mean() if objective == "average" else losses.max()
    if penalty > 0:
        # Retain the exact quadratic shrinkage when minimizing the supporting
        # plane. A first-order-only bound can be loose at a valid SLSQP solution.
        g = mixture @ gradients
        u = np.full(len(w), 1/len(w))
        v = u-g/(2*penalty)
        ordered = np.sort(v)[::-1]
        thresholds = (ordered.cumsum()-1)/np.arange(1, len(w)+1)
        rho = np.flatnonzero(ordered > thresholds)[-1]
        multiplier = 2*penalty*thresholds[rho]
        primal = np.maximum(u-(g+multiplier)/(2*penalty), 0)
        dual_min = np.sum((g+multiplier)*primal+penalty*np.square(primal-u))-multiplier
        lower = mixture @ losses-g @ w+dual_min
        gap = float(value+penalty*np.square(w-u).sum()-lower)
    else:
        gap = float(value - mixture @ losses + gradient @ w - gradient.min())
    if not np.isfinite(gap) or gap < -1e-10:
        raise ValueError("Invalid convex lower-bound certificate")
    return max(0., gap)


def fit_brier(p, y, groups, penalty, objective, solver):
    p = probabilities(p)
    for head in p:
        metrics(head, y, groups)
    if objective not in ("average", "maximum") or penalty < 0:
        raise ValueError("Invalid Brier objective")
    q, y, groups = p[:, :, 1].T, np.asarray(y), np.asarray(groups)
    h = p.shape[0]
    u = np.full(h, 1/h)

    def components(w):
        error = q @ w - y
        loss = np.array([np.square(error[groups == g]).mean() for g in GROUPS])
        grad = np.array([2*q[groups == g].T @ error[groups == g] / sum(groups == g) for g in GROUPS])
        return loss, grad

    def fun(v):
        loss, grad = components(v[:h])
        if objective == "maximum":
            return v[-1] + penalty*np.square(v[:h]-u).sum()
        return loss.mean() + penalty*np.square(v-u).sum()

    def jac(v):
        loss, grad = components(v[:h])
        reg = 2*penalty*(v[:h]-u)
        return np.r_[reg, 1.] if objective == "maximum" else grad.mean(0)+reg

    constraints = [dict(type="eq", fun=lambda v: v[:h].sum()-1,
                        jac=lambda v: np.r_[np.ones(h), 0.] if objective == "maximum" else np.ones(h))]
    initial = u
    if objective == "maximum":
        initial = np.r_[u, components(u)[0].max()]
        constraints.append(dict(type="ineq", fun=lambda v: v[-1]-components(v[:h])[0],
                                jac=lambda v: np.c_[-components(v[:h])[1], np.ones(4)]))
    bounds = [(0., 1.)]*h + ([(0., None)] if objective == "maximum" else [])
    attempts = []
    start = initial
    for attempt_index in range(2):
        result = minimize(fun, start, jac=jac, constraints=constraints, bounds=bounds,
                          method="SLSQP", options=dict(maxiter=solver["maxiter"],
                          ftol=solver["ftol"] if attempt_index == 0 else min(solver["ftol"], 1e-14)))
        attempts.append(dict(success=bool(result.success), message=str(result.message), iterations=int(result.nit)))
        v = result.x
        tol = solver["feasibility_tolerance"]
        feasible = (np.isfinite(v).all() and abs(v[:h].sum()-1) <= tol and v[:h].min() >= -tol
                    and all(np.min(c["fun"](v)) >= -tol for c in constraints[1:]))
        if feasible:
            w = np.maximum(v[:h], 0); w /= w.sum()
            loss, grad = components(w)
            gap = brier_certificate(w, loss, grad, penalty, objective)
            attempts[-1]["objective_gap_bound"] = gap
            value = (loss.mean() if objective == "average" else loss.max()) + penalty*np.square(w-u).sum()
            if gap <= tol and value <= fun(initial)+tol:
                return dict(weights=w.tolist(), objective=float(value), objective_gap_bound=gap,
                            group_brier=dict(zip(GROUPS, loss.tolist())), attempts=attempts)
        start = v
    raise ValueError(f"Brier fit failed feasibility/optimality checks: {attempts}")
