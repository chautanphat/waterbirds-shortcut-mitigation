"""Independent NumPy inference from data-only fit records; no torch model calls."""
import numpy as np
from scipy.special import expit, logsumexp, softmax


def adam_prediction(record, inputs):
    x = np.asarray(inputs, dtype=np.float64)
    state = {k: np.asarray(v, dtype=np.float64) for k, v in record["state"].items()}
    if record["family"] == "prototype":
        t = np.exp(state["log_temperature"]) if "log_temperature" in state else record["temperature"]
        alpha = state["theta"]-logsumexp(state["theta"], axis=0) if "theta" in state else np.full((15, 2), -np.log(15))
        evidence = logsumexp(x/t+alpha[:, None, :], axis=0)
        logit = evidence[:, 1]-evidence[:, 0]+state.get("bias", 0.)
    else:
        mean = np.clip(x.mean(1), 1e-12, 1-1e-12)
        logit = state["a"]*(np.log(mean)-np.log1p(-mean))+state["b"]
        d = x-x.mean(1, keepdims=True)
        if record["kind"] == "linear":
            logit += (d @ state["correction.weight"].T)[:, 0]
        elif record["kind"] == "nonlinear":
            hidden = np.tanh(d @ state["correction.0.weight"].T+state["correction.0.bias"])
            zero = np.tanh(state["correction.0.bias"])
            logit += ((hidden-zero) @ state["correction.2.weight"].T)[:, 0]
    q = expit(logit)
    return np.stack([1-q, q], axis=-1)


def anchor_prediction(record, p):
    x = 2*np.asarray(p, dtype=np.float64)[:, :, 1].T-1
    s = {k: np.asarray(v) for k, v in record["state"].items()}
    if record["kind"] == "logistic":
        logit = x @ s["network.weight"].T+s["network.bias"]
    else:
        h = np.tanh(x @ s["network.0.weight"].T+s["network.0.bias"])
        logit = h @ s["network.2.weight"].T+s["network.2.bias"]
    q = expit(logit[:, 0])
    return np.stack([1-q, q], axis=-1)


def geometry_prediction(z, spec):
    """Separate grouping implementation for geometry replay."""
    if spec["family"] == "uniform":
        return softmax(z, axis=-1).mean(0)
    z = np.asarray(z, dtype=float)/spec["temperature"]
    if spec["family"] == "local":
        selected = np.sort(z, axis=0)[-spec["k"]:]
        return softmax(logsumexp(selected, axis=0)-np.log(spec["k"]), axis=1)
    order = np.arange(15) if spec["membership"] == "cyclic" else np.random.RandomState(0).permutation(15)
    heads = []
    for i in range(15):
        subset = np.stack([z[order[(i+r)%15]] for r in range(spec["k"])])
        heads.append(softmax(logsumexp(subset, axis=0), axis=1))
    return np.mean(heads, axis=0)
