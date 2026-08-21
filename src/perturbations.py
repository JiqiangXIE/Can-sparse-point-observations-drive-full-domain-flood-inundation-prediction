import numpy as np

WET_OBS_THRESHOLD = 0.03

SCENARIOS = {
    "clean": dict(sigma=0.0, outlier_frac=0.0, outlier_amp=0.0,
                  missing=0.0, bias=0.0, max_shift=0),
    "mild": dict(sigma=0.02, outlier_frac=0.04, outlier_amp=0.15,
                 missing=0.10, bias=0.02, max_shift=0),
    "severe": dict(sigma=0.05, outlier_frac=0.04, outlier_amp=0.15,
                   missing=0.40, bias=0.05, max_shift=2),
}


def drop_sensors(h, rng, frac):
    if frac <= 0:
        return h, np.ones(h.shape[1], bool)
    keep = rng.random(h.shape[1]) >= frac
    if keep.sum() < 1:
        keep[rng.integers(h.shape[1])] = True
    return h[:, keep], keep


def time_shift(h, rng, max_shift):
    if max_shift <= 0:
        return h, np.zeros(h.shape[1], int)
    T, N = h.shape
    shift = rng.integers(-max_shift, max_shift + 1, N)
    out = np.empty_like(h)
    for j in range(N):
        s = shift[j]
        out[:, j] = np.roll(h[:, j], s)
        if s > 0:
            out[:s, j] = h[0, j]
        elif s < 0:
            out[s:, j] = h[-1, j]
    return out, shift


def add_bias(h, delta):
    if delta == 0:
        return h
    out = h.copy()
    wet = h > WET_OBS_THRESHOLD
    out[wet] = np.maximum(out[wet] + delta, 0.0)
    return out


def add_noise(h, rng, sigma, outlier_frac, outlier_amp):
    if sigma <= 0 and outlier_frac <= 0:
        return h
    out = h + rng.normal(0.0, sigma, h.shape) if sigma > 0 else h.copy()
    if outlier_frac > 0:
        hit = rng.random(h.shape) < outlier_frac
        out = out + hit * rng.uniform(-outlier_amp, outlier_amp, h.shape)
    return np.maximum(out, 0.0)


def apply_scenario(h_obs, obs_rc, scenario, seed=42):
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}")
    p = SCENARIOS[scenario]
    rng = np.random.default_rng(seed)

    h = np.asarray(h_obs, np.float32)
    h, keep = drop_sensors(h, rng, p["missing"])
    obs_rc = (np.ascontiguousarray(np.asarray(obs_rc[0])[keep]),
              np.ascontiguousarray(np.asarray(obs_rc[1])[keep]))
    h, _ = time_shift(h, rng, p["max_shift"])
    h = add_bias(h, p["bias"])
    h = add_noise(h, rng, p["sigma"], p["outlier_frac"], p["outlier_amp"])
    return h.astype(np.float32), obs_rc
