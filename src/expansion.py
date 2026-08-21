import numpy as np

from . import asff


def expand(h_obs_t, obs_rc, dem, dx, method="as_floodfill", slope_scale=1.0):
    obs_r = np.asarray(obs_rc[0])
    obs_c = np.asarray(obs_rc[1])
    dem = np.asarray(dem, np.float32)
    H, W = dem.shape

    if method not in ("as_floodfill", "floodfill"):
        raise ValueError(f"method must be as_floodfill or floodfill, got {method!r}")

    field = asff.as_floodfill(
        np.asarray(h_obs_t, np.float32), (obs_r, obs_c), dem, H, W, float(dx),
        adaptive=(method == "as_floodfill"), reduce=asff.REDUCTION,
        slope_scale=slope_scale)

    field[obs_r, obs_c] = np.maximum(h_obs_t, 0.0)
    assert np.isfinite(field).all(), "expansion produced NaN or inf"
    return field.astype(np.float32)


def expand_sequence(h_obs, obs_rc, dem, dx, method="as_floodfill",
                    slope_scale=1.0, n_jobs=1):
    T = h_obs.shape[0]
    H, W = dem.shape

    if n_jobs > 1 and T > 1:
        try:
            from joblib import Parallel, delayed
        except ImportError:
            n_jobs = 1
    if n_jobs > 1 and T > 1:
        bounds = np.linspace(0, T, min(n_jobs, T) + 1).astype(int)
        blocks = Parallel(n_jobs=min(n_jobs, T))(
            delayed(expand_sequence)(h_obs[a:b], obs_rc, dem, dx, method,
                                     slope_scale, 1)
            for a, b in zip(bounds[:-1], bounds[1:]) if b > a)
        return np.concatenate(blocks, axis=0)

    out = np.zeros((T, H, W), np.float32)
    for t in range(T):
        out[t] = expand(h_obs[t], obs_rc, dem, dx, method, slope_scale)
    return out
