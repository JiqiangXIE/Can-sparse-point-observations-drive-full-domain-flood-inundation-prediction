import numpy as np
from scipy.spatial import cKDTree

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    def njit(*a, **k):
        def deco(f):
            return f
        return deco if not a else a[0]


WET_OBS_THRESHOLD = 0.03
MIN_WET_OBS = 5
MAX_SLOPE = 0.01
MIN_SLOPE = 1e-5
DEFAULT_SLOPE = 1e-3
IDW_POWER = 2.0
K_NEIGHBORS = 8
WSE_MARGIN = 2.0
REDUCTION = "near"

_DR = np.array([-1, -1, -1, 0, 0, 1, 1, 1], np.int32)
_DC = np.array([-1, 0, 1, -1, 1, -1, 0, 1], np.int32)
_DD = np.array([1.41421356, 1.0, 1.41421356,
                1.0, 1.0,
                1.41421356, 1.0, 1.41421356], np.float64)


@njit(cache=True, fastmath=True)
def _dijkstra_fill(seed_r, seed_c, seed_wse, S, dem, H, W, DX, dr, dc, dd):
    NC = H * W
    wse = np.full(NC, -1e18, np.float64)
    done = np.zeros(NC, np.uint8)

    cap = 8 * NC + len(seed_r) + 64
    hk = np.empty(cap, np.float64)
    hi = np.empty(cap, np.int32)
    hn = 0

    for i in range(len(seed_r)):
        idx = seed_r[i] * W + seed_c[i]
        v = seed_wse[i]
        if v > wse[idx]:
            wse[idx] = v
            hk[hn] = v
            hi[hn] = idx
            j = hn
            hn += 1
            while j > 0:
                p = (j - 1) >> 1
                if hk[p] < hk[j]:
                    hk[p], hk[j] = hk[j], hk[p]
                    hi[p], hi[j] = hi[j], hi[p]
                    j = p
                else:
                    break

    while hn > 0:
        top_k = hk[0]
        top_i = hi[0]
        hn -= 1
        hk[0] = hk[hn]
        hi[0] = hi[hn]
        j = 0
        while True:
            l = 2 * j + 1
            r = l + 1
            big = j
            if l < hn and hk[l] > hk[big]:
                big = l
            if r < hn and hk[r] > hk[big]:
                big = r
            if big == j:
                break
            hk[big], hk[j] = hk[j], hk[big]
            hi[big], hi[j] = hi[j], hi[big]
            j = big

        if done[top_i] == 1:
            continue
        if top_k < wse[top_i] - 1e-9:
            continue
        done[top_i] = 1

        r0 = top_i // W
        c0 = top_i - r0 * W
        w0 = wse[top_i]

        for k in range(8):
            nr = r0 + dr[k]
            nc = c0 + dc[k]
            if nr < 0 or nr >= H or nc < 0 or nc >= W:
                continue
            nidx = nr * W + nc
            if done[nidx] == 1:
                continue
            w1 = w0 - S[nidx] * DX * dd[k]
            if w1 <= dem[nidx]:
                continue
            if w1 > wse[nidx] + 1e-6:
                wse[nidx] = w1
                hk[hn] = w1
                hi[hn] = nidx
                j = hn
                hn += 1
                while j > 0:
                    p = (j - 1) >> 1
                    if hk[p] < hk[j]:
                        hk[p], hk[j] = hk[j], hk[p]
                        hi[p], hi[j] = hi[j], hi[p]
                        j = p
                    else:
                        break

    filled = np.zeros(NC, np.uint8)
    for i in range(NC):
        if wse[i] > -1e17:
            filled[i] = 1
    return wse, filled


@njit(cache=True, fastmath=True)
def _dijkstra_near(seed_r, seed_c, seed_wse, S, foot, H, W, DX,
                   dr, dc, dd, cap):
    NC = H * W
    cost = np.full(NC, 1e18, np.float64)
    swse = np.full(NC, -1e18, np.float64)
    done = np.zeros(NC, np.uint8)
    err = np.zeros(1, np.int8)

    hk = np.empty(cap, np.float64)
    hi = np.empty(cap, np.int32)
    hn = 0

    for i in range(len(seed_r)):
        idx = seed_r[i] * W + seed_c[i]
        cost[idx] = 0.0
        swse[idx] = seed_wse[i]
        hk[hn] = 0.0
        hi[hn] = idx
        j = hn
        hn += 1
        while j > 0:
            p = (j - 1) >> 1
            if hk[p] > hk[j]:
                hk[p], hk[j] = hk[j], hk[p]
                hi[p], hi[j] = hi[j], hi[p]
                j = p
            else:
                break

    while hn > 0:
        c0 = hk[0]
        top = hi[0]
        hn -= 1
        hk[0] = hk[hn]
        hi[0] = hi[hn]
        j = 0
        while True:
            l = 2 * j + 1
            r = l + 1
            sm = j
            if l < hn and hk[l] < hk[sm]:
                sm = l
            if r < hn and hk[r] < hk[sm]:
                sm = r
            if sm == j:
                break
            hk[sm], hk[j] = hk[j], hk[sm]
            hi[sm], hi[j] = hi[j], hi[sm]
            j = sm

        if done[top] == 1:
            continue
        if c0 > cost[top] + 1e-12:
            continue
        done[top] = 1

        r0 = top // W
        c_0 = top - r0 * W
        for k in range(8):
            nr = r0 + dr[k]
            nc = c_0 + dc[k]
            if nr < 0 or nr >= H or nc < 0 or nc >= W:
                continue
            nidx = nr * W + nc
            if foot[nidx] == 0:
                continue
            if done[nidx] == 1:
                continue
            c1 = c0 + S[nidx] * DX * dd[k]
            if c1 < cost[nidx] - 1e-12:
                cost[nidx] = c1
                swse[nidx] = swse[top]
                if hn >= cap:
                    err[0] = 1
                    return cost, swse, err
                hk[hn] = c1
                hi[hn] = nidx
                j = hn
                hn += 1
                while j > 0:
                    p = (j - 1) >> 1
                    if hk[p] > hk[j]:
                        hk[p], hk[j] = hk[j], hk[p]
                        hi[p], hi[j] = hi[j], hi[p]
                        j = p
                    else:
                        break

    return cost, swse, err


def _fill_max(seed_r, seed_c, seed_wse, S, dem, H, W, DX):
    wse, filled = _dijkstra_fill(
        seed_r.astype(np.int32), seed_c.astype(np.int32),
        seed_wse.astype(np.float64),
        np.ascontiguousarray(S, np.float64).ravel(),
        np.ascontiguousarray(dem, np.float64).ravel(),
        int(H), int(W), float(DX), _DR, _DC, _DD)
    return wse.reshape(H, W), filled.reshape(H, W).astype(bool)


def _fill_near(seed_r, seed_c, seed_wse, S, dem, H, W, DX):
    S64 = np.ascontiguousarray(S, np.float64).ravel()
    dem64 = np.ascontiguousarray(dem, np.float64).ravel()
    assert S64.min() > 0.0, "slope field has non-positive entries"

    wse_max, filled = _dijkstra_fill(
        seed_r.astype(np.int32), seed_c.astype(np.int32),
        seed_wse.astype(np.float64), S64, dem64,
        int(H), int(W), float(DX), _DR, _DC, _DD)
    foot = (filled == 1) & (wse_max > dem64)
    foot_u8 = np.ascontiguousarray(foot.astype(np.uint8))
    n_foot = int(foot_u8.sum())

    cap = 8 * n_foot + len(seed_r) + 64
    cost, swse, err = _dijkstra_near(
        seed_r.astype(np.int32), seed_c.astype(np.int32),
        seed_wse.astype(np.float64), S64, foot_u8,
        int(H), int(W), float(DX), _DR, _DC, _DD, int(cap))
    assert err[0] == 0, "heap overflow in nearest-source pass"

    reach = cost < 1e17
    assert not (foot & ~reach).any(), "footprint cell unreachable in pass B"

    wse = np.full(H * W, -1e18, np.float64)
    wse[reach] = swse[reach] - cost[reach]
    return wse.reshape(H, W), reach.reshape(H, W)


_IDW_CACHE = {}
_IDW_CACHE_MAX = 8


def _idw_grid(vpts, H, W, DX, power=IDW_POWER, k=K_NEIGHBORS):
    key = (H, W, DX, power, k, vpts.tobytes())
    if key in _IDW_CACHE:
        return _IDW_CACHE[key]

    gr, gc = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    grid = np.stack([gr.ravel(), gc.ravel()], 1).astype(np.float64) * DX

    kk = min(k, len(vpts))
    gd, gi = cKDTree(vpts).query(grid, k=kk)
    if kk == 1:
        gd = gd[:, None]
        gi = gi[:, None]
    w = 1.0 / np.maximum(gd, DX * 0.5) ** power
    w /= w.sum(1, keepdims=True)

    w = w.astype(np.float32)
    gi = gi.astype(np.int32)

    if len(_IDW_CACHE) >= _IDW_CACHE_MAX:
        _IDW_CACHE.pop(next(iter(_IDW_CACHE)))
    _IDW_CACHE[key] = (w, gi)
    return w, gi


def adaptive_slope_field(obs_rc, wse_obs, wet, H, W, DX,
                         k=K_NEIGHBORS, power=IDW_POWER):
    wr = obs_rc[0][wet]
    wc = obs_rc[1][wet]
    wse = wse_obs[wet].astype(np.float64)
    n = len(wse)

    if n < 2:
        return np.full((H, W), DEFAULT_SLOPE, np.float32)

    pts = np.stack([wr, wc], 1).astype(np.float64) * DX
    kq = min(k + 1, n)
    dists, idxs = cKDTree(pts).query(pts, k=kq)
    if dists.ndim == 1:
        dists = dists[:, None]
        idxs = idxs[:, None]

    d = dists[:, 1:]
    nb = idxs[:, 1:]
    s = np.abs(wse[:, None] - wse[nb]) / np.maximum(d, 1e-9)
    bad = (d < DX * 0.5) | (s < MIN_SLOPE) | (s > MAX_SLOPE)
    s = np.where(bad, np.nan, s)

    with np.errstate(invalid="ignore"):
        slopes = np.nanmedian(s, axis=1)

    valid = np.isfinite(slopes)
    if valid.sum() == 0:
        return np.full((H, W), DEFAULT_SLOPE, np.float32)

    vpts = np.ascontiguousarray(pts[valid])
    vsl = slopes[valid]

    w, gi = _idw_grid(vpts, H, W, DX, power, K_NEIGHBORS)
    S = (w * vsl[gi]).sum(1)
    return np.clip(S.reshape(H, W), MIN_SLOPE, MAX_SLOPE).astype(np.float32)


def as_floodfill(h_obs_t, obs_rc, dem, H, W, DX, adaptive=True,
                 reduce=REDUCTION, slope_scale=1.0):
    obs_r, obs_c = obs_rc
    wet = h_obs_t > WET_OBS_THRESHOLD
    if wet.sum() < MIN_WET_OBS:
        pseudo = np.zeros((H, W), np.float32)
        pseudo[obs_r, obs_c] = np.maximum(h_obs_t, 0.0)
        return pseudo

    z_obs = dem[obs_r, obs_c]
    wse_obs = h_obs_t + z_obs
    S = (adaptive_slope_field(obs_rc, wse_obs, wet, H, W, DX) if adaptive
         else np.full((H, W), DEFAULT_SLOPE, np.float32))
    if slope_scale != 1.0:
        S = np.clip(S * slope_scale, MIN_SLOPE, MAX_SLOPE)

    fill_fn = _fill_near if reduce == "near" else _fill_max
    wse, filled = fill_fn(obs_r[wet], obs_c[wet], wse_obs[wet],
                          S, dem, H, W, DX)

    pseudo = np.zeros((H, W), np.float32)
    pseudo[filled] = np.maximum(wse[filled] - dem[filled], 0.0)
    pseudo[dem > float(wse_obs[wet].max()) + WSE_MARGIN] = 0.0
    pseudo[obs_r, obs_c] = np.maximum(h_obs_t, 0.0)
    return pseudo


def as_floodfill_series(h_obs, obs_rc, dem, H, W, DX, adaptive=True,
                        reduce=REDUCTION, slope_scale=1.0):
    T = h_obs.shape[0]
    out = np.zeros((T, H, W), np.float32)
    for t in range(T):
        out[t] = as_floodfill(h_obs[t], obs_rc, dem, H, W, DX,
                              adaptive, reduce, slope_scale)
    return out


def count_collapsed(h_obs):
    return int(((h_obs > WET_OBS_THRESHOLD).sum(1) < MIN_WET_OBS).sum())


def warmup():
    if not HAS_NUMBA:
        return False
    dem = np.zeros((8, 8), np.float32)
    S = np.full((8, 8), 1e-3, np.float32)
    sr = np.array([4], np.int32)
    sc = np.array([4], np.int32)
    sw = np.array([1.0])
    _fill_max(sr, sc, sw, S, dem, 8, 8, 10.0)
    _fill_near(sr, sc, sw, S, dem, 8, 8, 10.0)
    return True
