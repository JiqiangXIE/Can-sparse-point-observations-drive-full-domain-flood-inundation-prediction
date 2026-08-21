import argparse

import numpy as np
from scipy.ndimage import distance_transform_edt
from sklearn.cluster import KMeans, MiniBatchKMeans

DENSITIES = (5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000)
RANDOM_SEEDS = (42, 1, 2, 3, 4)
SEED = 42

TIER_FRAC = {"low": 0.4, "med": 0.4, "high": 0.2}
TIER_PERCENTILES = (10, 50)
CCTV_RIVER_FRAC = 0.10
FLOODNET_RIVER_FRAC = 0.10
MBK_THRESHOLD = 200
STRATEGIES = ("terrain", "random", "cctv", "floodnet")


def hand(dem, river_mask):
    _, inds = distance_transform_edt(~river_mask, return_distances=True,
                                     return_indices=True)
    z_drain = dem[inds[0], inds[1]]
    h = (dem - z_drain).astype(np.float32)
    h[river_mask] = 0.0
    return np.maximum(h, 0.0)


def elevation_tiers(dem, percentiles=TIER_PERCENTILES):
    p_lo = np.percentile(dem, percentiles[0])
    p_hi = np.percentile(dem, percentiles[1])
    return dem <= p_lo, (dem > p_lo) & (dem <= p_hi), dem > p_hi


def pick(mask, n, value=None, mode="centroid", seed=SEED):
    if n <= 0:
        return []
    r, c = np.where(mask)
    if len(r) == 0:
        return []
    if len(r) <= n:
        return list(zip(r, c))

    coords = np.stack([r, c], 1).astype(np.float64)
    if n > MBK_THRESHOLD:
        km = MiniBatchKMeans(n_clusters=n, random_state=seed, n_init=3,
                             batch_size=min(2048, len(r)),
                             max_iter=100).fit(coords)
    else:
        km = KMeans(n_clusters=n, random_state=seed, n_init=3,
                    max_iter=100).fit(coords)
    labels, centers = km.labels_, km.cluster_centers_

    out = []
    for k in range(n):
        ci = np.where(labels == k)[0]
        if len(ci) == 0:
            continue
        if mode == "min" and value is not None:
            best = ci[np.argmin(value[r[ci], c[ci]])]
        elif mode == "max" and value is not None:
            best = ci[np.argmax(value[r[ci], c[ci]])]
        else:
            best = ci[np.argmin(np.sum((coords[ci] - centers[k]) ** 2, 1))]
        out.append((r[best], c[best]))
    return out


def points_to_mask(points, shape):
    m = np.zeros(shape, np.uint8)
    for r, c in points:
        m[r, c] = 1
    return m


def terrain_stratified(dem, hand_field, n_points, seed=SEED,
                       tier_frac=TIER_FRAC, percentiles=TIER_PERCENTILES):
    zone_low, zone_med, zone_high = elevation_tiers(dem, percentiles)
    n_low = int(n_points * tier_frac["low"])
    n_med = int(n_points * tier_frac["med"])
    n_high = n_points - n_low - n_med
    points = (pick(zone_low, n_low, hand_field, "min", seed)
              + pick(zone_med, n_med, hand_field, "min", seed)
              + pick(zone_high, n_high, None, "centroid", seed))
    return points_to_mask(points, dem.shape)


def random_placement(dem, n_points, seed=SEED):
    rng = np.random.RandomState(seed)
    r, c = np.where(np.isfinite(dem))
    idx = rng.choice(len(r), size=min(n_points, len(r)), replace=False)
    return points_to_mask(list(zip(r[idx], c[idx])), dem.shape)


def cctv(cctv_candidate, cctv_step, road_available, river_mask, n_points,
         seed=SEED):
    n_river = max(1, int(n_points * CCTV_RIVER_FRAC))
    n_urban = n_points - n_river
    sel = pick(cctv_candidate, n_urban, seed=seed)
    if len(sel) < n_urban:
        sel = pick(cctv_step, n_urban, seed=seed)
    if len(sel) < n_urban:
        sel = pick(road_available, n_urban, seed=seed)
    return points_to_mask(sel + pick(river_mask, n_river, seed=seed),
                          river_mask.shape)


def floodnet(floodnet_score, ever_wet, river_mask, n_points, seed=SEED):
    n_river = max(1, int(n_points * FLOODNET_RIVER_FRAC))
    n_flood = n_points - n_river
    score = floodnet_score.copy()
    score[river_mask] = 0.0
    candidate = (ever_wet >= 1) & ~river_mask
    points = (pick(candidate, n_flood, score, "max", seed)
              + pick(river_mask, n_river, seed=seed))
    return points_to_mask(points, river_mask.shape)


def generate(site, strategy, n_points, seed=SEED, layers=None):
    from .placement_inputs import load_layers

    L = layers if layers is not None else load_layers(site)
    if strategy == "terrain":
        return terrain_stratified(L["dem"], L["hand"], n_points, seed)
    if strategy == "random":
        return random_placement(L["dem"], n_points, seed)
    if strategy == "cctv":
        return cctv(L["cctv_candidate"], L["cctv_step"], L["road_available"],
                    L["river_mask"], n_points, seed)
    if strategy == "floodnet":
        return floodnet(L["floodnet_score"], L["ever_wet"], L["river_mask"],
                        n_points, seed)
    raise ValueError(f"unknown strategy {strategy!r}")


def generate_all(site, densities=DENSITIES, random_seeds=RANDOM_SEEDS):
    from .config import obs_key, output_dir
    from .placement_inputs import load_layers

    layers = load_layers(site)
    out = output_dir("obs_masks", site)
    written = []
    for strategy in STRATEGIES:
        seeds = random_seeds if strategy == "random" else (SEED,)
        for seed in seeds:
            for n in densities:
                mask = generate(site, strategy, n, seed, layers)
                key = obs_key(strategy, n, seed)
                np.save(out / f"{key}.npy", mask)
                written.append((key, int(mask.sum())))
                print(f"  {key:<22} {int(mask.sum()):>5} points")
    return written


def main():
    from .config import obs_key, output_dir
    from .placement_inputs import load_layers

    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    ap.add_argument("--strategy", choices=STRATEGIES)
    ap.add_argument("--n", type=int)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    if args.all:
        generate_all(args.site)
        return

    mask = generate(args.site, args.strategy, args.n, args.seed,
                    load_layers(args.site))
    key = obs_key(args.strategy, args.n, args.seed)
    path = output_dir("obs_masks", args.site) / f"{key}.npy"
    np.save(path, mask)
    print(f"{key}: {int(mask.sum())} points -> {path}")


if __name__ == "__main__":
    main()
