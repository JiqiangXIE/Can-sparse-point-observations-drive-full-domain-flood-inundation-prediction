import argparse
import json
import os
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np

from . import asff
from .config import obs_key as make_obs_key
from .config import output_dir, site_config
from .metrics import MetricAccumulator
from .perturbations import apply_scenario


def label_key(obs_key, scenario="clean"):
    return obs_key if scenario == "clean" else f"{obs_key}__{scenario}"


def label_dir(site, obs_key, scenario="clean", create=True):
    return output_dir("pseudo", site, label_key(obs_key, scenario),
                      create=create)


def load_obs_mask(site, obs_key, scenario="clean"):
    d = label_dir(site, obs_key, scenario, create=False)
    fp = d / "obs_index.npy"
    if not fp.exists():
        fp = output_dir("obs_masks", site, create=False) / f"{obs_key}.npy"
        if not fp.exists():
            return None, None
        mask = (np.load(fp) > 0.5).astype(np.uint8)
        r, c = np.where(mask > 0)
        return mask, np.stack([r, c]).astype(np.int64)

    oi = np.load(fp).astype(np.int64)
    H, W = site_config(site)["grid"]
    mask = np.zeros((H, W), np.uint8)
    mask[oi[0], oi[1]] = 1
    return mask, oi


def load_pseudo(site, obs_key, events, fold_stats, scenario="clean"):
    d = label_dir(site, obs_key, scenario, create=False)
    mn, mx = fold_stats[site_config(site)["ch_depth"]]
    rng = max(mx - mn, 1e-10)
    out = {}
    for ev in events:
        fp = d / f"{ev}.npy"
        if not fp.exists():
            raise FileNotFoundError(fp)
        out[ev] = (np.load(fp).astype(np.float32) - mn) / rng
    return out


def load_h_obs(site, obs_key, events, fold_stats, scenario="clean"):
    d = label_dir(site, obs_key, scenario, create=False)
    mn, mx = fold_stats[site_config(site)["ch_depth"]]
    rng = max(mx - mn, 1e-10)
    out = {}
    for ev in events:
        fp = d / f"{ev}_h_obs.npy"
        if fp.exists():
            out[ev] = (np.load(fp).astype(np.float32) - mn) / rng
    return out


def _build_one(event, site, obs_key, scenario, seed, overwrite):
    from .data import load_event

    d = label_dir(site, obs_key, scenario)
    fp = d / f"{event}.npy"
    if fp.exists() and not overwrite:
        return dict(event=event, skipped=True)

    sc = site_config(site)
    H, W = sc["grid"]
    mask, oi = load_obs_mask(site, obs_key, "clean")
    if mask is None:
        raise FileNotFoundError(f"observation mask missing for {site}/{obs_key}")

    dem, truth = load_event(site, event)
    h_clean = truth[:, oi[0], oi[1]].astype(np.float32)
    h_obs, obs_rc = apply_scenario(h_clean, (oi[0], oi[1]), scenario, seed)

    T = truth.shape[0]
    labels = np.zeros((T, H, W), np.float16)
    acc = MetricAccumulator(depth_domain="union")
    for t in range(T):
        field = asff.as_floodfill(h_obs[t], obs_rc, dem, H, W, sc["dx"])
        labels[t] = field
        acc.update(field, truth[t])

    np.save(fp, labels)
    np.save(d / f"{event}_h_obs.npy", h_obs.astype(np.float16))
    np.save(d / "obs_index.npy", np.stack(obs_rc).astype(np.int32))

    m = acc.result()
    with open(d / f"{event}_reconstruction.json", "w") as f:
        json.dump(m, f, indent=2)

    return dict(event=event, skipped=False, n_obs=int(len(obs_rc[0])),
                collapsed=asff.count_collapsed(h_obs),
                CSI=m["CSI"], RMSE=m["RMSE"], Bias=m["Bias"])


def build(site, obs_key, events, scenario="clean", seed=42, workers=1,
          overwrite=False):
    asff.warmup()
    fn = partial(_build_one, site=site, obs_key=obs_key, scenario=scenario,
                 seed=seed, overwrite=overwrite)
    if workers <= 1:
        results = [fn(e) for e in events]
    else:
        with Pool(min(workers, cpu_count())) as pool:
            results = pool.map(fn, events)
    for r in results:
        if not r.get("skipped"):
            print(f"  {r['event']:<26} N={r['n_obs']:<5} CSI {r['CSI']:.3f}  "
                  f"RMSE {r['RMSE']:.3f}  bias {r['Bias']:+.3f}")
    return results


def main():
    from .data import all_events

    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    ap.add_argument("--strategy", default="terrain")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--placement-seed", type=int, default=42)
    ap.add_argument("--scenario", default="clean",
                    choices=["clean", "mild", "severe"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    key = make_obs_key(args.strategy, args.n, args.placement_seed)
    build(args.site, key, all_events(args.site), args.scenario, args.seed,
          args.workers, args.overwrite)


if __name__ == "__main__":
    main()
