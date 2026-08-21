import json

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .config import event_h5_path, site_config


def _norm_channels(sc):
    return list(sc["ch_terrain"]) + [sc["ch_inflow"], sc["ch_depth"]]


def _spread_river_inflow(raw, sc):
    if "ch_inlets" not in sc or "ch_q_pathway" not in sc:
        return raw

    qp = raw[0, sc["ch_q_pathway"]]
    inlets = raw[0, sc["ch_inlets"]]
    flow = raw[:, sc["ch_inflow"]].copy()
    T = raw.shape[0]

    q = {}
    for rid in (1, 2, 3):
        m = (inlets == rid)
        q[rid] = flow[:, m].mean(axis=1) if m.sum() > 0 else np.zeros(T, np.float32)

    seg_flow = {
        1: q[1],
        2: q[2],
        3: q[3],
        4: q[1] + q[2],
        5: q[1] + q[2] + q[3],
    }
    for seg, qseq in seg_flow.items():
        rmask = (qp == seg)
        if rmask.sum() == 0:
            continue
        for t in range(T):
            flow[t][rmask] = qseq[t]

    raw = raw.copy()
    raw[:, sc["ch_inflow"]] = flow
    return raw


def load_folds(site):
    with open(site_config(site)["folds"]) as f:
        j = json.load(f)
    return j["cross_validation"]["folds"], j["fixed_split"]


def all_events(site):
    cv, fixed = load_folds(site)
    events = set(fixed["train"]) | set(fixed["test"])
    for f in cv:
        events |= set(f["train"]) | set(f["test"])
    return sorted(events)


def load_event(site, event):
    sc = site_config(site)
    with h5py.File(event_h5_path(site, event), "r") as hf:
        d = hf["data"]
        dem = np.asarray(d[0, sc["ch_elev"]], np.float32)
        depth = np.asarray(d[:, sc["ch_depth"]], np.float32)
    return dem, np.maximum(depth, 0.0)


def load_static(site, event=None):
    sc = site_config(site)
    if event is None:
        event = all_events(site)[0]
    with h5py.File(event_h5_path(site, event), "r") as hf:
        d = hf["data"]
        dem = np.asarray(d[0, sc["ch_elev"]], np.float32)
        slope = np.asarray(d[0, sc["ch_terrain"][1]], np.float32)
        qp = (np.asarray(d[0, sc["ch_q_pathway"]], np.float32)
              if "ch_q_pathway" in sc else None)
    dist = np.load(sc["dist_to_river"]).astype(np.float32)
    if qp is None:
        qp = (dist <= sc["dx"] * 2).astype(np.float32)
    return dem, slope, qp, dist


def compute_fold_norm_stats(site, train_events):
    sc = site_config(site)
    chs = _norm_channels(sc)
    cmin = {c: np.inf for c in chs}
    cmax = {c: -np.inf for c in chs}

    for ev in train_events:
        with h5py.File(event_h5_path(site, ev), "r") as hf:
            raw = hf["data"][:]
        raw = _spread_river_inflow(raw, sc)
        for c in chs:
            cmin[c] = min(cmin[c], float(raw[:, c].min()))
            cmax[c] = max(cmax[c], float(raw[:, c].max()))
        del raw

    return {c: (cmin[c], cmax[c]) for c in chs}


def normalize(arr, mn, mx):
    rng = mx - mn
    if rng < 1e-10:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - mn) / rng).astype(np.float32)


def denormalize_depth(h_norm, stats, site):
    mn, mx = stats[site_config(site)["ch_depth"]]
    return (h_norm * (mx - mn) + mn).astype(np.float32)


class FloodDataset(Dataset):

    def __init__(self, site, events, fold_stats, obs_mask=None, preload=True):
        self.site = site
        self.sc = site_config(site)
        self.events = list(events)
        self.stats = fold_stats
        self.H, self.W = self.sc["grid"]

        dist = np.load(self.sc["dist_to_river"]).astype(np.float32)
        assert dist.shape == (self.H, self.W)
        self.dist_norm = torch.from_numpy(dist / max(dist.max(), 1.0))

        if obs_mask is not None:
            assert obs_mask.shape == (self.H, self.W)
            self.obs_mask = torch.from_numpy((obs_mask > 0.5).astype(np.float32))
        else:
            self.obs_mask = torch.ones(self.H, self.W, dtype=torch.float32)

        self.item_cache = None
        if preload:
            self.item_cache = []
            for ev in self.events:
                with h5py.File(event_h5_path(site, ev), "r") as hf:
                    raw = hf["data"][:]
                self.item_cache.append(self._build_item(raw, ev))

    def __len__(self):
        return len(self.events)

    def _build_item(self, raw, ev):
        raw = _spread_river_inflow(raw, self.sc)
        sc = self.sc
        T = raw.shape[0]

        terr = []
        for c in sc["ch_terrain"]:
            mn, mx = self.stats[c]
            terr.append(normalize(raw[:, c], mn, mx))
        terr = np.stack(terr, axis=1)

        ci = sc["ch_inflow"]
        mn, mx = self.stats[ci]
        inflow = normalize(raw[:, ci], mn, mx)[:, None]

        dist = self.dist_norm[None, None].expand(T, 1, self.H, self.W).numpy()
        inp = np.concatenate([terr, inflow, dist], axis=1)

        mn, mx = self.stats[sc["ch_depth"]]
        target_h = normalize(raw[:, sc["ch_depth"]], mn, mx)
        dem_phys = raw[0, sc["ch_elev"]].astype(np.float32)

        return {
            "input": torch.from_numpy(inp).float(),
            "target_h": torch.from_numpy(target_h).float(),
            "dem_phys": torch.from_numpy(dem_phys).float(),
            "obs_mask": self.obs_mask,
            "filename": ev,
        }

    def __getitem__(self, idx):
        if self.item_cache is not None:
            return self.item_cache[idx]
        with h5py.File(event_h5_path(self.site, self.events[idx]), "r") as hf:
            raw = hf["data"][:]
        return self._build_item(raw, self.events[idx])
