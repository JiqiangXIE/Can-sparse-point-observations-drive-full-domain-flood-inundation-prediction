import json
import os

import numpy as np
import torch

from . import expansion
from .config import WET_THRESHOLD
from .metrics import field_metrics, metrics_per_timestep, peak_timing_error


@torch.no_grad()
def predict_event(model, item, cfg, fold_stats, pathway):
    device = cfg["device"]
    h_min, h_max = fold_stats[cfg["ch_depth"]]
    h_range = h_max - h_min
    use_amp = torch.device(device).type == "cuda"

    inputs = item["input"].to(device)
    T = inputs.shape[0]
    H, W = inputs.shape[-2:]
    out = torch.empty((T, H, W), device=device, dtype=torch.float32)

    hidden = None
    for t in range(T):
        x = inputs[t].unsqueeze(0).unsqueeze(0)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            p, hidden = model(x, hidden)
        d = torch.clamp(p[0, 0, 0].float() * h_range + h_min, min=0)
        if pathway == "post":
            out[t] = d
        else:
            out[t] = d * (p[0, 0, 1].float() > 0)

    pred = out.cpu().numpy()
    del out, inputs
    return pred


@torch.no_grad()
def evaluate_dataset(model, dataset, cfg, fold_stats, pathway, out_dir=None,
                     save_pred=True, expand_jobs=1, verbose=True):
    model.eval()
    h_min, h_max = fold_stats[cfg["ch_depth"]]
    h_range = h_max - h_min
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    per_event = []
    for idx in range(len(dataset)):
        item = dataset[idx]
        fname = item["filename"]
        true = np.clip(item["target_h"].numpy() * h_range + h_min, 0, None)
        dem = item["dem_phys"].numpy()

        pred = predict_event(model, item, cfg, fold_stats, pathway)

        extra = {}
        if pathway == "post":
            obs_r, obs_c = np.where(item["obs_mask"].numpy() > 0.5)
            points = pred[:, obs_r, obs_c]
            true_points = true[:, obs_r, obs_c]
            wet = true_points > WET_THRESHOLD
            if wet.any():
                extra["wet_point_miss_rate"] = float(
                    ((points <= WET_THRESHOLD) & wet).sum() / wet.sum())
            if (~wet).any():
                extra["dry_point_false_rate"] = float(
                    ((points > WET_THRESHOLD) & ~wet).sum() / (~wet).sum())
            if out_dir and save_pred:
                np.save(f"{out_dir}/{fname}_points.npy",
                        points.astype(np.float32))
            pred = expansion.expand_sequence(points, (obs_r, obs_c), dem,
                                             cfg["dx"], n_jobs=expand_jobs)

        m = field_metrics(pred, true)
        m["peak_timing_error_h"] = peak_timing_error(pred, true, dt=cfg["dt"])
        m["event"] = fname
        m.update(extra)
        per_event.append(m)

        if out_dir and save_pred:
            np.save(f"{out_dir}/{fname}_h_pred.npy", pred.astype(np.float16))
            with open(f"{out_dir}/{fname}_per_timestep.json", "w") as f:
                json.dump(metrics_per_timestep(pred, true), f)

        if verbose:
            print(f"      {fname:<26} CSI {m['CSI']:.3f}  FAR {m['FAR']:.3f}  "
                  f"RMSE {m['RMSE']:.3f}", flush=True)
        del pred, true

    keys = [k for k in per_event[0] if k != "event"]
    agg = {k: float(np.nanmean([e[k] for e in per_event if k in e]))
           for k in keys}
    agg["per_event"] = per_event

    if out_dir:
        with open(f"{out_dir}/metrics.json", "w") as f:
            json.dump(agg, f, indent=2)
    return agg
