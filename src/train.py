import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from .config import (CHUNK, GRAD_CLIP, LAMBDA_DEPTH, LAMBDA_OCC, LR,
                     MAX_EPOCHS, TBPTT_K, WEIGHT_DECAY, WET_THRESHOLD)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def n_out_channels(pathway):
    return 1 if pathway == "post" else 2


def two_head_loss(pred, target, sup_mask, thr_norm):
    depth, logit = pred[:, 0], pred[:, 1]
    wet = (target > thr_norm) & sup_mask
    l_d = (F.mse_loss(depth[wet], target[wet]) if wet.any()
           else depth.sum() * 0.0)
    l_o = (F.binary_cross_entropy_with_logits(
        logit[sup_mask], (target[sup_mask] > thr_norm).float())
        if sup_mask.any() else logit.sum() * 0.0)
    return l_d, l_o


def point_mse_loss(pred, target, sup_mask):
    depth = pred[:, 0]
    return (F.mse_loss(depth[sup_mask], target[sup_mask]) if sup_mask.any()
            else depth.sum() * 0.0)


def _pseudo_cache(dataset, pseudo, anchor_obs_to_truth):
    cache = {}
    for idx in range(len(dataset)):
        item = dataset[idx]
        fname = item["filename"]
        ps = np.asarray(pseudo[fname], np.float32).copy()
        if anchor_obs_to_truth:
            obs_r, obs_c = torch.where(item["obs_mask"] > 0.5)
            obs_r = obs_r.numpy()
            obs_c = obs_c.numpy()
            ps[:, obs_r, obs_c] = item["target_h"].numpy()[:, obs_r, obs_c]
        valid = np.isfinite(ps)
        tgt = torch.from_numpy(
            np.where(valid, ps, 0.0).astype(np.float16)).pin_memory()
        sup = torch.from_numpy(valid).pin_memory()
        cache[fname] = (tgt, sup)
    return cache


def _event_cache(dataset, device, pathway, obs_override=None, gpu_cache=True):
    entries = []
    for idx in range(len(dataset)):
        item = dataset[idx]
        target = item["target_h"]
        obs = None
        if pathway != "full":
            obs = torch.where(item["obs_mask"] > 0.5)
        if obs_override is not None and pathway == "post":
            target = target.clone()
            target[:, obs[0], obs[1]] = torch.from_numpy(
                np.asarray(obs_override[item["filename"]], np.float32)).float()

        entry = dict(fname=item["filename"], inputs=item["input"],
                     target=target, obs=obs)

        placed = False
        if gpu_cache and torch.cuda.is_available():
            need = (entry["inputs"].numel() + entry["target"].numel()) * 4
            free = torch.cuda.mem_get_info()[0]
            if free > need + 8 * (1 << 30):
                try:
                    entry["inputs"] = entry["inputs"].to(device)
                    entry["target"] = entry["target"].to(device)
                    placed = True
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
        if not placed:
            entry["inputs"] = entry["inputs"].pin_memory()
            entry["target"] = entry["target"].pin_memory()
        entries.append(entry)
    return entries


def train(model, dataset, cfg, fold_stats, pathway, pseudo=None,
          anchor_obs_to_truth=True, obs_override=None,
          max_epochs=MAX_EPOCHS, log_path=None, verbose=True):
    device = cfg["device"]
    h_min, h_max = fold_stats[cfg["ch_depth"]]
    thr_norm = (WET_THRESHOLD - h_min) / (h_max - h_min)
    use_amp = torch.device(device).type == "cuda"

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs)

    pcache = (_pseudo_cache(dataset, pseudo, anchor_obs_to_truth)
              if pathway == "pre" else None)
    events = _event_cache(dataset, device, pathway, obs_override)

    log = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        t0 = time.time()
        s_tot = torch.zeros((), device=device)
        s_d = torch.zeros((), device=device)
        s_o = torch.zeros((), device=device)
        cTP = torch.zeros((), device=device, dtype=torch.long)
        cFP = torch.zeros_like(cTP)
        cFN = torch.zeros_like(cTP)
        n_chunks = 0

        for entry in events:
            inputs, target = entry["inputs"], entry["target"]
            T = inputs.shape[0]
            pc = pcache[entry["fname"]] if pcache else None

            hidden = None
            accum = torch.zeros((), device=device)
            since = 0

            for c0 in range(0, T, CHUNK):
                c1 = min(c0 + CHUNK, T)
                n_chunks += 1
                since += 1

                x = inputs[c0:c1].unsqueeze(0).to(device, non_blocking=True)
                y = target[c0:c1].to(device, non_blocking=True).float()

                with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                        enabled=use_amp):
                    pred, hidden = model(x, hidden)
                    p = pred[0]

                    if pathway == "full":
                        tgt = y
                        sup = torch.ones_like(y, dtype=torch.bool)
                        l_d, l_o = two_head_loss(p, tgt, sup, thr_norm)
                        loss = LAMBDA_DEPTH * l_d + LAMBDA_OCC * l_o
                    elif pathway == "pre":
                        tgt = pc[0][c0:c1].to(device, non_blocking=True).float()
                        sup = pc[1][c0:c1].to(device, non_blocking=True)
                        l_d, l_o = two_head_loss(p, tgt, sup, thr_norm)
                        loss = LAMBDA_DEPTH * l_d + LAMBDA_OCC * l_o
                    else:
                        tgt = y
                        sup = torch.zeros_like(y, dtype=torch.bool)
                        sup[:, entry["obs"][0], entry["obs"][1]] = True
                        l_d = point_mse_loss(p, tgt, sup)
                        l_o = torch.zeros((), device=device)
                        loss = LAMBDA_DEPTH * l_d

                    with torch.no_grad():
                        if pathway == "post":
                            pw = p[:, 0][sup] > thr_norm
                        else:
                            pw = ((p[:, 0] > thr_norm) & (p[:, 1] > 0))[sup]
                        tw = (tgt > thr_norm)[sup]
                        cTP += (pw & tw).sum()
                        cFP += (pw & ~tw).sum()
                        cFN += (~pw & tw).sum()

                accum = accum + loss
                s_tot += loss.detach()
                s_d += l_d.detach()
                s_o += l_o.detach()
                del x, y, pred, p

                if since >= TBPTT_K or c1 >= T:
                    optimizer.zero_grad(set_to_none=True)
                    (accum / since).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    optimizer.step()
                    hidden = tuple(h.detach() for h in hidden)
                    accum = torch.zeros((), device=device)
                    since = 0

        scheduler.step()
        TP, FP, FN = int(cTP), int(cFP), int(cFN)
        log.append(dict(epoch=epoch,
                        total=float(s_tot) / max(n_chunks, 1),
                        mse_depth=float(s_d) / max(n_chunks, 1),
                        bce_occ=float(s_o) / max(n_chunks, 1),
                        train_csi=TP / max(TP + FP + FN, 1),
                        sec=round(time.time() - t0, 1)))

        if log_path:
            with open(log_path, "w") as f:
                json.dump({"pathway": pathway, "epochs": max_epochs,
                           "log": log}, f, indent=1)

        if verbose and (epoch % 10 == 0 or epoch in (1, max_epochs)):
            e = log[-1]
            print(f"    ep{epoch:02d} {e['sec']:5.0f}s  loss {e['total']:.4f} "
                  f"(d {e['mse_depth']:.4f} o {e['bce_occ']:.4f})  "
                  f"train_csi {e['train_csi']:.4f}", flush=True)

    return log
