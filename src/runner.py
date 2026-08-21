import json
import time

from .config import HP, MAX_EPOCHS, SEED, output_dir, site_config
from .data import FloodDataset, compute_fold_norm_stats
from .evaluate import evaluate_dataset
from .model import create_model, set_seed
from .pseudo_labels import load_h_obs, load_obs_mask, load_pseudo
from .train import n_out_channels, train


def run_single(site, experiment, run_id, pathway, train_events, test_events,
               obs_key=None, scenario="clean", seed=SEED,
               max_epochs=MAX_EPOCHS, expand_jobs=1, verbose=True):
    rd = output_dir("runs", experiment, site, run_id)
    if (rd / "DONE").exists():
        print(f"  skip {experiment}/{site}/{run_id}")
        return None

    print(f"\n{'=' * 70}\n  {experiment} / {site} / {run_id} [{pathway}]"
          f"\n{'=' * 70}", flush=True)
    t0 = time.time()

    cfg = site_config(site)
    fold_stats = compute_fold_norm_stats(site, train_events)
    with open(rd / "norm_stats.json", "w") as f:
        json.dump({str(k): v for k, v in fold_stats.items()}, f, indent=2)

    obs_mask = None
    if pathway in ("pre", "post"):
        obs_mask, _ = load_obs_mask(site, obs_key, scenario)
        if obs_mask is None:
            raise FileNotFoundError(f"no observation mask for {site}/{obs_key}")

    train_ds = FloodDataset(site, train_events, fold_stats, obs_mask=obs_mask)
    test_ds = FloodDataset(site, test_events, fold_stats, obs_mask=obs_mask)

    pseudo = None
    obs_override = None
    if pathway == "pre":
        pseudo = load_pseudo(site, obs_key, train_events, fold_stats, scenario)
    if pathway == "post" and scenario != "clean":
        obs_override = load_h_obs(site, obs_key, train_events, fold_stats,
                                  scenario)

    set_seed(seed)
    model = create_model(in_channels=6, out_channels=n_out_channels(pathway),
                         hp=HP, device=cfg["device"])

    with open(rd / "config.json", "w") as f:
        json.dump(dict(site=site, experiment=experiment, run_id=run_id,
                       pathway=pathway, obs_key=obs_key, scenario=scenario,
                       seed=seed, epochs=max_epochs,
                       train=list(train_events), test=list(test_events)),
                  f, indent=2)

    train(model, train_ds, cfg, fold_stats, pathway=pathway, pseudo=pseudo,
          anchor_obs_to_truth=(scenario == "clean"), obs_override=obs_override,
          max_epochs=max_epochs, log_path=rd / "train_log.json",
          verbose=verbose)

    agg = evaluate_dataset(model, test_ds, cfg, fold_stats, pathway,
                           out_dir=str(rd / "predictions"),
                           expand_jobs=expand_jobs, verbose=verbose)

    (rd / "DONE").write_text("done")
    print(f"  done in {(time.time() - t0) / 60:.1f} min", flush=True)

    del model, train_ds, test_ds
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return agg
