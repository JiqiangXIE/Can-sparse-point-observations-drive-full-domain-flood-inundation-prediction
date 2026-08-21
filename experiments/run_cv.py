import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import SITES, obs_key, output_dir
from src.data import all_events, load_folds
from src.runner import run_single

PATHWAYS = ("full", "pre", "post")
OBS_KEY = obs_key("terrain", 50)
N_FOLDS = 5


def site_folds(site):
    cv, _ = load_folds(site)
    if len(cv) == N_FOLDS:
        return [dict(fold=f"fold{k + 1}", train=f["train"], test=f["test"])
                for k, f in enumerate(cv)]

    events = all_events(site)
    groups = np.array_split(np.array(events), N_FOLDS)
    folds = []
    for k, test in enumerate(groups, 1):
        test = [str(e) for e in test]
        folds.append(dict(fold=f"fold{k}",
                          train=[e for e in events if e not in test],
                          test=test))
    return folds


def summarise(sites):
    print(f"\n{'=' * 78}\n  cross-validated performance\n{'=' * 78}")
    for site in sites:
        print(f"\n[{site}]")
        print(f"  {'pathway':<10}{'CSI':>16}{'FAR':>16}{'POD':>16}{'RMSE':>16}")
        for pw in PATHWAYS:
            vals = {k: [] for k in ("CSI", "FAR", "POD", "RMSE")}
            for f in site_folds(site):
                fp = (output_dir("runs", "cv", site, f"{f['fold']}_{pw}",
                                 create=False) / "predictions" / "metrics.json")
                if not fp.exists():
                    continue
                m = json.load(open(fp))
                for k in vals:
                    vals[k].append(m[k])
            if not vals["CSI"]:
                continue
            cells = "".join(
                f"{np.mean(vals[k]):>10.3f} ±{np.std(vals[k]):.3f}"
                for k in ("CSI", "FAR", "POD", "RMSE"))
            print(f"  {pw:<10}{cells}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", nargs="+", default=list(SITES))
    ap.add_argument("--pathways", nargs="+", default=list(PATHWAYS))
    ap.add_argument("--expand-jobs", type=int, default=1)
    args = ap.parse_args()

    for site in args.sites:
        for f in site_folds(site):
            for pw in args.pathways:
                run_single(site=site, experiment="cv",
                           run_id=f"{f['fold']}_{pw}", pathway=pw,
                           train_events=f["train"], test_events=f["test"],
                           obs_key=OBS_KEY, expand_jobs=args.expand_jobs)
    summarise(args.sites)


if __name__ == "__main__":
    main()
