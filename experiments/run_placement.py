import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import SITES, obs_key, output_dir
from src.data import load_folds
from src.runner import run_single

DENSITIES = (25, 100)
DETERMINISTIC = ("terrain", "cctv", "floodnet")
RANDOM_SEEDS = (42, 1, 2)


def configurations():
    keys = [obs_key(s, n) for s in DETERMINISTIC for n in DENSITIES]
    keys += [obs_key("random", n, sd) for n in DENSITIES for sd in RANDOM_SEEDS]
    return keys


def summarise(sites):
    print(f"\n{'=' * 78}\n  observation density and placement strategy"
          f"\n{'=' * 78}")
    for site in sites:
        print(f"\n[{site}]")
        print(f"  {'configuration':<22}{'CSI':>9}{'FAR':>9}{'RMSE':>9}")
        for key in configurations():
            fp = (output_dir("runs", "placement", site, key, create=False)
                  / "predictions" / "metrics.json")
            if not fp.exists():
                continue
            m = json.load(open(fp))
            print(f"  {key:<22}{m['CSI']:>9.3f}{m['FAR']:>9.3f}"
                  f"{m['RMSE']:>9.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", nargs="+", default=list(SITES))
    args = ap.parse_args()

    for site in args.sites:
        _, fixed = load_folds(site)
        for key in configurations():
            run_single(site=site, experiment="placement", run_id=key,
                       pathway="pre", train_events=fixed["train"],
                       test_events=fixed["test"], obs_key=key)
    summarise(args.sites)


if __name__ == "__main__":
    main()
