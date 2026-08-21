import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import SITES, obs_key, output_dir
from src.data import load_folds
from src.runner import run_single

SCENARIOS = ("clean", "mild", "severe")
PATHWAYS = ("pre", "post")
OBS_KEY = obs_key("terrain", 50)


def summarise(sites):
    print(f"\n{'=' * 78}\n  sensitivity to imperfect observations\n{'=' * 78}")
    for site in sites:
        print(f"\n[{site}]")
        print(f"  {'scenario':<10}{'pathway':<8}{'CSI':>9}{'FAR':>9}"
              f"{'RMSE':>9}{'miss_rate':>11}")
        for scen in SCENARIOS:
            for pw in PATHWAYS:
                fp = (output_dir("runs", "robustness", site, f"{scen}_{pw}",
                                 create=False) / "predictions" / "metrics.json")
                if not fp.exists():
                    continue
                m = json.load(open(fp))
                miss = m.get("wet_point_miss_rate")
                miss_s = f"{miss:>11.3f}" if miss is not None else " " * 11
                print(f"  {scen:<10}{pw:<8}{m['CSI']:>9.3f}{m['FAR']:>9.3f}"
                      f"{m['RMSE']:>9.3f}{miss_s}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", nargs="+", default=list(SITES))
    ap.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    ap.add_argument("--expand-jobs", type=int, default=1)
    args = ap.parse_args()

    for site in args.sites:
        _, fixed = load_folds(site)
        for scen in args.scenarios:
            for pw in PATHWAYS:
                run_single(site=site, experiment="robustness",
                           run_id=f"{scen}_{pw}", pathway=pw,
                           train_events=fixed["train"],
                           test_events=fixed["test"],
                           obs_key=OBS_KEY, scenario=scen,
                           expand_jobs=args.expand_jobs)
    summarise(args.sites)


if __name__ == "__main__":
    main()
