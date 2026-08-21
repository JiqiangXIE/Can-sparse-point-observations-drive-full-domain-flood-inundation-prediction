from __future__ import annotations

import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"

_PATHS_FILE = CONFIG_DIR / "paths.yaml"
if not _PATHS_FILE.exists():
    raise FileNotFoundError(
        f"{_PATHS_FILE} not found: copy configs/paths.example.yaml to "
        f"configs/paths.yaml and set data_root"
    )

with open(_PATHS_FILE) as f:
    _PATHS = yaml.safe_load(f)

DATA_ROOT = Path(os.path.expanduser(_PATHS["data_root"])).resolve()
OUTPUT_ROOT = Path(os.path.expanduser(_PATHS.get("output_root", "outputs")))
DEVICE = _PATHS.get("device", "cuda")

SITES = ("carlisle", "bundaberg")

SEED = 42
MAX_EPOCHS = 80
LR = 1e-3
WEIGHT_DECAY = 1e-4
CHUNK = 50
TBPTT_K = 2
GRAD_CLIP = 1.0
WET_THRESHOLD = 0.1
LAMBDA_DEPTH = 1.0
LAMBDA_OCC = 0.1

HP = {
    "fno_modes_h": 12,
    "fno_modes_w": 16,
    "fno_width": 32,
    "num_fno_layers": 4,
    "lstm_hidden_dim": 64,
    "lstm_num_layers": 1,
}

_CACHE: dict[str, dict] = {}


def _resolve(rel: str) -> str:
    p = Path(rel)
    if p.is_absolute():
        return str(p)
    in_repo = REPO_ROOT / p
    return str(in_repo if in_repo.exists() else DATA_ROOT / p)


def site_config(site: str) -> dict:
    if site in _CACHE:
        return _CACHE[site]
    if site not in SITES:
        raise KeyError(f"unknown site {site!r}, expected one of {SITES}")

    with open(CONFIG_DIR / f"{site}.yaml") as f:
        cfg = yaml.safe_load(f)

    for key in ("raw_dir", "dist_to_river", "folds"):
        cfg[key] = _resolve(cfg[key])
    for key in ("osm", "landuse", "river_mask"):
        if cfg.get(key):
            cfg[key] = _resolve(cfg[key])

    ch = cfg.pop("channels")
    cfg["ch_terrain"] = list(ch["terrain"])
    cfg["ch_elev"] = ch["elevation"]
    cfg["ch_inflow"] = ch["inflow"]
    cfg["ch_depth"] = ch["depth"]
    if "inlets" in ch:
        cfg["ch_inlets"] = ch["inlets"]
    if "q_pathway" in ch:
        cfg["ch_q_pathway"] = ch["q_pathway"]

    cfg["grid"] = tuple(cfg["grid"])
    if "origin" in cfg:
        cfg["origin"] = tuple(cfg["origin"])
    cfg["device"] = DEVICE
    _CACHE[site] = cfg
    return cfg


def output_dir(*parts, create: bool = True) -> Path:
    d = OUTPUT_ROOT.joinpath(*[str(p) for p in parts])
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def event_h5_path(site: str, event: str) -> str:
    return os.path.join(site_config(site)["raw_dir"], f"{event}.h5")


def obs_key(strategy: str, n: int, seed: int = SEED) -> str:
    return f"{strategy}_N{n}" + (f"_s{seed}" if strategy == "random" else "")
