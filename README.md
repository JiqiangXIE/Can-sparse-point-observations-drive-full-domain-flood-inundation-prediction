# Unlocking full-domain flood inundation prediction from sparse point supervision in deep-learning models

Code accompanying:

> Xie, J., Zhang, Y., Lyu, H.\*, Wang, Q. J., Fu, S., & Zhang, C. (2026).
> *Unlocking full-domain flood inundation prediction from sparse point supervision in
> deep-learning models.* **Water Research**, 126760.
> <https://doi.org/10.1016/j.watres.2026.126760>
> \*Corresponding author: lyuheng@dlut.edu.cn

As stated in the Data Availability section of the paper, this repository contains the
model training code (`src/train.py`, `src/model.py`, `experiments/`), the observation
point selection strategy (`src/placement.py`, `src/placement_inputs.py`), and the
AS-FloodFill spatial expansion algorithm (`src/asff.py`, `src/expansion.py`).

## Framework

| Pathway  | Expansion applied | Supervision seen by the network |
|----------|-------------------|---------------------------------|
| Pre-Exp  | before training   | full-domain pseudo-labels       |
| Post-Exp | after inference   | observation points only         |
| FULL     | not used          | full-domain hydrodynamic fields |

## Layout

```
configs/            site configuration and local data paths
src/
  config.py         configuration loader and hyperparameters
  asff.py           AS-FloodFill: adaptive slope field, two-pass propagation
  expansion.py      point-to-field expansion interface
  placement.py      terrain-stratified / random / CCTV / FloodNet placement
  placement_inputs.py  HAND, OSM road and urban layers, inundation frequency
  data.py           HDF5 to model tensors, per-fold normalisation
  model.py          FNO-LSTM
  train.py          training for FULL / Pre-Exp / Post-Exp
  evaluate.py       autoregressive inference and metrics
  metrics.py        CSI / FAR / POD / RMSE / bias
  perturbations.py  Clean / Mild / Severe observation error scenarios
  pseudo_labels.py  offline pseudo-label generation
  runner.py         single-run driver
experiments/
  run_cv.py         Section 4.1, Table 2
  run_robustness.py Section 4.3, Table 1
  run_placement.py  Section 4.4
```

## Installation

```bash
conda env create -f environment.yml
conda activate flood-sparse
cp configs/paths.example.yaml configs/paths.yaml   # then set data_root
```

Tested with Python 3.11 and PyTorch (CUDA 12) on a single GPU. AS-FloodFill is
JIT-compiled with Numba and runs on CPU.

## Data

One HDF5 file per flood event, shape `(T, C, H, W)` in physical units. The channel
layout is declared per site in `configs/*.yaml`.

| Site      | Grid      | Resolution | Time step | Events | Hydrodynamic model |
|-----------|-----------|------------|-----------|--------|--------------------|
| Carlisle  | 305 × 475 | 10 m       | 5 min     | 9      | LISFLOOD-FP        |
| Bundaberg | 225 × 400 | 20 m       | 1 h       | 21     | TUFLOW             |

LISFLOOD-FP: <https://www.lisflood.co.uk>. The Bundaberg TUFLOW events are from
Fraehr et al. (2024). The processed HDF5 datasets are not distributed with this
repository.

## Reproducing the results

```bash
python -m src.placement_inputs --site carlisle
python -m src.placement --site carlisle --strategy terrain --n 50
python -m src.pseudo_labels --site carlisle --strategy terrain --n 50 --workers 16

python experiments/run_cv.py --sites carlisle bundaberg
python experiments/run_robustness.py
python experiments/run_placement.py
```

For the perturbed observation scenarios, generate the corresponding labels first:

```bash
python -m src.pseudo_labels --site carlisle --scenario mild   --workers 16
python -m src.pseudo_labels --site carlisle --scenario severe --workers 16
```

Each run writes `outputs/runs/<experiment>/<site>/<run_id>/` containing `config.json`,
`train_log.json`, `norm_stats.json` and `predictions/` with the predicted depth fields
and `metrics.json`. A run is skipped if its `DONE` marker exists.

## Citation

```bibtex
@article{xie2026unlocking,
  author  = {Xie, Jiqiang and Zhang, Yuhang and Lyu, Heng and Wang, Quan J.
             and Fu, Shengnan and Zhang, Chi},
  title   = {Unlocking full-domain flood inundation prediction from sparse point
             supervision in deep-learning models},
  journal = {Water Research},
  year    = {2026},
  pages   = {126760},
  doi     = {10.1016/j.watres.2026.126760}
}
```

## License

MIT, see [LICENSE](LICENSE).
