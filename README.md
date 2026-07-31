# Can-sparse-point-observations-drive-full-domain-flood-inundation-prediction
# Unlocking full-domain flood inundation prediction from sparse point supervision in deep-learning models

Code accompanying:

> Xie, J., Zhang, Y., Lyu, H.\*, Wang, Q. J., Fu, S., & Zhang, C. (2026).
> *Unlocking full-domain flood inundation prediction from sparse point supervision in
> deep-learning models.* Under review, **Water Research**.
> \*Corresponding author: lyuheng@dlut.edu.cn

Contents: the AS-FloodFill point-to-field expansion algorithm, the four observation
placement strategies, and the FNO-LSTM training and evaluation code behind all reported
results.

## Framework

| Pathway  | Expansion applied | Supervision seen by the network |
|----------|-------------------|---------------------------------|
| Pre-Exp  | before training   | full-domain pseudo-labels       |
| Post-Exp | after inference   | observation points only         |
| FULL     | not used          | full-domain hydrodynamic fields |

## Layout

```
configs/          site configuration and local data paths
src/
  config.py       configuration loader and hyperparameters
  asff.py         AS-FloodFill: adaptive slope field, two-pass propagation
  expansion.py    point-to-field expansion interface
  placement.py    terrain-stratified / random / CCTV / FloodNet placement
  data.py         HDF5 to model tensors, per-fold normalisation
  model.py        FNO-LSTM
  train.py        training for FULL / Pre-Exp / Post-Exp
  evaluate.py     autoregressive inference and metrics
  metrics.py      CSI / FAR / POD / RMSE / bias
  perturbations.py  Clean / Mild / Severe observation error scenarios
  pseudo_labels.py  offline pseudo-label generation
  runner.py       single-run driver
experiments/
  run_cv.py         Section 4.1, Table 2
  run_robustness.py Section 4.3, Table 1
  run_placement.py  Section 4.4
data/folds/       event splits (Tables S1, S2)
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
Fraehr et al. (2024). Processed datasets.



## Citation

```bibtex
@article{xie2026sparse,
  author  = {Xie, Jiqiang and Zhang, Yuhang and Lyu, Heng and Wang, Quan J.
             and Fu, Shengnan and Zhang, Chi},
  title   = {Unlocking full-domain flood inundation prediction from sparse point
             supervision in deep-learning models},
  journal = {Water Research},
  year    = {2026},
  note    = {Under review}
}
```

## License

MIT, see [LICENSE](LICENSE).
