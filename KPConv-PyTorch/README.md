# KPConv — structured sampling

The **KPConv** half of the ICIP 2026 paper *"How Sampling Strategy Affects Imbalance Mitigation in
LiDAR Segmentation: A Study of Structured vs. Random Point-Based Architectures"*.

This is a standalone codebase with its own environment and build step. The RandLA-Net half is in
[`../RandLA-Net`](../RandLA-Net) and is set up separately. The two are shipped together but are not run together. See the [top-level README](../README.md) for the paper, citation and licence.

Here we train KPConv on DALES, S3DIS and STPLS3D under 11 class-imbalance mitigation strategies (six loss reweighting schemes and five imbalance-aware losses) and characterise the loss landscape at the converged solutions.

## Installation

```bash
conda create --name kpconv-pt python=3.10
conda activate kpconv-pt
pip install -r requirements.txt
```

Then compile the C++ extensions:

```bash
cd cpp_wrappers && sh compile_wrappers.sh     # Linux
# Windows: run cpp_wrappers/cpp_neighbors/build.bat and cpp_wrappers/cpp_subsampling/build.bat
```

[INSTALL.md](./INSTALL.md) carries the upstream KPConv notes on CUDA/cuDNN setup.

## Data

Datasets go under `./data/`, which already contains the expected folder skeleton:

```
data/DALES/          data/S3DIS-aligned-version/          data/STPLS3D/
```

DALES ships as ASCII; convert it first with `convert_ascii2bin_DALES.py`.

## Running the experiments

Each of the 11 methods is one flag. Reweighting schemes go through `--weights`, imbalance-aware
losses through `--loss`; the uniform baseline is the default for both.

```bash
# uniform baseline (uni)
python train_DALES.py

# the five reweighting schemes
python train_DALES.py --weights invf     # also: cb, invl, invp, comf

# the five imbalance-aware losses
python train_DALES.py --loss focal_loss  # also: ldam_loss, ladj_loss, seesaw_loss, balanced_softmax
```

Substitute `train_S3DIS.py` or `train_STPLS3D.py` for the other datasets; the interface is
identical. Training uses seed 42 by default, 400 epochs for DALES and 500 for S3DIS/STPLS3D, with
an initial learning rate of 0.01.

Each run writes to `results/<DATASET>/<DATASET>-Log_<timestamp>_w_<scheme>_<loss>_seed_<seed>/`,
including a `parameters.txt` recording every hyperparameter and the exact class weights used.

Evaluate a trained run (per-class and mean IoU, aggregated over 10 voting passes):

```bash
python test_model.py --chosen_log results/DALES/DALES-Log_<timestamp>_w_none_cross_entropy_seed_42
```

### Class weights

The weight vectors for each scheme are precomputed in the `weighting_schemes` dict at the top of
each training script, normalised to sum to 1. They are derived from the full-resolution per-class
point counts and can be regenerated with:

```bash
python dataset_statistics/calculate_weights.py
```

## Loss landscape analysis

Flatness is the training-loss deviation under `K` filter-normalised random weight perturbations of
magnitude `ρ` (a percentage of the weight norm). Run it per trained model:

```bash
python diagnostics.py --chosen_log results/DALES/<run> \
    --compute_flatness --desired_percent 0.01,0.1,1.0,10.0,20.0 --K 20 --seed 42
```

Results are written into the run directory as `flatness_samples_*.csv` and
`flatness_summary_*.csv`.

## Figures

The flatness CSVs behind **Figure 5** are committed under `results/`, so the figure regenerates
without retraining:

```bash
python diagnostics_visualizer.py
```

Set `DATASET` and the matching `METHODS` pair at the top of the script to pick the panel — DALES
(`invp` vs `uni`), S3DIS (`uni` vs `BS`) or STPLS3D (`invl` vs `uni`). It plots one panel per run,
so switch and rerun for the others.

Figures 1–4, the class-wise IoU deltas, come from the `visualize-*` scripts in the
[parent directory](..) — they cover both architectures and so live above this one.

## Attribution

Built on [KPConv-PyTorch](https://github.com/HuguesTHOMAS/KPConv-PyTorch) by Hugues Thomas (MIT).
The loss landscape analysis follows [loss-landscape](https://github.com/tomgoldstein/loss-landscape)
(Li et al., NeurIPS 2018). Uses the [nanoflann](https://github.com/jlblancoc/nanoflann) library.
