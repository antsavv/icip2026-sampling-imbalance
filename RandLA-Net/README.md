# RandLA-Net — random sampling

The **RandLA-Net** half of the ICIP 2026 paper *"How Sampling Strategy Affects Imbalance Mitigation
in LiDAR Segmentation: A Study of Structured vs. Random Point-Based Architectures"*.

This is a standalone codebase with its own environment and build step. The KPConv half is in
[`../KPConv-PyTorch`](../KPConv-PyTorch) and is set up separately. The two are shipped together but
are not run together. See the [top-level README](../README.md) for the paper, citation and licence.

Here we train RandLA-Net on DALES, S3DIS and STPLS3D under 11 class-imbalance mitigation strategies
(six loss reweighting schemes and five imbalance-aware losses) and characterise the loss landscape
at the converged solutions.

## Installation

```bash
conda create --name randlanet-pt-310 python=3.10
conda activate randlanet-pt-310
pip install -r requirements.txt
```

Then build the native operators — nearest-neighbour search and grid subsampling:

```bash
sh compile_op.sh
```

## Data

Datasets go under `./data/`:

```
data/DALES/          data/S3DIS-aligned-version/          data/STPLS3D/
```

Each needs a one-off preparation pass that grid-subsamples the clouds and builds the KD-trees:

```bash
python data_prepare_DALES.py        # also: data_prepare_S3DIS.py, data_prepare_STPLS3D.py
```

## Running the experiments

Each of the 11 methods is one flag. Reweighting schemes go through `--weights`, imbalance-aware
losses through `--loss`; the uniform baseline is the default for both.

```bash
# uniform baseline (uni)
python train_randlanet.py --dataset DALES

# the five reweighting schemes
python train_randlanet.py --dataset DALES --weights invf    # also: cb, invl, invp, comf

# the five imbalance-aware losses
python train_randlanet.py --dataset DALES --loss focal      # also: balanced_softmax, ladj, ldam, seesaw
```

`--dataset` takes `DALES`, `S3DIS` or `STPLS3D`; for S3DIS add `--test_area 5`. Training runs for
100 epochs with an initial learning rate of 0.01, decayed 5% per epoch, on 40,960 points per sample.

Each run writes to `results/<DATASET>/<DATASET>-<timestamp>_<scheme>_<loss>/`.

Evaluate a trained run (per-class and mean IoU, aggregated over 10 voting passes):

```bash
python test_randlanet.py --dataset DALES \
    --checkpoint_path results/DALES/<run>/checkpoint.tar --num_votes 10
```

> **Note on S3DIS class order.** This codebase orders the S3DIS classes as `... door, table,
> chair, sofa, bookcase, board, clutter`, whereas KPConv and the paper's tables use `... door,
> chair, table, bookcase, sofa, board, clutter`. Per-class IoU printed here is therefore in a
> different order than Tables S5–S6 and Fig. 3: `chair`/`table` and `bookcase`/`sofa` are
> swapped. Permute before comparing against the other half or against the values embedded in
> `../visualize-S3DIS.py`. Class counts and weights are unaffected. Both are defined in this
> codebase's own order and match KPConv's up to the same permutation. DALES and STPLS3D share an
> identical order across both halves.

### Class weights

The weight vectors for each scheme are stored in `DataProcessing.get_class_weights` in
`helper_tool.py`, normalised to sum to 1 and derived from the full-resolution per-class point
counts. `utils/point_counts_S3DIS.py` recomputes the S3DIS counts.

## Loss landscape analysis

Flatness is the training-loss deviation under `K` filter-normalised random weight perturbations of
magnitude `ρ` (a percentage of the weight norm). Run it per trained model:

```bash
python diagnostics.py --dataset DALES \
    --checkpoint_path results/DALES/<run>/checkpoint.tar \
    --compute_flatness --desired_percent 0.01,0.1,1.0,10.0,20.0 --K 20 --seed 42
```

Results are written next to the checkpoint as `flatness_samples_*.csv` and
`flatness_summary_*.csv`.

## Figures

The flatness CSVs behind **Figure 6** are committed under `results/`, so the figure regenerates
without retraining:

```bash
python diagnostics_visualizer.py
```

Set `DATASET` and the matching `METHODS` pair at the top of the script to pick the panel — DALES
(`uni` vs `LDAM`), S3DIS (`uni` vs `LDAM`) or STPLS3D (`uni` vs `comf`). It plots one panel per run,
so switch and rerun for the others.

Figures 1–4, the class-wise IoU deltas, come from the `visualize-*` scripts in the
[parent directory](..) — they cover both architectures and so live above this one.

## Attribution

RandLA-Net was introduced by Hu et al. (CVPR 2020). This code derives from the PyTorch port at
[liuxuexun/RandLA-Net-Pytorch-New](https://github.com/liuxuexun/RandLA-Net-Pytorch-New), which in
turn builds on [qiqihaer/RandLA-Net-pytorch](https://github.com/qiqihaer/RandLA-Net-pytorch) and the
official TensorFlow implementation at
[QingyongHu/RandLA-Net](https://github.com/QingyongHu/RandLA-Net). Components carried over from the
official implementation remain under its CC BY-NC-SA 4.0 licence — see [`license.txt`](./license.txt).

The loss landscape analysis follows
[loss-landscape](https://github.com/tomgoldstein/loss-landscape) (Li et al., NeurIPS 2018). Uses the
[nanoflann](https://github.com/jlblancoc/nanoflann) library.
