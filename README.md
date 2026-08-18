# How Sampling Strategy Affects Imbalance Mitigation in LiDAR Segmentation

### A Study of Structured vs. Random Point-Based Architectures

Code and results for the ICIP 2026 paper by **Antonis Savva**, **Christos Kyrkou** and
**Theocharis Theocharides** — KIOS Research and Innovation Center of Excellence and Department of
Electrical and Computer Engineering, University of Cyprus.

We benchmark 11 class-imbalance mitigation strategies — six loss reweighting schemes and five
imbalance-aware losses — across three acquisition modalities (DALES, aerial LiDAR, 641:1; S3DIS,
indoor RGB-D, 56:1; STPLS3D, photogrammetric/synthetic, 101:1) and two point-based architectures
with contrasting sampling strategies: KPConv (structured) and RandLA-Net (random). We then relate
the outcomes to the geometry of the converged loss landscape.

## Layout

```
KPConv-PyTorch/      structured sampling  — training, testing, flatness analysis, Fig. 5
RandLA-Net/          random sampling      — training, testing, flatness analysis, Fig. 6
visualize-DALES.py   \
visualize-S3DIS.py    >  class-wise IoU deltas, Figs. 1-4 (both architectures)
visualize-STPLS3D.py /
```

The two halves are **separate codebases with separate environments and build steps**. They are
shipped together because the paper covers both. Set up whichever one you need, following its own README:

- [`KPConv-PyTorch/README.md`](KPConv-PyTorch/README.md)
- [`RandLA-Net/README.md`](RandLA-Net/README.md)

## Reproducing the figures

Every figure in the paper regenerates without retraining. The per-class IoU values are embedded in
the `visualize-*` scripts, and the loss landscape CSVs are committed under each half's `results/`.

| Paper artifact | Where | Notes |
| --- | --- | --- |
| Figs. 1–2 (DALES ΔIoU) | `visualize-DALES.py` | set `data` to `results_KPConv` or `results_RandLA` |
| Fig. 3 (S3DIS ΔIoU) | `visualize-S3DIS.py` | same toggle |
| Fig. 4 (STPLS3D ΔIoU) | `visualize-STPLS3D.py` | same toggle |
| Fig. 5 (KPConv flatness) | `KPConv-PyTorch/diagnostics_visualizer.py` | set `DATASET` + matching `METHODS` pair |
| Fig. 6 (RandLA-Net flatness) | `RandLA-Net/diagnostics_visualizer.py` | same |
| Table 1, Tables S1–S6 | per-class values inlined in the `visualize-*` scripts | |

The three root scripts need only `numpy`, `pandas` and `matplotlib`. The two
`diagnostics_visualizer.py` scripts additionally need `seaborn` and must be run from inside their
own directory, since they read `results/<DATASET>/` relative to it.

Each script plots **one panel per run**. Telection is a variable near the top of the file,
so switch the toggle and rerun to produce the other panel of a pair.

## Citation

```bibtex
@inproceedings{savva2026sampling,
  author    = {Savva, Antonis and Kyrkou, Christos and Theocharides, Theocharis},
  title     = {How Sampling Strategy Affects Imbalance Mitigation in {LiDAR} Segmentation:
               A Study of Structured vs. Random Point-Based Architectures},
  booktitle = {IEEE International Conference on Image Processing (ICIP)},
  year      = {2026}
}
```

## Licence

Our own contributions are released under the MIT Licence (see [`LICENSE`](./LICENSE)).

The two halves inherit different terms from their upstreams, and the distinction matters:

- **`KPConv-PyTorch/`** derives from [KPConv-PyTorch](https://github.com/HuguesTHOMAS/KPConv-PyTorch)
  (MIT). Uniformly MIT; see [`KPConv-PyTorch/LICENSE`](KPConv-PyTorch/LICENSE).
- **`RandLA-Net/`** derives from a PyTorch port of RandLA-Net whose lineage reaches the official
  implementation, released under **CC BY-NC-SA 4.0**. Components carried over from it remain under
  that licence, which restricts commercial use and requires share-alike. See
  [`RandLA-Net/license.txt`](RandLA-Net/license.txt) and the attribution section of that half's
  README.

## Acknowledgements

This work has received funding under Grant Agreement No. 101168067, GuardAI — Enhancing Robustness
and Security of Edge AI Systems for Safety-Critical Applications, with support from the European
Cybersecurity Competence Centre. The views and opinions expressed are those of the authors only and
do not necessarily reflect those of the European Union or the European Cybersecurity Competence
Centre. Neither the European Union nor the European Cybersecurity Competence Centre can be held
responsible for them.

Computational resources were provided by the High Performance Computing facility of the University
of Cyprus (UCY HPC).

The loss landscape analysis in both halves follows
[loss-landscape](https://github.com/tomgoldstein/loss-landscape) (Li et al., NeurIPS 2018).
