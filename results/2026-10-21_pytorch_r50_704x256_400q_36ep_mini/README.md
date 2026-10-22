# Baseline: PyTorch SparseBEV R50 704x256, 400 queries, on nuScenes v1.0-mini

Date: 2026-10-21
Purpose: Phase 0 reference numbers for the deployment checkpoint. Golden tensors in `golden/r50_nuimg_704x256_400q_36ep_mini/` come from this same checkpoint.

## Setup

| Item | Value |
|---|---|
| Model | SparseBEV R50, 704x256, 400 queries, 6 decoder layers, 8 frames |
| Checkpoint | `pretrain/r50_nuimg_704x256_400q_36ep.pth` (sha256 prefix `90f35eb9c42990fc`) |
| Config | `configs/r50_nuimg_704x256_400q_36ep_mini.py` (copy in this folder) |
| Data | nuScenes v1.0-mini val, 81 samples (`data/nuscenes/nuscenes_infos_val_mini_sweep.pkl`) |
| GPU | RTX 5080 Laptop GPU (sm_120), driver 595.99.02 |
| Env | conda `sparsebev-cu128`, same as the 900-query baseline. Full list in `pip_freeze.txt` |
| Sampling op | Custom CUDA `msmv_sampling` (compiled for sm_120) |
| Precision | FP32 with PyTorch's default TF32 convolutions |

## Commands

```
conda activate sparsebev-cu128
python val.py    --config configs/r50_nuimg_704x256_400q_36ep_mini.py --weights pretrain/r50_nuimg_704x256_400q_36ep.pth
python timing.py --config configs/r50_nuimg_704x256_400q_36ep_mini.py --weights pretrain/r50_nuimg_704x256_400q_36ep.pth
```

## Results

| Metric | 400q (this run) | 900q (`2026-10-21_pytorch_r50_704x256_mini`) |
|---|---|---|
| NDS | 0.4889 | 0.4746 |
| mAP | 0.4427 | 0.4316 |
| mATE | 0.6000 | 0.6375 |
| mASE | 0.4566 | 0.4447 |
| mAOE | 0.5148 | 0.5770 |
| mAVE | 0.4531 | 0.4497 |
| mAAE | 0.3003 | 0.3026 |
| FPS (timing.py, batch 1) | 14.7 / 14.9 | 10.0 / 9.9 |

Per-class:

| Class | AP | ATE | ASE | AOE | AVE | AAE |
|---|---|---|---|---|---|---|
| car | 0.706 | 0.384 | 0.155 | 0.085 | 0.115 | 0.072 |
| truck | 0.651 | 0.332 | 0.142 | 0.044 | 0.061 | 0.002 |
| bus | 0.765 | 0.421 | 0.124 | 0.054 | 0.864 | 0.126 |
| trailer | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| construction_vehicle | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| pedestrian | 0.607 | 0.550 | 0.263 | 0.339 | 0.266 | 0.203 |
| motorcycle | 0.562 | 0.589 | 0.269 | 0.706 | 0.057 | 0.000 |
| bicycle | 0.319 | 0.560 | 0.283 | 0.404 | 0.264 | 0.000 |
| traffic_cone | 0.818 | 0.166 | 0.330 | nan | nan | nan |
| barrier | 0.000 | 1.000 | 1.000 | 1.000 | nan | nan |

## Notes

- Same caveats as the 900-query run: mini val is small, and trailer, construction_vehicle and barrier score 0 AP. Use this run for relative comparisons (ONNX / TensorRT vs PyTorch), not against the paper's numbers.
- FPS here and in the paper is measured in PyTorch with Python overhead included. TensorRT numbers will come from a different measurement.

## Files

- `metrics_summary.json`, `metrics_details.json`, `plots/`: nuScenes devkit evaluation output
- `results_nusc.json`: raw predictions in nuScenes submission format
- `r50_nuimg_704x256_400q_36ep_mini.py`: config used
- `pip_freeze.txt`: exact package versions
