# Baseline: PyTorch SparseBEV R50 704x256 on nuScenes v1.0-mini



## Setup

| Item | Value |
|---|---|
| Model | SparseBEV R50, 704x256, 900 queries, 6 decoder layers, 8 frames |
| Checkpoint | `pretrain/r50_nuimg_704x256.pth` (sha256 prefix `1fa2e196973f7174`) |
| Config | `configs/r50_nuimg_704x256_mini.py`|
| Data | nuScenes v1.0-mini val, 81 samples (`data/nuscenes/nuscenes_infos_val_mini_sweep.pkl`) |
| GPU | RTX 5080 Laptop GPU (sm_120), driver 595.99.02 |
| Env | conda `sparsebev-cu128`: Python 3.10, PyTorch 2.7.1+cu128, mmcv-full 1.7.0 (built from source, sm_120), mmdet 2.28.2, mmdet3d 1.0.0rc6. Full list in `pip_freeze.txt` |
| Sampling op | Custom CUDA `msmv_sampling` (compiled for sm_120) |
| Precision | `val.py` sets `fp16_enabled=True`; `timing.py` runs FP32 |

Changes from orginal repo: `mmcv/parallel/_functions.py` line 75 wraps the device index in `torch.device("cuda", device)` (mmcv 1.x vs PyTorch 2.x incompatibility).

## Setup

```
conda activate sparsebev-cu128
python val.py    --config configs/r50_nuimg_704x256_mini.py --weights pretrain/r50_nuimg_704x256.pth
python timing.py --config configs/r50_nuimg_704x256_mini.py --weights pretrain/r50_nuimg_704x256.pth
```

## Results

| Metric | Value |
|---|---|
| NDS | 0.4746 |
| mAP | 0.4316 |
| mATE | 0.6375 |
| mASE | 0.4447 |
| mAOE | 0.5770 |
| mAVE | 0.4497 |
| mAAE | 0.3026 |
| FPS (timing.py, batch 1) | 10.0 / 9.9 (two runs, 40 timed samples each after 10 warmup) |

Per-class:

| Class | AP | ATE | ASE | AOE | AVE | AAE |
|---|---|---|---|---|---|---|
| car | 0.707 | 0.390 | 0.155 | 0.096 | 0.125 | 0.073 |
| truck | 0.590 | 0.393 | 0.156 | 0.042 | 0.085 | 0.005 |
| bus | 0.600 | 0.662 | 0.091 | 0.064 | 0.842 | 0.174 |
| trailer | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| construction_vehicle | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| pedestrian | 0.653 | 0.500 | 0.268 | 0.421 | 0.253 | 0.169 |
| motorcycle | 0.581 | 0.586 | 0.281 | 1.116 | 0.060 | 0.000 |
| bicycle | 0.297 | 0.662 | 0.193 | 0.453 | 0.234 | 0.000 |
| traffic_cone | 0.887 | 0.181 | 0.304 | nan | nan | nan |
| barrier | 0.000 | 1.000 | 1.000 | 1.000 | nan | nan |


## Files

- `metrics_summary.json`, `metrics_details.json`, `plots/`: nuScenes devkit evaluation output
- `results_nusc.json`: raw predictions in nuScenes submission format
- `r50_nuimg_704x256_mini.py`: config used
- `pip_freeze.txt`: env config
