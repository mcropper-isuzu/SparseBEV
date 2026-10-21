_base_ = ['./r50_nuimg_704x256.py']

# Evaluate on nuScenes v1.0-mini (81 val samples); the version is read from the pkl metadata
data = dict(
    workers_per_gpu=4,
    val=dict(ann_file='data/nuscenes/nuscenes_infos_val_mini_sweep.pkl'),
)
