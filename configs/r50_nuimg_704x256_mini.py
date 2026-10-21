_base_ = ['./r50_nuimg_704x256.py']

data = dict(workers_per_gpu=4,val=dict(ann_file='data/nuscenes/nuscenes_infos_val_mini_sweep.pkl'),)
