# dataset settings
dataset_type = 'CIFAR100'
# root = '/home/sr/mmclass/mmpretrain-0.25.0/'
root = "/yds/code/DataDownload/classification/data/datasets2mmcls/cifar100_20200721"
img_norm_cfg = dict(
    mean=[129.304, 124.070, 112.434],
    std=[68.170, 65.392, 70.418],
    to_rgb=False)
train_pipeline = [
    dict(type='RandomCrop', size=32, padding=4),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_prefix=root+'data/cifar100',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix=root+'data/cifar100',
        pipeline=test_pipeline,
        test_mode=True),
    test=dict(
        type=dataset_type,
        data_prefix=root+'data/cifar100',
        pipeline=test_pipeline,
        test_mode=True))
