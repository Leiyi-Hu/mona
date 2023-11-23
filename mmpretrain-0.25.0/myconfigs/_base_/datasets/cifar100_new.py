# 2.220服务器相对地址
# root = '/yds/code/DataDownload/classification/data/datasets2mmcls/cifar100_20200721/'

# V100服务器相对地址
# root = '/home/sr/datasets/DataDownload-master/classification/data/datasets2mmcls/caltech_101_20211007/'

root = '/jindofs_temp/nas/mnt/yindongshuo.yds/data/datasets2mmcls/cifar100_20200721/'

# dataset settings
dataset_type = 'CIFAR100N'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=32),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(32, -1)),
    dict(type='CenterCrop', crop_size=32),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_prefix=root,
        ann_file=root+'train2mmcls.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix=root,
        ann_file=root+'test2mmcls.txt',
        pipeline=test_pipeline),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        type=dataset_type,
        data_prefix=root,
        ann_file=root+'test2mmcls.txt',
        pipeline=test_pipeline))
# evaluation = dict(interval=1, metric='accuracy')
