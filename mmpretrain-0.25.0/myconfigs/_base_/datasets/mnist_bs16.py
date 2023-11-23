# dataset settings
dataset_type = 'MNIST'
root = '/home/sr/mmclass/mmpretrain-0.25.0/'
img_norm_cfg = dict(
    mean=[125.307, 122.961, 113.8575],
    std=[51.5865, 50.847, 51.255],
    to_rgb=False
)
train_pipeline = [
    # dict(type='RandomCrop', size=32, padding=4),
    dict(type='Resize', size=32),
    # dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    # dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='Resize', size=32),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    # dict(type='Collect', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type, data_prefix=root+'data/mnist',
        pipeline=train_pipeline
    ),
    val=dict(
        type=dataset_type,
        data_prefix=root+'data/mnist',
        pipeline=test_pipeline,
        test_mode=True),
    test=dict(
        type=dataset_type,
        data_prefix=root+'data/mnist',
        pipeline=test_pipeline,
        test_mode=True))
