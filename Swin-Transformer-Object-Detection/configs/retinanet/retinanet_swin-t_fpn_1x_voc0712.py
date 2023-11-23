_base_ = [
    './retinanet_swin-t_fpn.py',
    '../_base_/datasets/voc0712.py',
    '../_base_/default_runtime.py',
]
model = dict(bbox_head=dict(num_classes=20))
# optimizer
# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer = dict(type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05, )
optimizer_config = dict(grad_clip=None)
# learning policy
# actual epoch = 3 * 3 = 9
lr_config = dict(policy='step', step=[3])
# runtime settings
runner = dict(
    type='EpochBasedRunner', max_epochs=4)  # actual epoch = 4 * 3 = 12

fp16 = dict()
