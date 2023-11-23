# optimizer
# optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001) #bs=8*16=128 original
optimizer = dict(type='SGD', lr=0.0375, momentum=0.9, weight_decay=0.0001) #bs=3*16=48
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[70, 90])
runner = dict(type='EpochBasedRunner', max_epochs=100)


evaluation = dict(interval=10, metric='accuracy')