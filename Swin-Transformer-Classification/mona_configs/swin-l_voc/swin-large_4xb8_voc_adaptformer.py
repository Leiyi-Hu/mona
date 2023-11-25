# Only for evaluation
_base_ = [
    "../_base_/models/swin_transformer/large_224_det.py",
    "../_base_/datasets/voc_mona.py",
    "../_base_/schedules/imagenet_bs1024_adamw_swin.py",
    "../_base_/default_runtime.py",
]
load_from = "/pretrained_model/swin_large_patch4_window7_224_22k.pth"

bs_per_gpu = 4
gpu_num = 4

model = dict(
    backbone=dict(type="SwinTransformer_adaptformer"), head=dict(num_classes=20)
)
data = dict(samples_per_gpu=bs_per_gpu)
optimizer = dict(
    lr=5e-4 * bs_per_gpu * gpu_num / 512,
    weight_decay=0.05,
    eps=1e-8,
    betas=(0.9, 0.999),
)

runner = dict(type="EpochBasedRunner", max_epochs=100)

fp16 = dict(loss_scale="dynamic")

checkpoint_config = dict(interval=101)
