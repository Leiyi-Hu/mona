# model settings
# Only for evaluation
model = dict(
    type="ImageClassifier",
    backbone=dict(
        type="SwinTransformer",
        embed_dim=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=7,
        ape=False,
        drop_path_rate=0.3,
        patch_norm=True,
        use_checkpoint=False,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        out_indices=(3,),
    ),
    neck=dict(type="GlobalAveragePooling"),
    head=dict(
        type="LinearClsHead",
        num_classes=1000,
        in_channels=1536,
        loss=dict(type="CrossEntropyLoss", loss_weight=1.0),
        topk=(1, 5),
    ),
)
