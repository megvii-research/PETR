_base_ = [
    '../../../mmdetection3d/configs/_base_/datasets/nus-3d.py',
    '../../../mmdetection3d/configs/_base_/default_runtime.py'
]
backbone_norm_cfg = dict(type='LN', requires_grad=True)
plugin=True
plugin_dir='projects/mmdet3d_plugin/'

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[57.375, 57.120, 58.395], to_rgb=False)
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)
model = dict(
    type='Petr3D',
    use_grid_mask=True,
    img_backbone=dict(
        type='VoVNetCP', ###use checkpoint to save memory, you can use 'VoVNet' instead of 'VoVNetCP' to close checkpoint
        spec_name='V-99-eSE',
        norm_eval=True,
        frozen_stages=-1,
        input_ch=3,
        out_features=('stage4','stage5',)),
    img_neck=dict(
        type='CPFPN',  ###remove unused parameters 
        in_channels=[768, 1024],
        out_channels=256,
        num_outs=2),
    pts_bbox_head=dict(
        type='PETRv2Head',
        num_classes=10,
        in_channels=256,
        num_query=1500,
        LID=True,
        with_position=True,
        with_multiview=True,
        with_fpe=True,
        with_time=True,
        with_multi=True,
        position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        transformer=dict(
            type='PETRTransformer',
            decoder=dict(
                type='PETRTransformerDecoder',
                return_intermediate=True,
                num_layers=6,
                transformerlayers=dict(
                    type='PETRTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='PETRMultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        ],
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    with_cp=True,  ###use checkpoint to save memory
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')),
            )),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=10), 
        positional_encoding=dict(
            type='SinePositionalEncoding3D', num_feats=128, normalize=True),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)),
    # model training and testing settings
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0), # Fake cost. This is just to make it compatible with DETR head. 
            pc_range=point_cloud_range))))

dataset_type = 'CustomNuScenesDataset'
data_root = '/data/Dataset/nuScenes/'

file_client_args = dict(backend='disk')

db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'nuscenes_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(
            car=5,
            truck=5,
            bus=5,
            trailer=5,
            construction_vehicle=5,
            traffic_cone=5,
            barrier=5,
            motorcycle=5,
            bicycle=5,
            pedestrian=5)),
    classes=class_names,
    sample_groups=dict(
        car=2,
        truck=3,
        construction_vehicle=7,
        bus=4,
        trailer=6,
        barrier=2,
        motorcycle=6,
        bicycle=6,
        pedestrian=2,
        traffic_cone=2),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args))
ida_aug_conf = {
        "resize_lim": (0.94, 1.25),
        "final_dim": (640, 1600),
        "bot_pct_lim": (0.0, 0.0),
        "rot_lim": (0.0, 0.0),
        "H": 900,
        "W": 1600,
        "rand_flip": True,
    }
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadMultiViewImageFromMultiSweepsFiles', sweeps_num=1, to_float32=True, pad_empty_sweeps=True, test_mode=False, sweep_range=[3,27]),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='ResizeCropFlipImage', data_aug_conf = ida_aug_conf, training=True),
    dict(type='GlobalRotScaleTransImage',
            rot_range=[-0.3925, 0.3925],
            translation_std=[0, 0, 0],
            scale_ratio_range=[0.95, 1.05],
            reverse_angle=True,
            training=True
            ),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'],
            meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img', 'intrinsics', 'extrinsics',
                'pad_shape', 'scale_factor', 'flip', 'box_mode_3d', 'box_type_3d',
                'img_norm_cfg', 'sample_idx', 'timestamp'))
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadMultiViewImageFromMultiSweepsFiles', sweeps_num=1, to_float32=True, pad_empty_sweeps=True, sweep_range=[3,27]),
    dict(type='ResizeCropFlipImage', data_aug_conf = ida_aug_conf, training=False),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['img'],
            meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img', 'intrinsics', 'extrinsics',
                'pad_shape', 'scale_factor', 'flip', 'box_mode_3d', 'box_type_3d',
                'img_norm_cfg', 'sample_idx', 'timestamp'))
        ])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type='CBGSDataset',
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=[data_root + 'mmdet3d_nuscenes_30f_infos_train.pkl', data_root + 'mmdet3d_nuscenes_30f_infos_val.pkl'],
            pipeline=train_pipeline,
            classes=class_names,
            modality=input_modality,
            test_mode=False,
            use_valid_flag=True,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d='LiDAR'),
    ),
    val=dict(type=dataset_type, pipeline=test_pipeline, ann_file=data_root + 'mmdet3d_nuscenes_30f_infos_val.pkl', classes=class_names, modality=input_modality),
    test=dict(type=dataset_type, pipeline=test_pipeline, ann_file=data_root + 'mmdet3d_nuscenes_30f_infos_test.pkl', classes=class_names, modality=input_modality))


optimizer = dict(
    type='AdamW', 
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)

optimizer_config = dict(type='Fp16OptimizerHook', loss_scale=512., grad_clip=dict(max_norm=35, norm_type=2))

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
    )
total_epochs = 24
evaluation = dict(interval=24, pipeline=test_pipeline)
find_unused_parameters=False #### when use checkpoint, find_unused_parameters must be False
checkpoint_config = dict(interval=1, max_keep_ckpts=3)
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
load_from='ckpts/dd3d_det_final.pth'
resume_from=None


# Evaluating bboxes of pts_bbox
# add center
# mAP: 0.8412
# mATE: 0.3252
# mASE: 0.1556
# mAOE: 0.0708
# mAVE: 0.1758
# mAAE: 0.1784
# NDS: 0.8300
# Eval time: 192.6s

# Per-class results:
# Object Class    AP      ATE     ASE     AOE     AVE     AAE
# car     0.863   0.256   0.119   0.033   0.177   0.196
# truck   0.840   0.309   0.124   0.031   0.145   0.215
# bus     0.845   0.344   0.125   0.031   0.268   0.210
# trailer 0.779   0.430   0.135   0.042   0.111   0.095
# construction_vehicle    0.799   0.410   0.196   0.062   0.135   0.306
# pedestrian      0.815   0.370   0.212   0.195   0.196   0.141
# motorcycle      0.826   0.340   0.166   0.098   0.257   0.241
# bicycle 0.880   0.276   0.182   0.093   0.117   0.023
# traffic_cone    0.884   0.238   0.174   nan     nan     nan
# barrier 0.881   0.280   0.122   0.052   nan     nan

