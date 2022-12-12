_base_ = [
    '../_base_/datasets/scannet-3d-22class.py',
    '../_base_/models/groupfree3d.py', '../_base_/schedules/schedule_3x.py',
    '../_base_/default_runtime.py'
]

# model settings
model = dict(
    bbox_head=dict(
        num_classes=22,
        num_decoder_layers=12,
        size_cls_agnostic=False,
        bbox_coder=dict(
            type='GroupFree3DBBoxCoder',
            num_sizes=22,
            num_dir_bins=1,
            with_rot=False,
            size_cls_agnostic=False,
            mean_sizes=[[1.1853635,  1.08464234, 0.5300481 ],
                        [1.88765612, 1.85415067, 1.19528733],
                        [0.85874458, 1.08556705, 0.4470463 ],
                         [0.75720005, 1.04799915, 1.34582313],
                         [0.12992612, 0.13222266, 0.23328308],
                         [0.61181084, 0.62051982, 0.70508617],
                         [0.13554971, 0.13481881, 0.1222724 ],
                         [1.14209621, 0.69467313, 1.64300397],
                         [1.08016621, 1.49774906, 0.86502063],
                         [0.56031829, 0.60526205, 1.71870926],
                         [0.67985325, 0.92551969, 0.95939219],
                         [0.2944424,  0.40241896, 0.0523093 ],
                         [0.32375328, 0.3337317,  0.49952993],
                         [0.39801043, 0.40996502, 0.21692318],
                         [0.31293968, 0.49079205, 0.44510456],
                         [0.54616412, 0.58355285, 0.66404205],
                         [0.46672829, 0.45970987, 0.67335616],
                         [1.44830543, 1.57713572, 0.83410145],
                         [0.52950791, 0.53263195, 0.47203752],
                         [1.07988677, 1.28553691, 0.58918576],
                         [0.57768444, 0.59243448, 0.72572129],
                         [0.99421792, 1.20236225, 1.95497781]]),
        sampling_objectness_loss=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=8.0),
        objectness_loss=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        center_loss=dict(
            type='SmoothL1Loss', beta=0.04, reduction='sum', loss_weight=10.0),
        dir_class_loss=dict(
            type='CrossEntropyLoss', reduction='sum', loss_weight=1.0),
        dir_res_loss=dict(
            type='SmoothL1Loss', reduction='sum', loss_weight=10.0),
        size_class_loss=dict(
            type='CrossEntropyLoss', reduction='sum', loss_weight=1.0),
        size_res_loss=dict(
            type='SmoothL1Loss',
            beta=1.0 / 9.0,
            reduction='sum',
            loss_weight=10.0 / 9.0),
        semantic_loss=dict(
            type='CrossEntropyLoss', reduction='sum', loss_weight=1.0)),
    test_cfg=dict(
        sample_mod='kps',
        nms_thr=0.25,
        score_thr=0.0,
        per_class_proposal=True,
        prediction_stages='last_three'))

# dataset settings
dataset_type = 'ScanNetDataset'
data_root = '/data1/szh/fcaf3d_22_1208_50000/'
class_names = ('bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'chair', 'cup', 'curtain', 'desk', 'door', 'dresser',
                 'keyboard', 'lamp', 'laptop', 'monitor', 'night_stand', 'plant', 'sofa', 'stool', 'table', 'toilet',
                 'wardrobe')
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        load_dim=6,
        use_dim=[0, 1, 2]),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_mask_3d=True,
        with_seg_3d=True),
    dict(type='GlobalAlignment', rotation_axis=2),
    dict(
        type='PointSegClassMapping',
        valid_cat_ids=(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23)),
    dict(type='PointSample', num_points=50000),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.087266, 0.087266],
        scale_ratio_range=[1.0, 1.0]),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D',
        keys=[
            'points', 'gt_bboxes_3d', 'gt_labels_3d', 'pts_semantic_mask',
            'pts_instance_mask'
        ])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        load_dim=6,
        use_dim=[0, 1, 2]),
    dict(type='GlobalAlignment', rotation_axis=2),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(
                type='RandomFlip3D',
                sync_2d=False,
                flip_ratio_bev_horizontal=0.5,
                flip_ratio_bev_vertical=0.5),
            dict(type='PointSample', num_points=50000),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'scannet_infos_train.pkl',
            pipeline=train_pipeline,
            filter_empty_gt=False,
            classes=class_names,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d='Depth')),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'scannet_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'scannet_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth'))

# optimizer
lr = 0.006
optimizer = dict(
    lr=lr,
    weight_decay=0.0005,
    paramwise_cfg=dict(
        custom_keys={
            'bbox_head.decoder_layers': dict(lr_mult=0.1, decay_mult=1.0),
            'bbox_head.decoder_self_posembeds': dict(
                lr_mult=0.1, decay_mult=1.0),
            'bbox_head.decoder_cross_posembeds': dict(
                lr_mult=0.1, decay_mult=1.0),
            'bbox_head.decoder_query_proj': dict(lr_mult=0.1, decay_mult=1.0),
            'bbox_head.decoder_key_proj': dict(lr_mult=0.1, decay_mult=1.0)
        }))

optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
lr_config = dict(policy='step', warmup=None, step=[56, 68])

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=400)
checkpoint_config = dict(interval=1, max_keep_ckpts=100)
