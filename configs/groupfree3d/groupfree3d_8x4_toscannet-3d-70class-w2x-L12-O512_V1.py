_base_ = [
    '../_base_/datasets/toscannet-3d-70class.py',
    '../_base_/models/groupfree3d.py', '../_base_/schedules/schedule_3x.py',
    '../_base_/default_runtime.py'
]

# model settings
model = dict(
    backbone=dict(
        type='PointNet2SASSG',
        in_channels=3,
        num_points=(4096, 2048, 1024, 512),
        radius=(0.06, 0.25, 0.8, 1.2),
        num_samples=(64, 32, 16, 16),
        sa_channels=((128, 128, 256), (256, 256, 512), (256, 256, 512),
                     (256, 256, 512)),
        fp_channels=((512, 512), (512, 288)),
        norm_cfg=dict(type='BN2d'),
        sa_cfg=dict(
            type='PointSAModule',
            pool_mod='max',
            use_xyz=True,
            normalize_xyz=True)),
    bbox_head=dict(
        num_classes=70,
        num_decoder_layers=12,
        num_proposal=512,
        size_cls_agnostic=False,
        bbox_coder=dict(
            type='GroupFree3DBBoxCoder',
            num_sizes=70,
            num_dir_bins=1,
            with_rot=False,
            size_cls_agnostic=False,
            mean_sizes=[[0.75224076, 0.95897522, 0.95615736],
                           [1.78487791, 1.90840145, 1.13032958],
                           [0.62493454, 0.63033143, 0.70454235],
                           [1.45252933, 1.49741788, 0.82573491],
                           [0.97253736, 1.03950483, 0.62091357],
                           [0.57974676, 0.59866436, 1.75381654],
                           [1.30051826, 0.79971739, 1.23022978],
                           [0.85399874, 1.23248614, 1.66444992],
                           [0.21648139, 0.55280621, 0.61596324],
                           [1.275056  , 1.87434515, 0.25730364],
                           [1.03770949, 1.46746293, 0.8628066 ],
                           [1.44159924, 0.89343219, 1.62176913],
                           [0.63881446, 0.71485345, 1.37278993],
                           [0.3989548 , 0.39051149, 1.62742053],
                           [0.6023989 , 0.5958184 , 0.76425494],
                           [0.51142132, 0.51575776, 0.28226824],
                           [1.19828428, 1.04876873, 0.51574445],
                           [0.59937902, 0.56499158, 0.62321182],
                           [0.19000893, 0.19459082, 0.25698144],
                           [0.07626887, 0.07621189, 0.21926467],
                           [0.15710938, 0.15662019, 0.06197302],
                           [0.14460595, 0.15488624, 0.11060956],
                           [0.07469217, 0.07473594, 0.12507552],
                           [0.21965882, 0.21220613, 0.09652358],
                           [0.11399773, 0.1384257 , 0.18130018],
                           [0.27412293, 0.28684345, 0.02959056],
                           [0.25168264, 0.29472433, 0.29462903],
                           [0.17792574, 0.17499026, 0.07605004],
                           [0.13420875, 0.13507012, 0.23080428],
                           [0.17044723, 0.13477129, 0.01945199],
                           [0.18963662, 0.18633008, 0.35312872],
                           [0.32199187, 0.32643373, 0.20860781],
                           [0.14932994, 0.14189514, 0.16355464],
                           [0.28693407, 0.31837955, 0.1949697 ],
                           [0.10523664, 0.10895999, 0.10240284],
                           [0.3177627 , 0.32848609, 0.17334262],
                           [0.13071464, 0.11728173, 0.02350202],
                           [0.12887875, 0.12195973, 0.01863764],
                           [0.11800063, 0.13572134, 0.13910698],
                           [0.21458858, 0.20712223, 0.04192078],
                           [0.25425695, 0.25343559, 0.21480093],
                           [0.16946746, 0.16045822, 0.02840589],
                           [0.10020675, 0.09875793, 0.28235359],
                           [0.08797825, 0.08816052, 0.03792901],
                           [0.37941142, 0.38509657, 0.08005531],
                           [0.30384149, 0.27920013, 0.32247705],
                           [0.10235061, 0.11636419, 0.03117593],
                           [0.29520188, 0.32043443, 0.05667629],
                           [0.25773961, 0.26565499, 0.07573267],
                           [0.11332224, 0.12746926, 0.24915252],
                           [0.06069286, 0.05804837, 0.01917328],
                           [0.1556136 , 0.16013133, 0.05140929],
                           [0.26786305, 0.29499025, 0.27577095],
                           [0.14024703, 0.11015199, 0.0239145 ],
                           [0.10723671, 0.10744343, 0.09621158],
                           [0.20435824, 0.20951888, 0.26209073],
                           [0.22845026, 0.222675  , 0.10810672],
                           [0.08914751, 0.14451474, 0.22460698],
                           [0.20800873, 0.20257084, 0.02235973],
                           [0.13163254, 0.10251354, 0.018969  ],
                           [0.21345608, 0.21448668, 0.32671038],
                           [0.23196087, 0.23148341, 0.02379998],
                           [0.1749174 , 0.17826692, 0.16815628],
                           [0.16482577, 0.14308309, 0.01382655],
                           [0.27596562, 0.27450172, 0.18702465],
                           [0.13640101, 0.11890776, 0.02713102],
                           [0.20156312, 0.20813729, 0.16506436],
                           [0.29378766, 0.28393607, 0.21726394],
                           [0.14161967, 0.14055382, 0.26736649],
                           [0.19504214, 0.17907377, 0.08079748]]),
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
data_root = '/data1/szh/mmdet_toscannet/'
class_names = ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door',
                            'window', 'bookshelf','picture', 'counter', 'desk', 'curtain',
                            'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub', 'garbagebin',
                            "bag", "bottle", "bowl", "camera", "can",
                            "cap", "clock", "keyboard", "display", "earphone",
                            "jar", "knife", "lamp", "laptop", "microphone",
                            "microwave", "mug", "printer", "remote control", "phone",
                            "alarm", "book", "cake", "calculator", "candle",
                            "charger", "chessboard", "coffee_machine", "comb", "cutting_board",
                            "dishes", "doll", "eraser", "eye_glasses", "file_box",
                            "fork",  "fruit", "globe", "hat", "mirror",
                            "notebook", "pencil", "plant", "plate", "radio",
                            "ruler", "saucepan", "spoon", "tea_pot", "toaster",
                            "vase", "vegetables")
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
        valid_cat_ids=(3,4,5,6,7,8,9,10,11,12,14,16,24,28,33,34,36,39,
                                    41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
                                    59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75,
                                    76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 92, 93)),
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
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'toscannet_infos_train.pkl',
            pipeline=train_pipeline,
            filter_empty_gt=False,
            classes=class_names,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d='Depth')),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'toscannet_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'toscannet_infos_val.pkl',
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
runner = dict(type='EpochBasedRunner', max_epochs=80)
checkpoint_config = dict(interval=1, max_keep_ckpts=10)
