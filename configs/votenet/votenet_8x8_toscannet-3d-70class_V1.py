_base_ = [
    '../_base_/datasets/toscannet-3d-70class.py', '../_base_/models/votenet_toscannet_V1.py',
    '../_base_/schedules/schedule_3x.py', '../_base_/default_runtime.py'
]

# model settings
model = dict(
    bbox_head=dict(
        num_classes=70,
        bbox_coder=dict(
            type='PartialBinBasedBBoxCoder',
            num_sizes=70,
            num_dir_bins=1,
            with_rot=False,
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
                           [0.19504214, 0.17907377, 0.08079748]])))

# yapf:disable
log_config = dict(interval=30)
# yapf:enable
