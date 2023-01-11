import pickle
import numpy as np
import os
from os import path as osp
import mmcv
import xlwt
num_class = 70
type2class = {'cabinet':0, 'bed':1, 'chair':2, 'sofa':3, 'table':4, 'door':5,
                            'window':6,'bookshelf':7,'picture':8, 'counter':9, 'desk':10, 'curtain':11,
                            'refrigerator':12, 'showercurtrain':13, 'toilet':14, 'sink':15, 'bathtub':16, 'garbagebin':17,
                            "bag":18, "bottle":19, "bowl":20, "camera":21, "can":22,
                            "cap":23, "clock":24, "keyboard":25, "display":26, "earphone":27,
                            "jar":28, "knife":29, "lamp":30, "laptop":31, "microphone":32,
                            "microwave":33, "mug":34, "printer":35, "remote control":36, "phone":37,
                            "alarm":38, "book":39, "cake":40, "calculator":41, "candle":42,
                            "charger":43, "chessboard":44, "coffee_machine":45, "comb":46, "cutting_board":47,
                            "dishes":48, "doll":49, "eraser":50, "eye_glasses":51, "file_box":52,
                            "fork":53, "fruit":54, "globe":55, "hat":56, "mirror":57,
                            "notebook":58, "pencil":59, "plant":60, "plate":61, "radio":62,
                            "ruler":63, "saucepan":64, "spoon":65, "tea_pot":66, "toaster":67,
                            "vase":68, "vegetables":69}
classes = ['cabinet', 'bed', 'chair', 'sofa', 'table', 'door',
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
                            "vase", "vegetables"]
cat2label = {cat: classes.index(cat) for cat in classes}
label2cat = {cat2label[t]: t for t in cat2label}
cat_ids = np.array([3,4,5,6,7,8,9,10,11,12,14,16,24,28,33,34,36,39,
                                    41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
                                    59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75,
                                    76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 92, 93])
cat_ids2class = {nyu40id: i for i,nyu40id in enumerate(list(cat_ids))}




def main():
    root_dir = '/data1/szh/mmdet_toscannet'
    data_path = "/data1/szh/mmdet_toscannet/semantic_mask/"    #  '/data1/szh/TO-Scannet/to-scannet_dataset/data'
    all_scan_names = os.listdir(data_path)
    train_filenames = os.path.join('/data1/szh/TO-Scannet/meta_data/meta_data/TO-scannet/', 'train.txt')
    val_filenames = os.path.join('/data1/szh/TO-Scannet/meta_data/meta_data/TO-scannet/', 'val.txt')
    test_filenames = os.path.join('/data1/szh/TO-Scannet/meta_data/meta_data/TO-scannet/', 'test.txt')
    # with open(train_filenames, 'r') as f:
    #     train_scan_names = [name + '.npz' for name in f.read().splitlines()]
    # print('train_num_scans:', len(train_scan_names))
    #
    #
    # with open(val_filenames, 'r') as f:
    #     val_scan_names = [name + '.npz' for name in f.read().splitlines()]
    # print('val_num_scans:', len(val_scan_names))

    with open(train_filenames, 'r') as f:
        train_scan_names = [name + '.bin' for name in f.read().splitlines()]
    print('train_num_scans:', len(train_scan_names))


    with open(val_filenames, 'r') as f:
        val_scan_names = [name + '.bin' for name in f.read().splitlines()]
    print('val_num_scans:', len(val_scan_names))



    volume = {}
    for k in range(len(classes)):
        volume[classes[k]] = []

    class_num_point = np.zeros(70)
    infos_train = []
    for idx in range(len(train_scan_names)):
        scan_name_npz =train_scan_names[idx]
        # instance_bboxes = np.load(os.path.join(data_path, scan_name_npz))['bbox'].astype('float64')
        # for i in range(instance_bboxes.shape[0]):
        #     for j in range(len(classes)):
        #         if instance_bboxes[i][6] == cat_ids[j]:
        #             volume[classes[j]].append(instance_bboxes[i][3] * instance_bboxes[i][4] * instance_bboxes[i][5])
        #             break

        semantic_labels = np.fromfile(os.path.join(data_path, scan_name_npz), dtype=np.uint32)
        for i in range(semantic_labels.shape[0]):
            for j in range(len(classes)):
                if semantic_labels[i] == cat_ids[j]:
                    class_num_point[j] = class_num_point[j] + 1
                    break
        print(f'{train_filenames} sample_idx: {scan_name_npz}  {semantic_labels.shape[0]}')

    infos_val = []
    for idx in range(len(val_scan_names)):
        scan_name_npz = val_scan_names[idx]
        # instance_bboxes = np.load(os.path.join(data_path, scan_name_npz))['bbox'].astype('float64')
        # for i in range(instance_bboxes.shape[0]):
        #     for j in range(len(classes)):
        #         if instance_bboxes[i][6] == cat_ids[j]:
        #             volume[classes[j]].append(instance_bboxes[i][3] * instance_bboxes[i][4] * instance_bboxes[i][5])
        #             break
        semantic_labels = np.fromfile(os.path.join(data_path, scan_name_npz), dtype=np.uint32)
        for i in range(semantic_labels.shape[0]):
            for j in range(len(classes)):
                if semantic_labels[i] == cat_ids[j]:
                    class_num_point[j] = class_num_point[j] + 1
                    break

        print(f'{val_filenames} sample_idx: {scan_name_npz}  {semantic_labels.shape[0]}')

    print(class_num_point)


    # V_max = np.zeros(70)
    # V_min = np.zeros(70)
    # V_mean = np.zeros(70)
    # V_sum = np.zeros(70)
    # class_num = np.zeros(70)
    # for n in range(len(classes)):
    #     V_max[n] = max(volume[classes[n]])
    #     V_min[n] = min(volume[classes[n]])
    #     for m in range(len(volume[classes[n]])):
    #         V_sum[n] = V_sum[n] + volume[classes[n]][m]
    #     V_mean[n] = V_sum[n] / len(volume[classes[n]])
    #     class_num[n] = len(volume[classes[n]])
    #
    # print(V_max)
    # print(V_min)
    # print(V_mean)
    # print(class_num)

    toscannet = xlwt.Workbook(encoding='utf-8', style_compression=0)
    sheet = toscannet.add_sheet('static', cell_overwrite_ok=True)
    col = ('class', 'class_num', 'volume_max', 'volume_min', 'volume_sum', 'volume_mean')
    for i in range(0, 6):
        sheet.write(0, i, col[i])
    #
    for i in range(70):
        sheet.write(i + 1, 1, class_num_point[i])
        sheet.write(i + 1, 0, classes[i])
    #     sheet.write(i + 1, 1, class_num[i])
    #     sheet.write(i + 1, 2, V_max[i])
    #     sheet.write(i + 1, 3, V_min[i])
    #     sheet.write(i + 1, 4, V_sum[i])
    #     sheet.write(i + 1, 5, V_mean[i])
    toscannet.save('/data1/szh/toscannet_semantic.xls')

if __name__ == '__main__':
    main()

