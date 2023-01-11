import pickle
import numpy as np
import os
from os import path as osp
import mmcv
import xlwt
num_class = 22
classes = ['bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'chair', 'cup', 'curtain', 'desk', 'door', 'dresser',
                 'keyboard', 'lamp', 'laptop', 'monitor', 'night_stand', 'plant', 'sofa', 'stool', 'table', 'toilet',
                 'wardrobe']
cat2label = {cat: classes.index(cat) for cat in classes}
label2cat = {cat2label[t]: t for t in cat2label}
cat_ids = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])
cat_ids2class = {nyu40id: i for i,nyu40id in enumerate(list(cat_ids))}


def main():

    root_dir = '/data1/szh/mmdet_22'
    data_path = "/data1/szh/mmdet_22/scannet_instance_data/"
    all_scan_names = os.listdir(data_path)
    train_filenames = "/data/lxr/dataset/scannet/meta_data/scannetv2_train.txt"
    val_filenames = "/data/lxr/dataset/scannet/meta_data/scannetv2_val.txt"

    # with open(train_filenames, 'r') as f:
    #     train_scan_names = [name + '_aligned_bbox.npy' for name in f.read().splitlines()]
    # print('train_num_scans:', len(train_scan_names))
    #
    #
    # with open(val_filenames, 'r') as f:
    #     val_scan_names = [name + '_aligned_bbox.npy' for name in f.read().splitlines()]
    # print('val_num_scans:', len(val_scan_names))
    with open(train_filenames, 'r') as f:
        train_scan_names = [name + '_sem_label.npy' for name in f.read().splitlines()]
    print('train_num_scans:', len(train_scan_names))


    with open(val_filenames, 'r') as f:
        val_scan_names = [name + '_sem_label.npy' for name in f.read().splitlines()]
    print('val_num_scans:', len(val_scan_names))




    volume = {}
    for k in range(len(classes)):
        volume[classes[k]] = []

    class_num_point = np.zeros(22)


    infos_train = []
    for idx in range(len(train_scan_names)):
        scan_name_npz =train_scan_names[idx]
        semantic_labels = np.load(os.path.join(data_path, scan_name_npz)).astype('float64')
        # for i in range(instance_bboxes.shape[0]):
        #     for j in range(len(classes)):
        #         if instance_bboxes[i][6] == cat_ids[j]:
        #             volume[classes[j]].append(instance_bboxes[i][3] * instance_bboxes[i][4] * instance_bboxes[i][5])
        #             break
        for i in range(semantic_labels.shape[0]):
            for j in range(len(classes)):
                if semantic_labels[i] == cat_ids[j]:
                    class_num_point[j] = class_num_point[j] + 1
                    break
        print(f'{train_filenames} sample_idx: {scan_name_npz} {semantic_labels.shape[0]}')

    infos_val = []
    for idx in range(len(val_scan_names)):
        scan_name_npz = val_scan_names[idx]
        semantic_labels = np.load(os.path.join(data_path, scan_name_npz)).astype('float64')
        # for i in range(instance_bboxes.shape[0]):
        #     for j in range(len(classes)):
        #         if instance_bboxes[i][6] == cat_ids[j]:
        #             volume[classes[j]].append(instance_bboxes[i][3] * instance_bboxes[i][4] * instance_bboxes[i][5])
        #             break

        for i in range(semantic_labels.shape[0]):
            for j in range(len(classes)):
                if semantic_labels[i] == cat_ids[j]:
                    class_num_point[j] = class_num_point[j] + 1
                    break

        print(f'{val_filenames} sample_idx: {scan_name_npz}  {semantic_labels.shape[0]}')
    print(class_num_point)
    toscannet = xlwt.Workbook(encoding='utf-8', style_compression=0)
    sheet = toscannet.add_sheet('static', cell_overwrite_ok=True)
    col = ('class', 'class_num', 'volume_max', 'volume_min', 'volume_sum', 'volume_mean')
    for i in range(0, 6):
        sheet.write(0, i, col[i])
    #
    for i in range(22):
        sheet.write(i + 1, 1, class_num_point[i])
        sheet.write(i + 1, 0, classes[i])
    toscannet.save('/data1/szh/scannet_semantic.xls')
    # V_max = np.zeros(22)
    # V_min = np.zeros(22)
    # V_mean = np.zeros(22)
    # V_sum = np.zeros(22)
    # class_num = np.zeros(22)
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

    # toscannet = xlwt.Workbook(encoding='utf-8', style_compression=0)
    # sheet = toscannet.add_sheet('static', cell_overwrite_ok=True)
    # col = ('class', 'class_num', 'volume_max', 'volume_min', 'volume_sum', 'volume_mean')
    # for i in range(0, 6):
    #     sheet.write(0, i, col[i])
    #
    # for i in range(22):
    #     sheet.write(i + 1, 0, classes[i])
    #     sheet.write(i + 1, 1, class_num[i])
    #     sheet.write(i + 1, 2, V_max[i])
    #     sheet.write(i + 1, 3, V_min[i])
    #     sheet.write(i + 1, 4, V_sum[i])
    #     sheet.write(i + 1, 5, V_mean[i])
    # toscannet.save('/data1/szh/scannet_static.xls')

if __name__ == '__main__':
    main()

