import os
import sys
import numpy as np
from plyfile import PlyData, PlyElement



def write_ply(points, filename, text=True):
    """ input: Nx3, write points to filename as PLY format. """
    points = [(points[i,0], points[i,1], points[i,2], points[i,3], points[i,4], points[i,5]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4'),('red', 'u1'), ('green', 'u1'),('blue', 'u1')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=text).write(filename)



if __name__ == '__main__':
    root_dir = '/data1/szh/mmdet_toscannet'
    data_path = '/data1/szh/TO-Scannet/to-scannet_dataset/data'
    all_scan_names = os.listdir(data_path)
    train_filenames = os.path.join('/data1/szh/TO-Scannet/meta_data/meta_data/TO-scannet/', 'train.txt')
    val_filenames = os.path.join('/data1/szh/TO-Scannet/meta_data/meta_data/TO-scannet/', 'val.txt')
    test_filenames = os.path.join('/data1/szh/TO-Scannet/meta_data/meta_data/TO-scannet/', 'test.txt')




    points = np.fromfile(os.path.join(data_path, 'scene0000_00_01.bin'), dtype=np.float32)
    points = points.reshape(-1, 6)
    write_ply(points, '/data1/szh/scan0000_00_001.ply')