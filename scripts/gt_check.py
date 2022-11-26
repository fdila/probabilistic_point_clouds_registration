import open3d as o3d 
import numpy as np
import pandas as pd
from os.path import join
import subprocess
import pickle
from math import pi

def get_pcd_name_darpa(pcd_name):
    pcd_name = pcd_name.replace('.', '')
    pcd_name = pcd_name[:-3]
    return pcd_name

def get_lidar2baseframe_transform():

    baseframe2cam = np.eye(4)
    baseframe2cam[:3,:3] = o3d.geometry.get_rotation_matrix_from_xyz([-pi/2, pi/2, 0])

    #T_ci (imu to cam0)
    T_c0i = np.array([[ 0.70992163, -0.70414167,  0.01399269,  0.01562753],
                        [ 0.02460003,  0.00493623, -0.99968519, -0.01981648],
                        [ 0.70385092,  0.71004236,  0.02082624, -0.07544143],
                        [ 0,          0,          0,          1.        ]])
    
    T_imu2os = np.eye(4)
    T_imu2os[0, 3] = -0.006253
    T_imu2os[1, 3] = 0.011775
    T_imu2os[2, 3] = 0.028535
    T_imu2os[:3, :3] = o3d.geometry.get_rotation_matrix_from_quaternion([0, 0, 0, 1])

    total_transform = baseframe2cam @ T_c0i @ T_imu2os
    return total_transform

def get_pcd_name_newer_college(sec, nsec):
    return "cloud_" + str(sec) + "_" + str(nsec)

if __name__ == '__main__':
    executable = '/root/repos/probabilistic_point_clouds_registration/build/probabilistic_benchmark'
    dataset_path = '/data/datasets/newer-college-dataset/2020-ouster-os1-64-realsense/01_short_experiment/raw_format/ouster_zip_files/ouster_scan-001/ouster_scan/'
    gt_path = '/data/datasets/newer-college-dataset/2020-ouster-os1-64-realsense/01_short_experiment/ground_truth/gt_0.csv'
    problems_path = '/data/datasets/newer-college-dataset/2020-ouster-os1-64-realsense/01_short_experiment/problems_0.txt'
    output_path = '/data/datasets/newer-college-dataset/2020-ouster-os1-64-realsense/01_short_experiment/gt_check_0.dat'
    
    dataset_type = 'newer_college'

    if dataset_type == 'darpa':
        gt_df = pd.read_csv(
                gt_path,
                names=['ts', 'x', 'y', 'z', 'q_x', 'q_y', 'q_z', 'q_w'],
                sep=' ',
                dtype={'ts': 'str'})
        gt_df['id'] = gt_df.apply(lambda row: get_pcd_name_darpa(row['ts']),
                                    axis=1)
    elif dataset_type == 'newer_college':
        gt_df = pd.read_csv(
            gt_path,
            names=['sec', 'nsec', 'x', 'y', 'z', 'q_x', 'q_y', 'q_z', 'q_w'],
            sep=',',
            dtype={
                'sec': 'str',
                'nsec': 'str'
            },
            comment='#')
        gt_df['id'] = gt_df.apply(
            lambda row: get_pcd_name_newer_college(row['sec'], row['nsec']),
            axis=1)
    
    problems_df = pd.read_csv(problems_path,
                     sep=r'\s*,\s*',
                     engine='python',
                     dtype=str)

    results = []
    T_source2lidar = get_lidar2baseframe_transform()
    for index, row in problems_df.iterrows():

        if dataset_type == 'darpa':
            source_name = row['source'].replace('.', '') + '.pcd'
            target_name = row['target'].replace('.', '') + '.pcd'
        elif dataset_type == 'newer_college':
            source_name = 'cloud_' + row['source'] + '.pcd'
            target_name = 'cloud_' + row['target'] + '.pcd'

        source = o3d.io.read_point_cloud(join(dataset_path, source_name))
        target = o3d.io.read_point_cloud(join(dataset_path, target_name))

        if dataset_type == 'darpa':
            source_gt = gt_df.loc[gt_df['id'] == row['source']]
            target_gt = gt_df.loc[gt_df['id'] == row['target']]
        elif dataset_type == 'newer_college':
            source_gt = gt_df.loc[gt_df['id'] == 'cloud_' + row['source']]
            target_gt = gt_df.loc[gt_df['id'] == 'cloud_' + row['target']]

        T_map2source_gt = np.eye(4)
        T_map2source_gt[0, 3] = source_gt['x'].values[0]
        T_map2source_gt[1, 3] = source_gt['y'].values[0]
        T_map2source_gt[2, 3] = source_gt['z'].values[0]
        T_map2source_gt[:3, :
                        3] = o3d.geometry.get_rotation_matrix_from_quaternion([
                            source_gt['q_w'], source_gt['q_x'],
                            source_gt['q_y'], source_gt['q_z']
                        ])

        T_map2target_gt = np.eye(4)
        T_map2target_gt[0, 3] = target_gt['x'].values[0]
        T_map2target_gt[1, 3] = target_gt['y'].values[0]
        T_map2target_gt[2, 3] = target_gt['z'].values[0]
        T_map2target_gt[:3, :3] = o3d.geometry.get_rotation_matrix_from_quaternion([
            target_gt['q_w'], target_gt['q_x'],
            target_gt['q_y'], target_gt['q_z']])
        
        if dataset_type == 'darpa':
            source = source.transform(T_map2source_gt)
            target = target.transform(T_map2target_gt)
        elif dataset_type == 'newer_college':
            source = source.transform(T_map2source_gt @ T_source2lidar)
            target = target.transform(T_map2target_gt @ T_source2lidar)
        
        o3d.io.write_point_cloud('/tmp/source.pcd', source)
        o3d.io.write_point_cloud('/tmp/target.pcd', target)

        command = [executable, "-r 0.1","-m 10","/tmp/source.pcd","/tmp/target.pcd"]
        print(command)
        result = subprocess.check_output(command).decode("utf8").split(", ")
        result = [float(x) for x in result]
        results.append(result)
        print(result)

        with open(output_path, 'wb') as out_file:
            pickle.dump(command, out_file)
            pickle.dump(results, out_file)


