import pickle
import subprocess
from os.path import join

import open3d as o3d
import pandas as pd

import fdila.datasets.darpa as darpa
import fdila.datasets.newer_college as newer_college
from fdila.pointcloud.misc import remove_zero_points

from tqdm import tqdm

if __name__ == '__main__':
    executable = '/root/repos/probabilistic_point_clouds_registration/build/probabilistic_benchmark'

    dataset_path = '/data/datasets/newer-college-dataset/2020-ouster-os1-64-realsense/01_short_experiment/raw_format/ouster_zip_files/ouster_scan/'
    gt_path = '/data/datasets/newer-college-dataset/2020-ouster-os1-64-realsense/01_short_experiment/ground_truth/registered_poses.csv'
    problems_path = '/data/datasets/newer-college-dataset/2020-ouster-os1-64-realsense/01_short_experiment/problems_full_dataset.txt'
    output_path = '/data/datasets/newer-college-dataset/2020-ouster-os1-64-realsense/01_short_experiment/gt_check_full_dataset_kalibr_trimmed_r01.dat'

    dataset_type = 'newer_college'

    if dataset_type == 'darpa':
        gt_df = darpa.get_ground_truth_df(gt_path)
    elif dataset_type == 'newer_college':
        gt_df = newer_college.get_ground_truth_df(gt_path)
        T_source2lidar = newer_college.get_base2lidar_transform_kalibr()

    problems_df = pd.read_csv(problems_path,
                              sep=r'\s*,\s*',
                              engine='python',
                              dtype=str)

    results = []
    for index, row in tqdm(problems_df.iterrows(), total=problems_df.shape[0]):

        if dataset_type == 'darpa':
            source_name = row['source'].replace('.', '') + '.pcd'
            target_name = row['target'].replace('.', '') + '.pcd'
        elif dataset_type == 'newer_college':
            source_name = row['source'] + '.pcd'
            target_name = row['target'] + '.pcd'

        source = o3d.t.io.read_point_cloud(join(dataset_path, source_name))
        target = o3d.t.io.read_point_cloud(join(dataset_path, target_name))

        source = remove_zero_points(source)
        target = remove_zero_points(target)

        if dataset_type == 'darpa':
            T_map2source_gt = darpa.get_transform_from_gt(gt_df, row['source'])
            T_map2target_gt = darpa.get_transform_from_gt(gt_df, row['target'])
        elif dataset_type == 'newer_college':
            T_map2source_gt = newer_college.get_transform_from_gt(
                gt_df, row['source'])
            T_map2target_gt = newer_college.get_transform_from_gt(
                gt_df, row['target'])

        if dataset_type == 'darpa':
            source = source.transform(T_map2source_gt)
            target = target.transform(T_map2target_gt)
        elif dataset_type == 'newer_college':
            source = source.transform(T_map2source_gt @ T_source2lidar)
            target = target.transform(T_map2target_gt @ T_source2lidar)

        o3d.t.io.write_point_cloud('/tmp/source.pcd', source)
        o3d.t.io.write_point_cloud('/tmp/target.pcd', target)

        command = [executable, "-r 0.1", "-m 20",
                   "/tmp/source.pcd", "/tmp/target.pcd"]
        print(command)
        result = subprocess.check_output(command).decode("utf8").split(", ")
        result = [float(x) for x in result]
        results.append(result)
        print(result)

    with open(output_path, 'wb') as out_file:
        pickle.dump(command, out_file)
        pickle.dump(results, out_file)
