# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os

import argparse
import numpy as np
import PIL.Image as pil

from utils import readlines
from kitti_utils import generate_depth_map


def export_gt_depths_kitti_part2(lines,traj=0):


    gt_depths = []
    for line in lines:

        folder, frame_id, _ = line.split()
        frame_id = int(frame_id)

        split = "eigen_full"
        data_path = "/Users/ibrahimhassan/Documents/Documents/ekf-imu-depth-v2/kitti"
        
        calib_dir = os.path.join(data_path, folder.split("/")[0])
        velo_filename = os.path.join(data_path, folder,
                                        "velodyne_points/data", "{:010d}.bin".format(frame_id))
        gt_depth = generate_depth_map(calib_dir, velo_filename, 2, True)

        gt_depths.append(gt_depth.astype(np.float32))

    output_path = os.path.join(f"/Users/ibrahimhassan/Documents/Documents/ekf-imu-depth-v2/Part2/Traj_{traj}_gt", "gt_depths.npz")

    print("Saving to {}".format(output_path))

    np.savez_compressed(output_path, data=np.array(gt_depths))


