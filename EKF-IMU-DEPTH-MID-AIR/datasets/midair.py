# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil
from torchvision import transforms
from PIL import Image  # using pillow-simd for increased speed
import scipy.misc as misc

# from kitti_utils import generate_depth_map
from .mono_dataset import MonoDataset
# from mono_dataset import MonoDataset

class MidAirDataset(MonoDataset):
    """Class for MidAir dataset loader
        """

    def __init__(self, *args, **kwargs):
        super(MidAirDataset, self).__init__(*args, **kwargs)

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size
        # self.K = np.array([[0.5, 0, 0.5, 0],
        #                    [0, 0.5, 0.5, 0],
        #                    [0, 0, 1, 0],
        #                    [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (512, 512)

        fx = cx = self.full_res_shape[0] / 2.0
        fy = cy = self.full_res_shape[1] / 2.0
        self.K = np.array([
            [fx, 0, cx,0],
            [0, fy, cy,0],
            [0, 0, 1,0],
            [0, 0, 0,1]
        ], dtype=np.float32)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
        self.loader = self.pil_loader

    def pil_loader(self,path):
        # open path as file to avoid ResourceWarning
        # (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            with pil.open(f) as img:
                img = img.convert("RGB")
                img = img.resize((self.width, self.height))
                return img

    # def check_depth(self):
    #     line = self.filenames[0].split()
    #     folder = line[0]
    #     frame_index = int(line[1])*4

    #     f_str = str(frame_index * 4).zfill(6) + "." + "PNG"
    #     folder = folder.replace("sensor", "stereo_disparity")

    #     depth_path = os.path.join(self.data_path, folder, f_str)

    #     return os.path.isfile(depth_path)

    def get_image_path(self, folder, frame_index, side):
        f_str = str(frame_index).zfill(6) + "." + "JPEG"
        if side == "l":
            folder = folder.replace("sensor", "color_left")
        else:
            folder = folder.replace("sensor", "color_right")
        image_path = os.path.join(self.data_path, folder, f_str)
        return image_path

    # def get_color(self, folder, frame_index, side, do_flip):
    #     color = self.loader(self.get_image_path(folder, frame_index*4, side))

    #     if do_flip:
    #         color = color.transpose(pil.FLIP_LEFT_RIGHT)

    #     return color
    def get_color(self,folder):
        loader = self.pil_loader
        color = loader(os.path.join(self.data_path,folder))       
        # color = self.to_tensor(color)
        return color

    def get_disp(self,folder,frame_index):
        disp_path = os.path.join(
            self.data_path,
            folder[frame_index])

        disp_gt = pil.open(disp_path)
        disp_gt = disp_gt.resize((self.width, self.height))
        img = np.asarray(disp_gt, np.uint16)
        img.dtype = np.float16
        disp_gt = self.to_tensor(img)
        return disp_gt

import sys
# sys.path.append('monodepth2')
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
from options_midair import MonodepthOptions
# sys.path.append('../')
import torch
from utils import readlines


# opt = MonodepthOptions().parse()
# # splits_dir = os.path.join(os.path.dirname(__file__), "splits")
# splits_dir = "splits"

# # encoder_path = "monodepth2/pretrained_model/mono_1024x320/encoder.pth"
# # decoder_path = "monodepth2/pretrained_model/mono_1024x320/depth.pth"

# # encoder_dict = torch.load(encoder_path,map_location=torch.device('cpu'))

# filenames = readlines(os.path.join(splits_dir, "m4depth_midair", "train_files.txt"))
# a = MidAirDataset(opt.data_path, filenames[:20], select_file_names = True,
#                 height =512, width = 512,
#                 frame_idxs =[0,-1,1], num_scales =1, is_train=False, use_imu=True,use_ekf=True)

# print(a.__getitem__(0))
# # print(f"length of a is : {len(a)}")
# # print(a.__getitem__(548))
# # a.check_corectness(549)

