# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import os
import random
import numpy as np
import copy
from PIL import Image  # using pillow-simd for increased speed
from tqdm import tqdm
import h5py

import torch
import torch.utils.data as data
from torchvision import transforms
from liegroups.numpy import SO3 as SO3_np
import torchvision.transforms.functional as F
from utils import *

from torchvision.transforms import InterpolationMode

import re



def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

        
def quat_mul(q0, q1):
    """Multiply two quaternions in np.array
    -> Input: numpy arrays
    -> Output: liegroups.numpy.SO3 object
    """
    w0, x0, y0, z0 = q0[0], q0[1], q0[2], q0[3]
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    
    q0q1_w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
    q0q1_x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1 
    q0q1_y = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1 
    q0q1_z = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1

    q0q1 = np.array([q0q1_w, q0q1_x, q0q1_y, q0q1_z])
    return SO3_np.from_quaternion(q0q1)

def custom_sort_key(item):
    filepath = item[0]  # The key of the dictionary item is the filepath
    
    # Extract components from the filepath
    parts = filepath.split('/')
    weather = parts[1]
    sensor = parts[2]
    trajectory = parts[3]
    range_str = parts[4]

    # Extract the last two digits of the trajectory number
    trajectory_last_two = int(trajectory[-2:])

    # Extract the range numbers
    range_numbers = list(map(int, range_str.split('-')))

    return (trajectory_last_two, sensor, weather, *range_numbers)

class MonoDataset(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_path
        filenames
        height
        width
        frame_idxs
        num_scales
        is_train
        img_ext
    """
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 select_file_names= False,
                 use_imu=False,
                 use_ekf=False,
                 k_imu_clip=5,
                 is_train=False,
                 img_ext='.jpg',
                 img_noise_type="clean",
                 img_noise_brightness=0.5,
                 img_noise_contrast=0.5,
                 img_noise_saturation=0.5,
                 img_noise_hue=0.5,
                 img_noise_gamma=0.5,
                 img_noise_gain=0.5,
                 img_mask_num=1,
                 img_mask_size=50,
                 avoid_quat_check=False):
        super(MonoDataset, self).__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.select_file_names = select_file_names
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = InterpolationMode.BILINEAR


        self.frame_idxs = frame_idxs

        self.is_train = is_train
        self.img_ext = img_ext
        self.frame_idxs = frame_idxs
        self.use_imu = use_imu
        self.use_ekf = use_ekf 
        self.k_imu_clip = k_imu_clip # previous 12 x k IMU records will be fed into CNN
        self.img_noise_type = img_noise_type
        self.avoid_quat_check = avoid_quat_check

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1
            
        
        # "bcsh": brightness/contrast/saturation/hue noises, "gg": gamma/gain noises
        self.img_noiser = None
        if self.img_noise_type == "bcsh":
            self.img_noiser = transforms.ColorJitter(
                brightness = img_noise_brightness,
                contrast = img_noise_contrast,
                saturation = img_noise_saturation,
                hue = img_noise_hue 
            )
        
        if self.img_noise_type == "gg":
            self.gamma_range = img_noise_gamma # e.g. 0.5
            self.gain_range = img_noise_gain   # e.g. 0.5
        
        if self.img_noise_type == "mask":
            self.mask_num = img_mask_num
            self.mask_size = img_mask_size
            self.mask = Image.new("RGB", (self.mask_size, self.mask_size), (0, 0, 0))

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)
            
            self.resize[i] = transforms.Resize((self.height // s, self.width // s), interpolation=self.interp)


        
        # if self.use_imu:
        self.seqs, self.scenes = self.parse_seqs()
        self.preint_imus, self.raw_imus = self.preintegrate_imu()

        print("==============================")
        print("=> Num of filenames before filtering based on IMU nan: {}".format(len(self.filenames)))
        print("=> Num of filenames After filtering based on IMU nan: {}".format(len(self.filenames)))


    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            if "color" in k:
                n, im, i = k
                if self.img_noise_type == "bcsh":
                    inputs[(n, im, -1)] = self.img_noiser(inputs[(n, im, -1)])
                
                if self.img_noise_type == "gg":
                    aug_gamma = random.uniform(1 - self.gamma_range, 1 + self.gamma_range)
                    aug_gain = random.uniform(1 - self.gain_range, 1 + self.gain_range)
                    inputs[(n, im, -1)] = transforms.functional.adjust_gamma(inputs[(n, im, -1)], gamma=aug_gamma, gain=aug_gain)
                
                if self.img_noise_type == "mask":
                    w, h = inputs[(n, im, -1)].size
                    for _ in range(self.mask_num):
                        wshift = random.randint(0, w - self.mask_size)
                        hshift = random.randint(0, h - self.mask_size)
                        inputs[(n, im, -1)].paste(self.mask, (wshift, hshift))
                    
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])                    


        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}
        

        if self.select_file_names:
            rel_keys = []
            for i in self.filenames:
                base_dir, frame_index, _ = i.split(" ")
                rel_keys.append(f"{base_dir}/{frame_index}-{int(frame_index)+1}")
            key_dict = rel_keys
        else:
            key_dict = list(self.preint_imus.keys())


        item = self.preint_imus[key_dict[index]]
        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        img_folder = item["img_filenames"]
        disp_folder = item["disp_img_filenames"]
        
        
        for i in self.frame_idxs:
            inputs[("color",i,-1)] = self.get_color(img_folder[i-2])

       
        inv_K = np.linalg.inv(self.K)
        inputs[("K", 0)] = torch.from_numpy(self.K)
        inputs[("inv_K", 0)] = torch.from_numpy(inv_K)
        

        if do_color_aug:

            order, brightness_factor, contrast_factor, saturation_factor, hue_factor = transforms.ColorJitter.get_params(self.brightness, self.contrast, self.saturation, self.hue)
    
            # Define a function that applies the transformations in the specified order
            def color_aug(img):
                # Create a list of transformations and their corresponding functions
                transformations = [
                    (0, lambda x: F.adjust_brightness(x, brightness_factor)),
                    (1, lambda x: F.adjust_contrast(x, contrast_factor)),
                    (2, lambda x: F.adjust_saturation(x, saturation_factor)),
                    (3, lambda x: F.adjust_hue(x, hue_factor)),
                ]
                
                # Apply the transformations in the order specified by `order`
                for idx in order.tolist():  # Convert the tensor to a list for iteration
                    img = transformations[idx][1](img)  # Apply the corresponding transformation
                    
                return img

        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

        inputs["disp_gt"] = self.get_disp(disp_folder, 0)
        # print(item.keys())
        

        if self.use_imu:
            key = list(self.preint_imus.keys())[index]
            # Add IMU preintegrations
                                                    
            try:
                # key above is stored as -1 -> 0, frame_index corresponds to frame -1
                inputs[("preint_imu", -1, 0)] = copy.deepcopy(self.preint_imus[key_dict[index-1]])
                inputs[("preint_imu", 0, 1)] = copy.deepcopy(self.preint_imus[key_dict[index+1]])
            except (KeyError, IndexError):
               
                if self.select_file_names:
                    if index+1 < len(self.filenames)-1:
                        print("oi")
                        return self.__getitem__(index+1)
                    else:
                        return {}

                else:
                    if index+1 < len(self.preint_imus)-1:
                        print("oi")
                        return self.__getitem__(index+1)
                    else:
                        return {}

        
            # Remove keys
            del inputs[("preint_imu", -1, 0)]["img_filenames"]
            del inputs[("preint_imu", -1, 0)]["disp_img_filenames"]
            del inputs[("preint_imu", 0, 1)]["img_filenames"]
            del inputs[("preint_imu", 0, 1)]["disp_img_filenames"]

            for imu_pair in [("preint_imu", -1, 0), ("preint_imu", 0, 1)]:
                for key_imu in inputs[imu_pair].keys():
                    inputs[imu_pair][key_imu] = torch.from_numpy(np.array(inputs[imu_pair][key_imu]))
        
        if self.use_ekf:
            
            def get_key(offset):
                    tmp_key = key_dict[index+offset]
                    split_key = tmp_key.split("/")
                    tmp_key_season, tmp_key_trajectory = split_key[1], split_key[3]
                    actual_season, actual_trajectory = key_dict[index].split("/")[1], key_dict[index].split("/")[3]
                    if tmp_key_season != actual_season or tmp_key_trajectory != actual_trajectory:
                        return "nan"
                    return key_dict[index+offset]

            imu_list = []
            for i in range(self.k_imu_clip):   
                tmp_key = get_key(i - self.k_imu_clip + 1)
                
                if tmp_key in self.raw_imus.keys():
                    tmp_imu = copy.deepcopy(self.raw_imus[tmp_key])
                    if tmp_imu in ["nan", "none"]:
                        imu_list.append(np.zeros((4, 6)))
                    else:
                        wa_xyz_ = tmp_imu["wa_xyz"] # [4, 6] 
                        if wa_xyz_.shape[0] == 3:
                            wa_xyz_ = np.concatenate([wa_xyz_, np.expand_dims(wa_xyz_[-1, :], 0)], 0)
                        imu_list.append(wa_xyz_)
                else:
                    imu_list.append(np.zeros((4, 6)))

            # deal with the overlapped imu records at current end and next beginning
            imus = np.concatenate(imu_list, axis=0)
            assert imus.shape[0] == self.k_imu_clip * 4

            inputs["imus"] = torch.from_numpy(imus)


        return inputs


    def proc_imu(self,imu,velo_init):
        
        imu = np.array(imu, dtype=float)
        
        # calculate the preintegration terms
        a_xyz, w_xyz = imu[:,:3], imu[:,3:]
        v_flu = velo_init
        
        wa_xyz =imu
        
        delta_t, alpha, beta = 0, 0, 0
        q = np.identity(3)
        linearized_ba, linearized_bg = 0, 0
        
        # R_b: list of all preintegrated rotations R_bk_bt, t=t_k, t_k + 1, ..., t_k+1
        # dts: list of all delta_t, length is len(R_b) - 1
        R_b = []
        dts = []
        
        # R_b.append(q.mat)
        R_b.append(q)
        
        for idx in range(imu.shape[0]):
            if idx == 0: continue  
            dt = 1/100 # unit: s

            acc_0 = q @ (a_xyz[idx-1] - linearized_ba)
            gyr = 0.5 * (w_xyz[idx-1] + w_xyz[idx]) - linearized_bg 
            
            # NOTE: There are two ways to compute delta_q
            q = q @ SO3_np.exp(dt * gyr).mat
            
            acc_1 = q @ (a_xyz[idx] - linearized_ba)
            acc = 0.5 * (acc_0 + acc_1)
            alpha += acc * dt * dt 
            beta += acc * dt
            delta_t += dt
            
            dts.append(dt) # unit: s

            # R_b.append(q.mat)
            R_b.append(q)

        # NOTE: We should only take the v_norm at the beginning, not accumulated!
        v_norm = np.linalg.norm(v_flu[0])
        
        return wa_xyz, delta_t, alpha, beta, R_b, v_norm, dts
    
    def preintegrate_imu(self):
        preint_imus, raw_imus = dict(), dict()
        print("Preintegrating IMU data (Save raw wa_xyz also)...")

        seqs, scenes = self.parse_seqs()

        for seq in tqdm(seqs):

            weather_condition = seq.split("/")[1]
            scene = seq.split('/sensor')[0] # scene: e.g. "cloudy"
            filename = os.path.join(self.data_path,scene,"sensor_records.hdf5")
            current_scene_path = os.path.join(self.data_path,scene)
            print(f"current_scene_path: {current_scene_path}")
            imu_acc, imu_gyr = [], []
                
            trajectories = {}              
            with h5py.File(filename, 'r') as f:
                # Extract the relative path to use as keys in the dictionary
                
                a = fill_structure(f)
                trajectories.update(fill_structure(f))

            reduce_nr_of_trajectories(trajectories,29)

            for traj_key in trajectories.keys():
                trajectory = trajectories[traj_key]
                acc = trajectory['imu']['accelerometer']
                gyr = trajectory['imu']['gyroscope']
                velo_init = trajectory["gps"]["velocity"][0]
                imu = np.concatenate((acc,gyr),1)
                
                group_batch_size = 4
                # if rows % batch_size !=0 then trim to the nearest multiple of 4
                if imu.shape[0] % group_batch_size != 0:
                    imu = imu[:imu.shape[0] - (imu.shape[0] % group_batch_size)]
                imu_batch = imu.reshape(-1, group_batch_size, imu.shape[1])


                for idx, imu_seq in tqdm(enumerate(imu_batch)):
                    
                    if idx +2 < len(imu_batch):

                        wa_xyz, delta_t, alpha, beta, R_b, v_norm, dts = self.proc_imu(imu_seq, velo_init)
                        

                        R_cb = np.array([
                            [0, 0, 1],
                            [0, 1, 0],
                            [1, 0, 0]
                        ])
                        t_cb = np.array([0, 0, 0])
                        R_bc = R_cb # transpose of anti-diagonal matrix is the same
                        t_bc = t_cb

                        alpha = R_cb @ alpha 
                        beta = R_cb @ beta 
                        R_c = R_cb @ R_b[-1] @ R_bc 
                        R_c_inv = np.linalg.inv(R_c)
                        R_cbt_bc = R_cb @ t_bc
                        key = "{}/{}/{}-{}".format(seq, f"sensor/{traj_key}", idx, idx+1)

                        preint_imus[key] = {
                                "delta_t": delta_t, 
                                "alpha": alpha, "beta": beta, 
                                "R_c": R_c, "R_c_inv": R_c_inv, 
                                "R_cbt_bc": R_cbt_bc, "v_norm": v_norm,
                                "R_cb": R_cb, "R_bc": R_bc, "t_bc": t_bc,
                                
                            }
                        raw_imus[key] = {
                            "wa_xyz": wa_xyz, 
                        }
                        
                        ## Used for the derivative of H in EKF update
                        # phi_c: the log (so3) of R_c 
                        # J_l_inv_neg_R_cb: J_l(-phi_c)^{-1} @ R_cb
                        # R_cb_p_bc_wedge_neg = -R_ckbk+1 @ p_bc_wedge
                        phi_c = SO3_np(R_c).log()
                        J_l_inv_neg_R_cb = SO3_np.inv_left_jacobian(-phi_c) @ R_cb
                        R_cb_p_bc_wedge_neg = - R_cb @ R_b[-1] @ SO3_np.wedge(t_bc)
                        preint_imus[key]["phi_c"] = phi_c
                        preint_imus[key]["J_l_inv_neg_R_cb"] = J_l_inv_neg_R_cb
                        preint_imus[key]["R_cb_p_bc_wedge_neg"] = R_cb_p_bc_wedge_neg
                        
                        
                        ## Used for F and G in EKF propagation
                        # R_ckbt: [], # list of R_cb @ R_{b_kb_t}} for all t from t_k to t_k+1
                        dts_full = np.array(dts)
                        wa_xyz_full = wa_xyz
                        R_ckbt_full = np.array([R_cb @ tmpR for tmpR in R_b])
                        
                        # For batch ops, the IMU length should be the same. But now 11 or 12
                        # Pad no-motion to the end to fix len at 12, but save the true length in "imu_len"
                        imu_len = R_ckbt_full.shape[0]
                    
                                            
                        preint_imus[key]["dts_full"] = np.array(dts_full)  # [3,]
                        preint_imus[key]["R_ckbt_full"] = np.array(R_ckbt_full) # [4, 3, 3]
                        preint_imus[key]["wa_xyz_full"] = np.array(wa_xyz_full) # [4, 6]
                        preint_imus[key]["imu_len"] = imu_len # 4

                        img_filenames = [os.path.join(seq,str(trajectory['camera_data']['color_left'][idx]).split("'")[1]),os.path.join(seq,str(trajectory['camera_data']['color_left'][idx+1]).split("'")[1]),os.path.join(seq,str(trajectory['camera_data']['color_left'][idx+2]).split("'")[1])]
                        preint_imus[key]["img_filenames"] = img_filenames
                        
                        # Add corresponding disp image filenames
                        # depth_filenames = [os.path.join(str(seq),str(trajectory['camera_data']['depth'][idx]).split("'")[1]),os.path.join(str(seq),str(trajectory['camera_data']['depth'][idx+1]).split("'")[1]),os.path.join(str(seq),str(trajectory['camera_data']['depth'][idx+2]).split("'")[1])]
                        # NOTE: Only return the disp ground truth of the target image
                        # NOTE: File ordering is [-1,0,1], target frame is 0
                        disp_filenames = [os.path.join(str(seq),str(trajectory['camera_data']['stereo_disparity'][idx+1]).split("'")[1])]
                        preint_imus[key]["disp_img_filenames"] = disp_filenames
            
        print("done with preintegrated")
        return dict(sorted(preint_imus.items(), key=custom_sort_key)), dict(sorted(raw_imus.items(),key=custom_sort_key))


    def parse_seqs(self):
        folders = set()
        scenes = set()
        filenames = ["Kite_training/cloudy", "Kite_training/sunny", "Kite_training/sunset", "Kite_training/foggy"]

        for line in filenames:

            folder = line.split()[0]
            folders.add(folder)
            scenes.add(folder.split('/')[0])

        return list(folders), list(scenes)

    def check_corectness(self,index):
        
        preint_imus, raw_imus = dict(), dict()
        seqs, scenes = self.parse_seqs()


        for seq in tqdm(seqs):

            weather_condition = seq.split("/")[1]
            if weather_condition == "foggy":
                continue
            print(f"seq: {seq}")
            scene = seq.split('/sensor')[0] # scene: e.g. "cloudy"
            print(f"scene: {scene}")
            filename = os.path.join(self.data_path,scene,"sensor_records.hdf5")
            current_scene_path = os.path.join(self.data_path,scene)
            print(f"current_scene_path: {current_scene_path}")
            imu_acc, imu_gyr = [], []
                
            trajectories = {}              
            with h5py.File(filename, 'r') as f:
                # Extract the relative path to use as keys in the dictionary
                
                a = fill_structure(f)
                trajectories.update(fill_structure(f))

            reduce_nr_of_trajectories(trajectories,29)
            # print(trajectories.keys())

            for traj_key in trajectories.keys():
                trajectory = trajectories[traj_key]
                acc = trajectory['imu']['accelerometer']
                gyr = trajectory['imu']['gyroscope']
                velo_init = trajectory["gps"]["velocity"][0]
                imu = np.concatenate((acc,gyr),1)
                
                group_batch_size = 4
                # if rows % batch_size !=0 then trim to the nearest multiple of 4
                if imu.shape[0] % group_batch_size != 0:
                    imu = imu[:imu.shape[0] - (imu.shape[0] % group_batch_size)]
                imu_batch = imu.reshape(-1, group_batch_size, imu.shape[1])
                idx = 0
                img_filenames = [os.path.join(seq,str(trajectory['camera_data']['color_left'][idx]).split("'")[1]),os.path.join(seq,str(trajectory['camera_data']['color_left'][idx+1]).split("'")[1]),os.path.join(seq,str(trajectory['camera_data']['color_left'][idx+2]).split("'")[1])]



                for idx, imu_seq in tqdm(enumerate(imu_batch)):
                    
                    if idx +2 < len(imu_batch):

                        wa_xyz, delta_t, alpha, beta, R_b, v_norm, dts = self.proc_imu(imu_seq, velo_init)
                        
                        #Converting these coordinates to the drone body frame b simply be obtained by the following transformation: (xb,yb,zb)=(zc,yc,xc)
                        R_cb = np.array([
                            [0, 0, 1],
                            [0, 1, 0],
                            [1, 0, 0]
                        ])
                        t_cb = np.array([0, 0, 0])
                        R_bc = R_cb # transpose of anti-diagonal matrix is the same
                        t_bc = t_cb

                        alpha = R_cb @ alpha 
                        beta = R_cb @ beta 
                        R_c = R_cb @ R_b[-1] @ R_bc 
                        R_c_inv = np.linalg.inv(R_c)
                        R_cbt_bc = R_cb @ t_bc
                        key = "{}/{}/{}-{}".format(seq, f"sensor/{traj_key}", idx, idx+1)

                        preint_imus[key] = {
                                "delta_t": delta_t, 
                                "alpha": alpha, "beta": beta, 
                                "R_c": R_c, "R_c_inv": R_c_inv, 
                                "R_cbt_bc": R_cbt_bc, "v_norm": v_norm,
                                "R_cb": R_cb, "R_bc": R_bc, "t_bc": t_bc,
                                
                            }
                        raw_imus[key] = {
                            "wa_xyz": wa_xyz, # [11, 6]
                        }
                        
                        ## Used for the derivative of H in EKF update
                        # phi_c: the log (so3) of R_c 
                        # J_l_inv_neg_R_cb: J_l(-phi_c)^{-1} @ R_cb
                        # R_cb_p_bc_wedge_neg = -R_ckbk+1 @ p_bc_wedge
                        phi_c = SO3_np(R_c).log()
                        J_l_inv_neg_R_cb = SO3_np.inv_left_jacobian(-phi_c) @ R_cb
                        R_cb_p_bc_wedge_neg = - R_cb @ R_b[-1] @ SO3_np.wedge(t_bc)
                        preint_imus[key]["phi_c"] = phi_c
                        preint_imus[key]["J_l_inv_neg_R_cb"] = J_l_inv_neg_R_cb
                        preint_imus[key]["R_cb_p_bc_wedge_neg"] = R_cb_p_bc_wedge_neg
                        
                        
                        ## Used for F and G in EKF propagation
                        # R_ckbt: [], # list of R_cb @ R_{b_kb_t}} for all t from t_k to t_k+1
                        dts_full = np.array(dts)
                        wa_xyz_full = wa_xyz
                        R_ckbt_full = np.array([R_cb @ tmpR for tmpR in R_b])
                        
                        imu_len = R_ckbt_full.shape[0]
                
                        
                        preint_imus[key]["dts_full"] = np.array(dts_full)  # [3,]
                        preint_imus[key]["R_ckbt_full"] = np.array(R_ckbt_full) # [4, 3, 3]
                        preint_imus[key]["wa_xyz_full"] = np.array(wa_xyz_full) # [4, 6]
                        preint_imus[key]["imu_len"] = imu_len # 4

                        
                        img_filenames = [os.path.join(seq,str(trajectory['camera_data']['color_left'][idx]).split("'")[1]),os.path.join(seq,str(trajectory['camera_data']['color_left'][idx+1]).split("'")[1]),os.path.join(seq,str(trajectory['camera_data']['color_left'][idx+2]).split("'")[1])]
                        preint_imus[key]["img_filenames"] = img_filenames
                        
                        # Add corresponding disp image filenames
                        # depth_filenames = [os.path.join(str(seq),str(trajectory['camera_data']['depth'][idx]).split("'")[1]),os.path.join(str(seq),str(trajectory['camera_data']['depth'][idx+1]).split("'")[1]),os.path.join(str(seq),str(trajectory['camera_data']['depth'][idx+2]).split("'")[1])]
                        # NOTE: Only return the disp ground truth of the target image
                        # NOTE: File ordering is [-1,0,1], target frame is 0
                        disp_filenames = [os.path.join(str(seq),str(trajectory['camera_data']['stereo_disparity'][idx+1]).split("'")[1])]
                        preint_imus[key]["disp_img_filenames"] = disp_filenames
      