import os
import cv2
import random
from helperfunctions import compute_iou_matrix, maximum_overlap_association, estimate_transform_ecc, extract_roi
import time
import torch
import numpy as np

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.structures import Instances
import sys
sys.path.append('CutLER/cutler/demo')
sys.path.append('CutLER/cutler/')
sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('../'))
from predictor import VisualizationDemo
from detectron2.utils.visualizer import ColorMode, Visualizer


class make_datset():

    def __init__(self,options,cfg,nr_files=5,starting_file_nr=0):

        self.opt = options
        self.data_path = self.opt.data_path
        self.nr_files = nr_files
        self.starting_file_nr = starting_file_nr
        self.demo = VisualizationDemo((cfg))
        self.result_dict, self.output_frames = self.segment_frames()
        self.association_dict = self.get_associations()
        self.tracklets = self.get_tracklets()


    def readlines(self,filename):
        """Read all the lines in a text file and return as a list
        """
        with open(filename, 'r') as f:
            lines = f.read().splitlines()
        return lines
    
    def get_files(self):
    
        def extract_sort_keys(item):
            prefix, rest = item.split('_sync ')
            sync_number, side = rest.split(' ')
            return (prefix, int(sync_number), side)

        fpath = os.path.join("splits", "eigen_zhou", "{}_files.txt")
        train_filenames = self.readlines(fpath.format("train"))
        filtered_data = [entry for entry in train_filenames if entry.endswith(' l')]
        train_filenames = sorted(filtered_data, key=extract_sort_keys)

        return train_filenames[self.starting_file_nr:self.starting_file_nr+self.nr_files]
    
    def segment_frames(self):

        def generate_random_color():
            return (random.random(), random.random(), random.random())
    
        output_frames = []
        result_dict = {}

        trajectory_files = self.get_files()

        for idx, frame in enumerate(trajectory_files):

            origin = "0000000000"  # origin as a string
            seq, sub_number = frame.split(" ")[:2]
            number = str(int(origin) + int(sub_number)).zfill(len(origin))  # Keep the same number of digits as origin
            img = read_image(os.path.join(self.data_path, seq, "image_02/data", number + ".jpg"), format="BGR")
            
            start_time = time.time()
            predictions, _ = self.demo.run_on_image(img)  # We don't use the original visualized output
           
            setup_logger(name="fvcore")
            logger = setup_logger()
            # logger.info("Arguments: " + str(self.opt))
            logger.info(
                "{}: {} in {:.2f}s".format(
                    self.opt.input,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )
            
            instances = predictions["instances"]
            
            # Filter out masks that cover more than 30% of the image
            total_pixels = img.shape[0] * img.shape[1]
            masks = instances.pred_masks
            valid_indices = torch.sum(masks, dim=(1,2)) <= 0.20 * total_pixels
            
            # Create new Instances object with filtered data
            filtered_instances = Instances(instances.image_size)
            for field in instances.get_fields():
                filtered_instances.set(field, instances.get(field)[valid_indices])
            
            # Update predictions with filtered instances
            predictions["instances"] = filtered_instances
            
            # Create a new visualizer with only valid masks
            v = Visualizer(img, metadata=self.demo.metadata, scale=1.2)
            masks = filtered_instances.pred_masks.cpu().numpy()
            
            # Generate a unique color for each instance
            num_instances = len(masks)
            colors = [generate_random_color() for _ in range(num_instances)]
            
            for mask, color in zip(masks, colors):
                v.draw_binary_mask(
                    mask,
                    color=color,
                    edge_color=np.zeros(3),
                    alpha=0.8,
                )
            
            output_frame = v.output.get_image()[:, :, ::-1]  # Convert BGR to RGB
            output_frames.append(output_frame)
            
            result_dict[f"frame_{idx}"] = {
                "bounding_boxes": filtered_instances.pred_boxes.tensor.cpu().numpy(),
                "masks": filtered_instances.pred_masks.cpu().numpy(),
                "scores": filtered_instances.scores.cpu().numpy(),
                "original_image": img,
                "mask_image": output_frame
            }
        
        return result_dict, output_frames 
    
    def get_associations(self):

        len_result_dict = len(self.result_dict)
        association_dict = {}

        for i in range(len_result_dict - 1):
            # Extract bounding boxes of instances in frame k and frame k+1
            instances_k = self.result_dict[f"frame_{i}"]["masks"]
            instances_k1 = self.result_dict[f"frame_{i + 1}"]["masks"]

            # Compute the IoU matrix between instances in frame k and frame k+1
            IOU_k = compute_iou_matrix(instances_k, instances_k1)

            if IOU_k.size == 0:
                continue  # Skip this iteration if the IoU matrix is empty

            # Perform maximum overlap association between instances
            associations = maximum_overlap_association(IOU_k)
            mask_frame_0 =  [x[0] for x in associations]
            mask_frame_1 = [x[1] for x in associations]

            association_dict[f"frame_{i}-{i + 1}"] = {"association": associations,f"mask_frame_{i}":instances_k[mask_frame_0],f"mask_frame_{i+1}":instances_k1[mask_frame_1],f"image_{i}": self.result_dict[f"frame_{i}"]["original_image"], f"image_{i+1}": self.result_dict[f"frame_{i + 1}"]["original_image"]}

        return association_dict

    def get_tracklets(self):

        tracklets = {}
        next_tracklet_id = 0

        # Initialize the tracklets with instances from the first frame
        for instance_idx in range(len(self.result_dict["frame_0"]["masks"])):
            tracklets[next_tracklet_id] = {"instances":[(0, instance_idx)], "frames": {}, "masks": {}}
            next_tracklet_id += 1

        # Now, build tracklets by linking associations
        len_result_dict = len(self.result_dict)

        for i in range(len_result_dict - 1):
            try:
                associations = self.association_dict[f"frame_{i}-{i + 1}"]['association']
            except KeyError:
                continue
            ############
            # Here select the association masks, because in the tracklet now all masks all forwarded for every tracklet instance, this is not what we want
            # We want to select the mask that was associated with the instance
            ############
            
            frames = [self.association_dict[f"frame_{i}-{i + 1}"][f"image_{i}"],self.association_dict[f"frame_{i}-{i + 1}"][f"image_{i+1}"]]
            masks = [self.association_dict[f"frame_{i}-{i + 1}"][f"mask_frame_{i}"],self.association_dict[f"frame_{i}-{i + 1}"][f"mask_frame_{i+1}"]]

            # Create a mapping from instances in frame i to their tracklet IDs
            instance_to_tracklet = {}
            for tracklet_id, instances in tracklets.items():
                last_frame, last_instance_idx = instances["instances"][-1]
                if last_frame == i:
                    instance_to_tracklet[last_instance_idx] = tracklet_id

            # Update tracklets based on associations
            for index, (i_max, j_max) in enumerate(associations):
                if i_max in instance_to_tracklet:
                    tracklet_id = instance_to_tracklet[i_max]
                    tracklets[tracklet_id]["instances"].append((i + 1, j_max))
                    tracklets[tracklet_id]["frames"][f"frame_{i}"] = frames[0]
                    tracklets[tracklet_id]["frames"][f"frame_{i + 1}"] = frames[1]
                    tracklets[tracklet_id]["masks"][f"mask_frame_{i}"] = masks[0][index]
                    tracklets[tracklet_id]["masks"][f"mask_frame_{i + 1}"] = masks[1][index]
                else:
                    # If the instance is not in any tracklet, start a new one
                    tracklets[next_tracklet_id] = {"instances": [(i + 1, j_max)], "frames": {}, "masks": {}}
                    tracklets[next_tracklet_id]["frames"][f"frame_{i}"] = frames[0]
                    tracklets[next_tracklet_id]["frames"][f"frame_{i + 1}"] = frames[1]
                    tracklets[next_tracklet_id]["masks"][f"mask_frame_{i}"] = masks[0][index]
                    tracklets[next_tracklet_id]["masks"][f"mask_frame_{i + 1}"] = masks[1][index]
                    next_tracklet_id += 1

        return tracklets

    def get_perspective_transform(self):

        warp_matrices = {}  # Initialize a dictionary to store warp matrices
        aligned_sources_dict = {}  # Initialize a dictionary to store aligned sources and original targets

        for tracklet_id, data in self.tracklets.items():
            print(f"key: {tracklet_id}/{len(self.tracklets)}")
            instances = data['instances']
            # if tracklet_id == 2:
            #     break
            
            for idx in range(len(instances) - 1):
                print(f"idx: {idx}/{len(instances) - 1}")
                frame_number_i, _ = instances[idx]
                frame_number_j, _ = instances[idx + 1]
                
                # Extract ROIs for the object in both frames
                roi_i = extract_roi(frame_number_i, None, data)  # Source frame
                roi_j = extract_roi(frame_number_j, None, data)  # Target frame
                
                # Estimate the transformation using ECC
                warp_matrix = estimate_transform_ecc(roi_i, roi_j, warp_mode=cv2.MOTION_HOMOGRAPHY)
                
                if warp_matrix is not None:
                    # Store warp matrix using tracklet_id and index
                    warp_matrices[(tracklet_id, idx)] = warp_matrix
                    
                    # Apply the warp matrix to align roi_i to roi_j
                    h_j, w_j = roi_j.shape[:2]  # Get height and width of target frame
                    aligned_source_frame = cv2.warpPerspective(roi_i, warp_matrix, (w_j, h_j), flags=cv2.INTER_LINEAR)
                    
                    # Store both the aligned source frame and the original target frame
                    aligned_sources_dict[(tracklet_id, idx)] = {
                        'aligned_source': aligned_source_frame,
                        'original_target': roi_j
                    }

        return aligned_sources_dict, warp_matrices
    
    def get_affine_transform(self,aligned_sources_dict):

        warp_matrices_affine_transformatiom = {}  # Initialize a dictionary to store warp matrices
        aligned_sources_after_affine_transformation = {}  # Initialize a dictionary to store aligned sources
        for tracklet_id, instance in aligned_sources_dict.items():
            print(f"key: {tracklet_id}")
            # if tracklet_id == 2:
            #     break

            # Estimate the transformation using ECC with affine transform
            warp_matrix = estimate_transform_ecc(instance["aligned_source"], instance["original_target"], warp_mode=cv2.MOTION_AFFINE)
            # Save the warp matrix in the dictionary
            if warp_matrix is not None:
                # Store warp matrix using tracklet_id and index
                warp_matrices_affine_transformatiom[tracklet_id] = warp_matrix

                # Apply the warp matrix to align instance["aligned_source"] to instance["original_target"]
                h_j, w_j = instance["original_target"].shape[:2]  # Get height and width of target frame
                aligned_perspective_frame = cv2.warpAffine(instance["aligned_source"], warp_matrix, (w_j, h_j), flags=cv2.INTER_LINEAR)
                
                # Store both the aligned source frame and the original target frame
                aligned_sources_after_affine_transformation[(tracklet_id)] = {
                    'aligned_source': aligned_perspective_frame,
                    'original_target': instance["original_target"]
                }

        return aligned_sources_after_affine_transformation, warp_matrices_affine_transformatiom
            # Now you have warp_matrices stored by tracklet_id and index


