import cv2
import os
from get_dataset import make_datset
from option_config import setup_cfg
from option_config import get_parser

def save_images_in_folders(affine_trans, train_folder='train_data', ground_truth_folder='ground_truth', start_index=0):
    # Create folders if they don't exist
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(ground_truth_folder, exist_ok=True)
    
    for i, key in enumerate(affine_trans.keys(), start=start_index):
        print(i)
        original_image = affine_trans[key]["original_target"]
        aligned_image = affine_trans[key]["aligned_source"]
        
        # Generate the numerical filename (padded with zeros)
        filename = f"{i:07d}.png"  # e.g., 0000000.png
        
        # Save the original image in the ground_truth folder
        original_image_path = os.path.join(ground_truth_folder, filename)
        cv2.imwrite(original_image_path, original_image)
        
        # Save the aligned image in the train_data folder
        aligned_image_path = os.path.join(train_folder, filename)
        cv2.imwrite(aligned_image_path, aligned_image)

# Inputs for the configuration
print("Starting the dataset creation process...")
inputs = ['--config-file', "/Users/ibrahimhassan/Documents/Documents/Frame_prediction/CutLER/cutler/model_zoo/configs/CutLER-ImageNet/cascade_mask_rcnn_R_50_FPN_demo.yaml", 
          '--input', "imgs/demo1.jpg", '--confidence-threshold', '0.5', 
          "--opts", "MODEL.WEIGHTS", "/Users/ibrahimhassan/Documents/Documents/Frame_prediction/CutLER/cutler/demo/cutler_cascade_final.pth", 
          "MODEL.DEVICE", "cpu"]

args = get_parser(inputs)
cfg = setup_cfg(args)

# Dataset creation and transformation retrieval
print("Make dataset")
md = make_datset(args, cfg, starting_file_nr=7629, nr_files=5000)
print("Get perspective transform")
perspec_trans, warp_perspec_matrices = md.get_perspective_transform()

# Start saving images with numerical filenames
start_index = 0  # Set this to the starting index you want for numbering

for i in range(len(perspec_trans)):
    print(f"Batch {i}/{len(perspec_trans)}")
    affine_trans, warp_affine_matrices = md.get_affine_transform(perspec_trans)
    save_images_in_folders(affine_trans, train_folder='train_data', ground_truth_folder='ground_truth', start_index=start_index)

    # Increment the start index for the next batch of files (in case of multiple iterations)
    start_index += len(affine_trans)

