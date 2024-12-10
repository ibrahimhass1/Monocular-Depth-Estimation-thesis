from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader

from layers import disp_to_depth
from utils import readlines
from options_midair import MonodepthOptions
import datasets
import datasets.midair

import networks
from PIL import Image

import matplotlib.pyplot as plt

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)


splits_dir = os.path.join(os.path.dirname(__file__), "splits")

# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR =10

def custom_collate(batch):
    
    # remove empty items
    batch = [item for item in batch if item !={}]
    return torch.utils.data.default_collate(batch)

def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    
    if opt.ext_disp_to_eval is None:

        # opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

        # assert os.path.isdir(opt.load_weights_folder), \
        #     "Cannot find a folder at {}".format(opt.load_weights_folder)

        # print("-> Loading weights from {}".format(opt.load_weights_folder))
        print(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))

        # filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
        filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
        print(filenames)
        # opt.load_weights_folder =  "/Users/ibrahimhassan/Documents/Documents/thesis_DynaDepth/Trained_models/dynadepth_epoch_85_sep_3/models/weights_84"
        # opt.load_weights_folder = "/Users/ibrahimhassan/Documents/Documents/thesis_DynaDepth/Trained_models/dynadepth_epoch_16_sep_6/models/weights_16"
        # opt.load_weights_folder = "/Users/ibrahimhassan/Documents/Documents/thesis_DynaDepth/Trained_models/dynadepth_epoch_30_sep_10_nr_traj_14/models/weights_17"
        # opt.load_weights_folder = "/Users/ibrahimhassan/Documents/Documents/thesis_DynaDepth/Trained_models/dynadepth_midair_epoch_30_sep_24/models/weights_14"
        # opt.load_weights_folder = "/Users/ibrahimhassan/Documents/Documents/thesis_DynaDepth/Trained_models/dynadepth_midair_epoch_50_sep_29/models/weights_49"

        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")


        encoder_dict = torch.load(encoder_path,map_location=torch.device('cpu'))
        
        dataset = datasets.midair.MidAirDataset(opt.data_path, filenames,
                                                 encoder_dict['height'], encoder_dict['width'],
                                                 [0],num_scales =1,use_imu = True, use_ekf=False,is_train=False,select_file_names = True )
       
        dataloader = DataLoader(dataset, 16, shuffle=False, num_workers=opt.num_workers,
                                pin_memory=True, drop_last=False, collate_fn= custom_collate)

        encoder = networks.ResnetEncoder(opt.num_layers, False)
        depth_decoder = networks.DepthDecoder(encoder.num_ch_enc,scales=range(1))

        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict(torch.load(decoder_path,map_location=torch.device('cpu')))

        # encoder.cuda()
        encoder.eval()
        # depth_decoder.cuda()
        depth_decoder.eval()

        pred_disps = []

        #ADDED
        input_color_list = []

        print("-> Computing predictions with size {}x{}".format(
            encoder_dict['width'], encoder_dict['height']))
        depth_gt = []
        with torch.no_grad():
            for idx, data in enumerate(dataloader):
                print("Processing {}/{}".format(idx, len(dataloader)))
                input_color = data[("color", 0, 0)]
                disp_gt_batch = data[("disp_gt")]
                disp_gt_batch = (opt.height // 2) / disp_gt_batch.to(torch.float32)
                disp_gt_batch = np.clip(disp_gt_batch, 1., 100.)
                depth_gt.append(disp_gt_batch[:,0].detach())


                if opt.post_process:
                    # Post-processed results require each image to have two forward passes
                    input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

                output = depth_decoder(encoder(input_color))

                pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
                pred_disp = pred_disp.cpu()[:, 0]

                # fig, (ax1,ax2,ax3) = plt.subplots(1,3)
                if opt.post_process:
                    N = pred_disp.shape[0] // 2
                    pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

                pred_disps.append(pred_disp)
                input_color_list.append(input_color)
                # ax1.imshow(pred_disp[0])
                # ax1.set_title("Predicted disparity")

                # print(data["depth_gt"].shape)
                # ax2.imshow(data["depth_gt"][0][0])
                # # ax2.imshow(data["depth_gt"][0].permute(2, 1, 0).numpy())
                # # ax2.set_title("Ground truth depth")
                # ax3.imshow(data["color", 0, 0][0].permute(1,2,0))

                # plt.show()
                if idx == 300:
                    break

        pred_disps = torch.cat(pred_disps)
        pred_disps = pred_disps.numpy()
        depth_gt = torch.cat(depth_gt)
        depth_gt = depth_gt.numpy()
        input_color_list = torch.cat(input_color_list)
        input_color_list = input_color_list.numpy()
    else:
        # Load predictions from file
        print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
        pred_disps = np.load(opt.ext_disp_to_eval)

        if opt.eval_eigen_to_benchmark:
            eigen_to_benchmark_ids = np.load(
                os.path.join(splits_dir, "benchmark", "eigen_to_benchmark_ids.npy"))

            pred_disps = pred_disps[eigen_to_benchmark_ids]

    if opt.save_pred_disps:
        output_path = os.path.join(
            opt.load_weights_folder, "disps_{}_split.npy".format(opt.eval_split))
        print("-> Saving predicted disparities to ", output_path)
        np.save(output_path, pred_disps)

    if opt.no_eval:
        print("-> Evaluation disabled. Done.")
        quit()

    elif opt.eval_split == 'benchmark':
        save_dir = os.path.join(opt.load_weights_folder, "benchmark_predictions")
        print("-> Saving out benchmark predictions to {}".format(save_dir))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for idx in range(len(pred_disps)):
            disp_resized = cv2.resize(pred_disps[idx], (1216, 352))
            depth = STEREO_SCALE_FACTOR / disp_resized
            depth = np.clip(depth, 0, 80)
            depth = np.uint16(depth * 256)
            save_path = os.path.join(save_dir, "{:010d}.png".format(idx))
            cv2.imwrite(save_path, depth)

        print("-> No ground truth is available for the KITTI benchmark, so not evaluating. Done.")
        quit()

    # gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
    # gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1')["data"]

    gt_depths = depth_gt

    print("-> Evaluating")

    
    print("   Mono evaluation - using median scaling")

    errors = []
    ratios = []

    for i in range(pred_disps.shape[0]):
        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp

        #Added
        input_color = input_color_list[i]

        # def save_maps(gt, est, id):
        def save_maps(gt, est,input_color, id):

            def save_depth(map, path):
                map = np.clip(map, np.exp(0.), 80.)
                I8_content = (np.log(map) * 255.0 / np.log(80.)).astype(np.uint8)
                img_tmp = Image.fromarray(np.stack((I8_content, I8_content, I8_content), axis=2))
                # img_tmp.save(path)
                plt.imsave(path, I8_content,cmap='viridis_r')  # Save directly to the path using a grayscale colormap 

            save_dir = os.path.join(opt.log_dir, "saved_maps")
            os.makedirs(save_dir, exist_ok=True)
            filename_gt = os.path.join(save_dir, str(id).zfill(6) + "_gt.png")
            save_depth(gt, filename_gt)
            filename_est = os.path.join(save_dir, str(id).zfill(6) + "_est.png")
            save_depth(est, filename_est)
            filename_input = os.path.join(save_dir, str(id).zfill(6) + "_input.png")
            plt.imsave(filename_input, input_color.transpose(1, 2, 0))


        if opt.eval_split == "eigen":
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

        else:
            mask = gt_depth > 0
            
        gt_depth_save = gt_depth
        pred_depth_save = pred_depth

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]
        
        gt_depth[gt_depth < MIN_DEPTH] = MIN_DEPTH
        gt_depth[gt_depth > MAX_DEPTH] = MAX_DEPTH

        pred_depth *= opt.pred_depth_scale_factor
        ratio = 1
        if not opt.disable_median_scaling:
            ratio = np.median(gt_depth) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH
        
        if opt.export_pics:
            save_maps(gt_depth_save, pred_depth_save*ratio,input_color, i)

        errors.append(compute_errors(gt_depth, pred_depth))

        # fig, (ax1,ax2) = plt.subplots(1,2)
        
        # ax1.imshow(gt_depth_save, cmap='viridis_r')
        # ax1.set_title("Ground truth depth")
        # ax2.imshow(pred_depth_save * ratio, cmap='viridis_r')
        # ax2.set_title("Predicted depth")
        # plt.show()


    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    mean_errors = np.array(errors).mean(0)

    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")


if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
