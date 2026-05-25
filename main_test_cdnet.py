import os
import sys
import argparse
import logging
import numpy as np
import torch
import cv2
from collections import OrderedDict
from torch.utils.data import DataLoader

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from data.select_dataset import define_Dataset
from models.select_model import define_Model


def main(json_path='options/test_cdnet.json'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to test option JSON file.')
    parser.add_argument('--save_comparison', action='store_true', default=True, help='Save side-by-side comparison images')
    
    opt = option.parse(parser.parse_args().opt, is_train=False)
    
    # Setup directories
    save_dir = os.path.join('results', opt['task'])
    save_output_dir = os.path.join(save_dir, 'output')
    save_comparison_dir = os.path.join(save_dir, 'comparison')
    os.makedirs(save_output_dir, exist_ok=True)
    os.makedirs(save_comparison_dir, exist_ok=True)
    
    # Logger
    logger_name = 'test'
    utils_logger.logger_info(logger_name, os.path.join(save_dir, f'{logger_name}.log'))
    logger = logging.getLogger(logger_name)
    logger.info(option.dict2str(opt))
    
    # Convert to NoneDict
    opt = option.dict_to_nonedict(opt)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Model
    model = define_Model(opt)
    model.netG.eval()
    model.netG = model.netG.to(device)
    
    logger.info(f'Model loaded from: {opt["path"]["pretrained_netG"]}')
    logger.info(f'Testing on TRAINING SET: {opt["datasets"]["test"]["dataroot_lq"]}')
    
    # Load Test Dataset (actually training data)
    test_set = define_Dataset(opt['datasets']['test'])
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, 
                             num_workers=0, drop_last=False, pin_memory=True)
    
    logger.info(f'Number of test videos: {len(test_loader)}')
    
    # Testing metrics
    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    
    # Test each video
    for idx, test_data in enumerate(test_loader):
        key = test_data.get('key', [f'video_{idx}'])[0]
        
        # Extract video name and frame index from key (format: "PETS2006_0")
        if '_' in key:
            video_name, frame_number_str = key.rsplit('_', 1)
            original_frame_number = int(frame_number_str)
        else:
            video_name = key
            original_frame_number = 0
        
        # Setup output directory for this video
        video_output_dir = os.path.join(save_output_dir, video_name)
        os.makedirs(video_output_dir, exist_ok=True)
        
        # Check if frame already processed (for resume capability)
        expected_output_path = os.path.join(video_output_dir, f'{original_frame_number:04d}.png')
        if os.path.exists(expected_output_path):
            logger.info(f'\nSkipping [{idx+1}/{len(test_loader)}]: {video_name} frame {original_frame_number} (already processed)')
            continue
        
        logger.info(f'\nTesting [{idx+1}/{len(test_loader)}]: {video_name} frame {original_frame_number}')
        
        # Move data to device
        lq = test_data['L'].to(device)
        gt = test_data['H'].to(device) if 'H' in test_data else None
        
        # Inference
        with torch.no_grad():
            output = model.netG(lq)
        
        # Get center frame (the actual frame we're processing)
        center_frame_idx = output.shape[1] // 2
        
        # Only process and save the center frame
        output_frame = output[0, center_frame_idx, :, :, :].cpu()
        output_img = util.tensor2uint(output_frame)
        
        # Save output with original frame number
        save_path = os.path.join(video_output_dir, f'{original_frame_number:04d}.png')
        util.imsave(output_img, save_path)
        
        # Calculate metrics for center frame only (GT vs Model)
        if gt is not None:
            gt_frame = gt[0, center_frame_idx, :, :, :].cpu()
            gt_img = util.tensor2uint(gt_frame)
            
            # Get mask for this frame
            mask_frame = test_data['M'][0, center_frame_idx, 0, :, :].cpu().numpy()  # [H, W]
            mask_roi = (mask_frame >= 0.5).astype(np.uint8)  # ROI mask (1 for ROI, 0 for background)
            mask_bg = 1 - mask_roi  # Background mask
            
            # Overall metrics (full image) - GT vs Model
            psnr_full = util.calculate_psnr(output_img, gt_img, border=0)
            ssim_full = util.calculate_ssim(output_img, gt_img, border=0)
            
            # ROI-only metrics (moving objects) - GT vs Model
            if np.sum(mask_roi) > 0:  # If ROI exists
                output_roi = output_img * mask_roi[:, :, None]
                gt_roi = gt_img * mask_roi[:, :, None]
                
                psnr_roi = util.calculate_psnr(output_roi, gt_roi, border=0)
                ssim_roi = util.calculate_ssim(output_roi, gt_roi, border=0)
            else:
                psnr_roi = psnr_full
                ssim_roi = ssim_full
            
            # Background-only metrics - GT vs Model
            if np.sum(mask_bg) > 0:  # If background exists
                output_bg = output_img * mask_bg[:, :, None]
                gt_bg = gt_img * mask_bg[:, :, None]
                
                psnr_bg = util.calculate_psnr(output_bg, gt_bg, border=0)
                ssim_bg = util.calculate_ssim(output_bg, gt_bg, border=0)
            else:
                psnr_bg = psnr_full
                ssim_bg = ssim_full
            
            # Log metrics (GT vs Model only)
            logger.info(f'  Frame {original_frame_number:04d}:')
            logger.info(f'    FULL  - Model vs GT: {psnr_full:.2f}dB / {ssim_full:.4f}')
            logger.info(f'    ROI   - Model vs GT: {psnr_roi:.2f}dB / {ssim_roi:.4f} <-- PRIORITY')
            logger.info(f'    BG    - Model vs GT: {psnr_bg:.2f}dB / {ssim_bg:.4f}')
            
            # Store for summary
            test_results['psnr'].append(psnr_full)
            test_results['ssim'].append(ssim_full)
            
            # Create side-by-side comparison periodically (GT vs Model)
            if original_frame_number % 100 == 0 and parser.parse_args().save_comparison:
                comparison_img = create_comparison_image_gt_model(output_img, gt_img, psnr_full, ssim_full)
                comparison_path = os.path.join(save_comparison_dir, f'{video_name}_{original_frame_number:04d}_comparison.png')
                cv2.imwrite(comparison_path, comparison_img)
    
    # Overall Summary
    if len(test_results['psnr']) > 0:
        avg_psnr = np.mean(test_results['psnr'])
        avg_ssim = np.mean(test_results['ssim'])
        
        logger.info('\n' + '='*80)
        logger.info('OVERALL RESULTS (Model vs Ground Truth):')
        logger.info(f'  Average PSNR: {avg_psnr:.2f} dB')
        logger.info(f'  Average SSIM: {avg_ssim:.4f}')
        logger.info(f'  Total frames processed: {len(test_results["psnr"])}')
        logger.info('='*80)
    else:
        logger.info('Testing completed (no ground truth for metrics)')
    
    logger.info(f'\n✓ Output videos saved to: {save_output_dir}')
    logger.info(f'✓ Comparison images saved to: {save_comparison_dir}')
    logger.info(f'✓ Log file saved to: {os.path.join(save_dir, "test.log")}')


def create_comparison_image_gt_model(output_img, gt_img, psnr, ssim):
    """Create side-by-side comparison: Model Output | Ground Truth"""
    
    # Ensure all images are 3-channel
    if len(output_img.shape) == 2:
        output_img = cv2.cvtColor(output_img, cv2.COLOR_GRAY2BGR)
    if len(gt_img.shape) == 2:
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_GRAY2BGR)
    
    # Add text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    color_model = (0, 255, 0)  # Green
    color_gt = (255, 255, 0)  # Cyan
    
    # Create copies for text overlay
    output_labeled = output_img.copy()
    gt_labeled = gt_img.copy()
    
    # Add labels
    cv2.putText(output_labeled, 'MODEL OUTPUT', (10, 30), font, font_scale, color_model, thickness)
    cv2.putText(output_labeled, f'PSNR vs GT: {psnr:.2f}dB', (10, 60), font, font_scale, color_model, thickness)
    cv2.putText(output_labeled, f'SSIM vs GT: {ssim:.4f}', (10, 90), font, font_scale, color_model, thickness)
    
    cv2.putText(gt_labeled, 'GROUND TRUTH', (10, 30), font, font_scale, color_gt, thickness)
    
    # Concatenate horizontally
    comparison = np.hstack([output_labeled, gt_labeled])
    
    return comparison


if __name__ == '__main__':
    main()
