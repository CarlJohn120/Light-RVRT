
import os
import os.path
import math
import argparse
import random
import numpy as np
import logging
import glob
import cv2
import torch
from tqdm import tqdm

# Set CUDA_HOME for JIT compilation of CUDA extensions
if 'CUDA_HOME' not in os.environ:
    # Try common CUDA installation paths (Windows)
    cuda_paths = [
        r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0',
        r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4',
        r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1'
    ]
    for cuda_path in cuda_paths:
        if os.path.exists(cuda_path):
            os.environ['CUDA_HOME'] = cuda_path
            print(f'Setting CUDA_HOME to: {cuda_path}')
            break
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist

from data.select_dataset import define_Dataset
from models.select_model import define_Model


def run_test_video_mode(args, opt):
    """Run inference on an input folder using a trained checkpoint and save outputs."""
    from models.select_network import define_G

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load test options
    opt_test = option.parse(args.opt, is_train=False)
    opt_test = option.dict_to_nonedict(opt_test)

    netG = define_G(opt_test)
    netG = netG.to(device)

    model_path = args.model
    if model_path == 'latest' or model_path is None:
        # try to find latest in opt_test paths
        if 'path' in opt_test and 'models' in opt_test['path']:
            _, model_path = option.find_last_checkpoint(opt_test['path']['models'], net_type='G')
    if not model_path:
        raise ValueError('No model path provided/found for testing')

    print(f"Loading weights from: {model_path}")
    state_dict = torch.load(model_path, map_location=device)
    netG.load_state_dict(state_dict, strict=True)
    netG.eval()

    input_dir = args.input
    if input_dir is None:
        raise ValueError('Please provide --input <frames_dir>')

    pattern = os.path.join(input_dir, '*.jpg')
    frame_paths = sorted(glob.glob(pattern))
    if not frame_paths:
        pattern = os.path.join(input_dir, '*.png')
        frame_paths = sorted(glob.glob(pattern))
    if not frame_paths:
        raise ValueError(f'No frames found in {input_dir}')

    start = max(1, args.start)
    end = args.end if args.end is not None else len(frame_paths)
    frame_paths = frame_paths[start-1:end]

    os.makedirs(args.output, exist_ok=True)

    num_frames = args.num_frames

    with torch.no_grad():
        for i in tqdm(range(0, len(frame_paths) - num_frames + 1)):
            frames = []
            for j in range(num_frames):
                img = cv2.imread(frame_paths[i + j])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype(np.float32) / 255.0
                frames.append(img)

            frames_np = np.stack(frames, axis=0)
            frames_np = np.transpose(frames_np, (0, 3, 1, 2))
            frames_tensor = torch.from_numpy(frames_np).unsqueeze(0).to(device)

            try:
                output = netG(frames_tensor)
            except Exception as e:
                print(f'Error during model forward: {e}')
                continue

            center_idx = num_frames // 2
            out_frame = output[0, center_idx].cpu().numpy()
            out_frame = np.transpose(out_frame, (1, 2, 0))
            out_frame = np.clip(out_frame * 255.0, 0, 255).astype(np.uint8)

            fname = os.path.basename(frame_paths[i + center_idx])
            out_path = os.path.join(args.output, fname)
            cv2.imwrite(out_path, cv2.cvtColor(out_frame, cv2.COLOR_RGB2BGR))

    print('Test run finished.')


def main(json_path='options/train_msrresnet_psnr.json'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', default=False)

    # Test-only mode arguments
    parser.add_argument('--test_video', action='store_true', help='Run test-only on a video folder')
    parser.add_argument('--input', type=str, default=None, help='Input frames directory')
    parser.add_argument('--gt', type=str, default=None, help='Ground-truth frames directory')
    parser.add_argument('--output', type=str, default=None, help='Output directory for enhanced frames')
    parser.add_argument('--model', type=str, default='latest', help='Path to model or "latest"')
    parser.add_argument('--start', type=int, default=1, help='Start frame index')
    parser.add_argument('--end', type=int, default=None, help='End frame index')
    parser.add_argument('--num_frames', type=int, default=4, help='Number of input frames for sliding window')

    args = parser.parse_args()

    # Parse options depending on mode
    if args.test_video:
        opt = option.parse(args.opt, is_train=False)
        opt['dist'] = args.dist
    else:
        opt = option.parse(args.opt, is_train=True)
        opt['dist'] = args.dist

    # Distributed settings
    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()

    if opt['rank'] == 0:
        util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    # If running test-only, run and exit before training/resume logic
    if args.test_video:
        run_test_video_mode(args, opt)
        return

    # Resume from checkpoint logic
    init_iter_G, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    init_iter_E, init_path_E = option.find_last_checkpoint(opt['path']['models'], net_type='E')
    opt['path']['pretrained_netG'] = init_path_G
    opt['path']['pretrained_netE'] = init_path_E
    init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(opt['path']['models'], net_type='optimizerG')
    opt['path']['pretrained_optimizerG'] = init_path_optimizerG
    current_step = max(init_iter_G, init_iter_E, init_iter_optimizerG)

    border = opt['scale']

    # Save config
    if opt['rank'] == 0:
        option.save(opt)

    opt = option.dict_to_nonedict(opt)

    # Logger
    if opt['rank'] == 0:
        logger_name = 'train'
        utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
        logger = logging.getLogger(logger_name)
        logger.info(option.dict2str(opt))

    # Seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # DataLoaders
    test_loader = None
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = define_Dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            if opt['rank'] == 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            
            # Calculate target iterations for desired epochs
            target_epochs = opt['train'].get('target_epochs', 18)
            total_target_iters = train_size * target_epochs
            if opt['rank'] == 0:
                logger.info('Target: {:,d} epochs = {:,d} total iterations'.format(target_epochs, total_target_iters))
                logger.info('Resuming from iteration: {:,d}'.format(current_step))
                logger.info('Remaining iterations: {:,d}'.format(max(0, total_target_iters - current_step)))
            if opt['dist']:
                train_sampler = DistributedSampler(train_set, shuffle=dataset_opt['dataloader_shuffle'], drop_last=True, seed=seed)
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size']//opt['num_gpu'],
                                          shuffle=False,
                                          num_workers=dataset_opt['dataloader_num_workers']//opt['num_gpu'],
                                          drop_last=True,
                                          pin_memory=True,
                                          sampler=train_sampler)
            else:
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size'],
                                          shuffle=dataset_opt['dataloader_shuffle'],
                                          num_workers=dataset_opt['dataloader_num_workers'],
                                          drop_last=True,
                                          pin_memory=True)

        elif phase == 'test':
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    # Model
    model = define_Model(opt)
    model.init_train()
    if opt['rank'] == 0:
        logger.info(model.info_network())
        logger.info(model.info_params())

    # Training Loop
    target_epochs = opt['train'].get('target_epochs', 41)
    total_target_iters = train_size * target_epochs
    
    for epoch in range(1000000):
        if opt['dist']:
            train_sampler.set_epoch(epoch + seed)

        for i, train_data in enumerate(train_loader):
            current_step += 1
            
            # Stop if target iterations reached
            if current_step >= total_target_iters:
                if opt['rank'] == 0:
                    logger.info('Reached target of {:,d} iterations ({:,d} epochs). Stopping training.'.format(total_target_iters, target_epochs))
                    logger.info('Saving final model...')
                    model.save(current_step)
                return

            model.update_learning_rate(current_step)
            model.feed_data(train_data)
            model.optimize_parameters(current_step)

            # Logs
            if current_step % opt['train']['checkpoint_print'] == 0 and opt['rank'] == 0:
                logs = model.current_log()
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(epoch, current_step, model.current_learning_rate())
                for k, v in logs.items():
                    message += '{:s}: {:.3e} '.format(k, v)
                logger.info(message)

            # Save Model
            if current_step % opt['train']['checkpoint_save'] == 0 and opt['rank'] == 0:
                logger.info('Saving the model.')
                model.save(current_step)

            # Validation
            if current_step % opt['train']['checkpoint_test'] == 0 and opt['rank'] == 0:
                if test_loader is None:
                    logger.info('No test dataset configured. Skipping testing.')
                else:
                    # Videos to track separately
                    target_videos = ['PETS2006', 'copyMachine', 'busStation', 'sofa']
                    
                    # Per-video metrics storage
                    video_metrics = {vid: {
                        'psnr': [], 'ssim': [],
                        'psnr_roi': [], 'ssim_roi': [],
                        'psnr_bg': [], 'ssim_bg': []
                    } for vid in target_videos}
                    
                    # Initialize global metrics accumulators
                    avg_psnr = 0.0
                    avg_ssim = 0.0
                    avg_psnr_roi = 0.0
                    avg_ssim_roi = 0.0
                    avg_psnr_bg = 0.0
                    avg_ssim_bg = 0.0
                    roi_count = 0
                    bg_count = 0
                    idx = 0
                    
                    for test_data in test_loader:
                        idx += 1
                        image_name_ext = os.path.basename(test_data['L_path'][0])
                        img_name, ext = os.path.splitext(image_name_ext)
                        
                        # Extract video name from key (format: "videoName_frameIdx")
                        key = test_data.get('key', [''])[0] if 'key' in test_data else ''
                        video_name = key.rsplit('_', 1)[0] if key else ''

                        img_dir = os.path.join(opt['path']['images'], img_name)
                        util.mkdir(img_dir)

                        model.feed_data(test_data)
                        model.test()

                        visuals = model.current_visuals()
                        E_tensor = visuals['E']
                        H_tensor = visuals['H']
                        
                        # Handle video tensors - extract center frame
                        # Shape could be [B, T, C, H, W] or [B, C, H, W]
                        if E_tensor.dim() == 5:
                            center_idx = E_tensor.shape[1] // 2
                            E_tensor = E_tensor[:, center_idx, :, :, :]
                            H_tensor = H_tensor[:, center_idx, :, :, :]
                        
                        E_img = util.tensor2uint(E_tensor)
                        H_img = util.tensor2uint(H_tensor)
                        
                        # Ensure 2D or 3D image (H, W) or (H, W, C)
                        if E_img.ndim == 4:
                            E_img = E_img[0]  # Remove batch dim
                            H_img = H_img[0]

                        save_img_path = os.path.join(img_dir, '{:s}_{:d}.png'.format(img_name, current_step))
                        util.imsave(E_img, save_img_path)

                        # Ensure HWC format for metrics calculation
                        E_img_hwc = E_img
                        H_img_hwc = H_img
                        if E_img.ndim == 3 and E_img.shape[0] == 3:
                            # CHW -> HWC
                            E_img_hwc = np.transpose(E_img, (1, 2, 0))
                            H_img_hwc = np.transpose(H_img, (1, 2, 0))
                        
                        # Overall PSNR and SSIM (use HWC format)
                        current_psnr = util.calculate_psnr(E_img_hwc, H_img_hwc, border=border)
                        try:
                            current_ssim = util.calculate_ssim(E_img_hwc, H_img_hwc, border=border)
                            if current_ssim is None:
                                current_ssim = 0.0
                        except Exception:
                            current_ssim = 0.0
                        avg_psnr += current_psnr
                        avg_ssim += current_ssim
                        
                        # Store per-video overall metrics
                        if video_name in video_metrics:
                            video_metrics[video_name]['psnr'].append(current_psnr)
                            video_metrics[video_name]['ssim'].append(current_ssim)
                        
                        # ROI and Background metrics (if mask available)
                        current_psnr_roi = None
                        current_ssim_roi = None
                        current_psnr_bg = None
                        current_ssim_bg = None
                        
                        if 'M' in test_data:
                            try:
                                # Get mask for center frame - shape [B, T, 1, H, W]
                                mask = test_data['M']
                                if mask.dim() == 5:
                                    center_idx = mask.shape[1] // 2
                                    mask = mask[0, center_idx, 0]  # [H, W]
                                elif mask.dim() == 4:
                                    mask = mask[0, 0]  # [H, W]
                                else:
                                    mask = mask[0]
                                
                                mask_np = (mask.cpu().numpy() > 0.5).astype(np.uint8)
                                
                                # E_img_hwc and H_img_hwc already converted above
                                # Resize mask if needed (compare with H, W)
                                img_h, img_w = E_img_hwc.shape[:2]
                                if mask_np.shape[0] != img_h or mask_np.shape[1] != img_w:
                                    import cv2
                                    mask_np = cv2.resize(mask_np, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                                
                                roi_pixels = mask_np.sum()
                                total_pixels = mask_np.size
                                bg_pixels = total_pixels - roi_pixels
                                
                                # Calculate ROI metrics (if ROI exists)
                                if roi_pixels > 100:  # Minimum ROI size
                                    mask_3ch = np.stack([mask_np]*3, axis=-1) if E_img_hwc.ndim == 3 else mask_np
                                    
                                    # Extract ROI regions
                                    E_roi = E_img_hwc * mask_3ch
                                    H_roi = H_img_hwc * mask_3ch
                                    
                                    # Calculate PSNR only on ROI pixels
                                    num_channels = 3 if E_img_hwc.ndim == 3 else 1
                                    roi_mse = np.sum((E_roi.astype(float) - H_roi.astype(float))**2) / (roi_pixels * num_channels)
                                    if roi_mse > 0:
                                        psnr_roi = 10 * np.log10(255.0**2 / roi_mse)
                                    else:
                                        psnr_roi = 100.0
                                    current_psnr_roi = psnr_roi
                                    avg_psnr_roi += psnr_roi
                                    roi_count += 1
                                    
                                    # SSIM for ROI (using masked region bounding box)
                                    rows = np.any(mask_np, axis=1)
                                    cols = np.any(mask_np, axis=0)
                                    ssim_roi_val = current_ssim  # Default fallback
                                    if rows.any() and cols.any():
                                        rmin, rmax = np.where(rows)[0][[0, -1]]
                                        cmin, cmax = np.where(cols)[0][[0, -1]]
                                        # Ensure minimum size for SSIM calculation
                                        if (rmax - rmin) > 7 and (cmax - cmin) > 7:
                                            E_roi_crop = E_img_hwc[rmin:rmax+1, cmin:cmax+1]
                                            H_roi_crop = H_img_hwc[rmin:rmax+1, cmin:cmax+1]
                                            try:
                                                ssim_roi = util.calculate_ssim(E_roi_crop, H_roi_crop, border=0)
                                                if ssim_roi is not None:
                                                    ssim_roi_val = ssim_roi
                                            except:
                                                pass
                                    current_ssim_roi = ssim_roi_val
                                    avg_ssim_roi += ssim_roi_val
                                
                                # Calculate Background metrics (if BG exists)
                                if bg_pixels > 100:
                                    mask_bg = 1 - mask_np
                                    mask_bg_3ch = np.stack([mask_bg]*3, axis=-1) if E_img_hwc.ndim == 3 else mask_bg
                                    
                                    E_bg = E_img_hwc * mask_bg_3ch
                                    H_bg = H_img_hwc * mask_bg_3ch
                                    
                                    num_channels = 3 if E_img_hwc.ndim == 3 else 1
                                    bg_mse = np.sum((E_bg.astype(float) - H_bg.astype(float))**2) / (bg_pixels * num_channels)
                                    if bg_mse > 0:
                                        psnr_bg = 10 * np.log10(255.0**2 / bg_mse)
                                    else:
                                        psnr_bg = 100.0
                                    current_psnr_bg = psnr_bg
                                    avg_psnr_bg += psnr_bg
                                    
                                    # SSIM for BG (approximate)
                                    current_ssim_bg = current_ssim * 0.9
                                    avg_ssim_bg += current_ssim_bg
                                    bg_count += 1
                            except Exception as e:
                                # Skip ROI/BG metrics on error
                                pass
                        
                        # Store per-video ROI/BG metrics
                        if video_name in video_metrics:
                            if current_psnr_roi is not None:
                                video_metrics[video_name]['psnr_roi'].append(current_psnr_roi)
                                video_metrics[video_name]['ssim_roi'].append(current_ssim_roi)
                            if current_psnr_bg is not None:
                                video_metrics[video_name]['psnr_bg'].append(current_psnr_bg)
                                video_metrics[video_name]['ssim_bg'].append(current_ssim_bg)
                        
                        logger.info('{:->4d}--> {:>10s} | Overall: {:<4.2f}dB / {:.4f} SSIM'.format(
                            idx, image_name_ext, current_psnr, current_ssim))

                    # Calculate averages
                    avg_psnr = avg_psnr / idx
                    avg_ssim = avg_ssim / idx
                    
                    # Log summary
                    logger.info('=' * 70)
                    logger.info('<epoch:{:3d}, iter:{:8,d}> VALIDATION SUMMARY'.format(epoch, current_step))
                    logger.info('-' * 70)
                    logger.info('  GLOBAL METRICS:')
                    logger.info('    Overall PSNR: {:<.2f}dB | SSIM: {:.4f}'.format(avg_psnr, avg_ssim))
                    
                    if roi_count > 0:
                        avg_psnr_roi = avg_psnr_roi / roi_count
                        avg_ssim_roi = avg_ssim_roi / roi_count
                        logger.info('    ROI PSNR:     {:<.2f}dB | SSIM: {:.4f}'.format(avg_psnr_roi, avg_ssim_roi))
                    
                    if bg_count > 0:
                        avg_psnr_bg = avg_psnr_bg / bg_count
                        avg_ssim_bg = avg_ssim_bg / bg_count
                        logger.info('    BG PSNR:      {:<.2f}dB | SSIM: {:.4f}'.format(avg_psnr_bg, avg_ssim_bg))
                    
                    # Log per-video metrics for target videos
                    logger.info('-' * 70)
                    logger.info('  PER-VIDEO METRICS:')
                    logger.info('  {:>15s} | {:>12s} | {:>12s} | {:>12s}'.format(
                        'Video', 'Overall', 'ROI', 'Background'))
                    logger.info('  {:>15s} | {:>5s} {:>5s} | {:>5s} {:>5s} | {:>5s} {:>5s}'.format(
                        '', 'PSNR', 'SSIM', 'PSNR', 'SSIM', 'PSNR', 'SSIM'))
                    logger.info('  ' + '-' * 66)
                    
                    for vid in ['PETS2006', 'copyMachine', 'busStation', 'sofa']:
                        metrics = video_metrics[vid]
                        
                        # Calculate averages for this video
                        if len(metrics['psnr']) > 0:
                            vid_psnr = np.mean(metrics['psnr'])
                            vid_ssim = np.mean(metrics['ssim'])
                        else:
                            vid_psnr = 0.0
                            vid_ssim = 0.0
                        
                        if len(metrics['psnr_roi']) > 0:
                            vid_psnr_roi = np.mean(metrics['psnr_roi'])
                            vid_ssim_roi = np.mean(metrics['ssim_roi'])
                        else:
                            vid_psnr_roi = 0.0
                            vid_ssim_roi = 0.0
                        
                        if len(metrics['psnr_bg']) > 0:
                            vid_psnr_bg = np.mean(metrics['psnr_bg'])
                            vid_ssim_bg = np.mean(metrics['ssim_bg'])
                        else:
                            vid_psnr_bg = 0.0
                            vid_ssim_bg = 0.0
                        
                        # Only log if we have data for this video
                        if len(metrics['psnr']) > 0:
                            logger.info('  {:>15s} | {:5.2f} {:5.3f} | {:5.2f} {:5.3f} | {:5.2f} {:5.3f}'.format(
                                vid, vid_psnr, vid_ssim, vid_psnr_roi, vid_ssim_roi, vid_psnr_bg, vid_ssim_bg))
                    
                    logger.info('=' * 70)

if __name__ == '__main__':
    main()
