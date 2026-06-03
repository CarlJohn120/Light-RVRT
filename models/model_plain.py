from collections import OrderedDict
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler
from models.select_network import define_G
from models.model_base import ModelBase
from models.loss import CharbonnierLoss
from models.loss_ssim import SSIMLoss
from utils.utils_model import test_mode

class ModelPlain(ModelBase):
    def __init__(self, opt):
        super(ModelPlain, self).__init__(opt)
        self.opt_train = self.opt['train']
        self.netG = define_G(opt)
        self.netG = self.model_to_device(self.netG)
        if self.opt_train['E_decay'] > 0:
            self.netE = define_G(opt).to(self.device).eval()

    def init_train(self):
        self.load()
        self.netG.train()
        self.define_loss()
        self.define_optimizer()
        self.load_optimizers()
        self.define_scheduler()
        self.log_dict = OrderedDict()
        
        # Initialize AMP (Mixed Precision Training)
        self.use_amp = self.opt_train.get('use_amp', False)
        if self.use_amp:
            self.scaler = GradScaler()
            print('✅ Mixed Precision Training (AMP) enabled - ~40% memory savings, ~30% faster')

    def load(self):
        load_path_G = self.opt['path']['pretrained_netG']
        if load_path_G is not None:
            print('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, strict=self.opt_train['G_param_strict'], param_key='params')
        load_path_E = self.opt['path']['pretrained_netE']
        if self.opt_train['E_decay'] > 0:
            if load_path_E is not None:
                self.load_network(load_path_E, self.netE, strict=self.opt_train['E_param_strict'], param_key='params_ema')
            else:
                self.update_E(0)
            self.netE.eval()

    def load_optimizers(self):
        load_path_optimizerG = self.opt['path']['pretrained_optimizerG']
        if load_path_optimizerG is not None and self.opt_train['G_optimizer_reuse']:
            self.load_optimizer(load_path_optimizerG, self.G_optimizer)

    def save(self, iter_label):
        self.save_network(self.save_dir, self.netG, 'G', iter_label)
        if self.opt_train['E_decay'] > 0:
            self.save_network(self.save_dir, self.netE, 'E', iter_label)
        if self.opt_train['G_optimizer_reuse']:
            self.save_optimizer(self.save_dir, self.G_optimizer, 'optimizerG', iter_label)

    def define_loss(self):
        G_lossfn_type = self.opt_train['G_lossfn_type']
        if G_lossfn_type == 'l1': self.G_lossfn = nn.L1Loss().to(self.device)
        elif G_lossfn_type == 'l2': self.G_lossfn = nn.MSELoss().to(self.device)
        elif G_lossfn_type == 'charbonnier': self.G_lossfn = CharbonnierLoss(self.opt_train['G_charbonnier_eps']).to(self.device)
        else: raise NotImplementedError('Loss type [{:s}] is not found.'.format(G_lossfn_type))
        self.G_lossfn_weight = self.opt_train['G_lossfn_weight']

    def define_optimizer(self):
        G_optim_params = []
        for k, v in self.netG.named_parameters():
            if v.requires_grad: G_optim_params.append(v)
        if self.opt_train['G_optimizer_type'] == 'adam':
            self.G_optimizer = Adam(G_optim_params, lr=self.opt_train['G_optimizer_lr'], betas=self.opt_train['G_optimizer_betas'], weight_decay=self.opt_train['G_optimizer_wd'])
        else: raise NotImplementedError

    def define_scheduler(self):
        if self.opt_train['G_scheduler_type'] == 'MultiStepLR':
            self.schedulers.append(lr_scheduler.MultiStepLR(self.G_optimizer, self.opt_train['G_scheduler_milestones'], self.opt_train['G_scheduler_gamma']))
        else: raise NotImplementedError

    # --- CRITICAL FIX: Save Data Dict ---
    def feed_data(self, data, need_H=True):
        self.data = data 
        self.L = data['L'].to(self.device)
        if need_H: self.H = data['H'].to(self.device)

    def netG_forward(self):
        self.E = self.netG(self.L)

    def optimize_parameters(self, current_step):
        self.G_optimizer.zero_grad()
        
        # --- THESIS LOGIC WITH AMP SUPPORT ---
        if self.use_amp:
            with autocast():
                self.E = self.netG(self.L)
                
                # 1. Load Mask (Range 0 to 1)
                mask = self.data['M'].to(self.device)

                # 2. Calculate Thesis Weights
                thesis_weights = mask * 1.0 + (1.0 - mask) * 0.01

                # 3. Calculate Basic Pixel Loss
                if self.opt['train']['G_lossfn_type'] == 'charbonnier':
                    eps = 1e-12
                    diff = torch.add(self.E, -self.H)
                    error = torch.sqrt(diff * diff + eps)
                elif self.opt['train']['G_lossfn_type'] == 'l1':
                    error = torch.abs(self.E - self.H)
                else:
                    error = torch.abs(self.E - self.H)

                # 4. Apply Thesis Weights
                weighted_error = error * thesis_weights
                
                # 5. False Negative Mitigation
                if mask.dim() == 5:
                    mask_coverage = mask.sum(dim=[1, 2, 3, 4]) / mask[0].numel()
                else:
                    mask_coverage = mask.sum(dim=[1, 2, 3]) / mask[0].numel()
                
                valid_samples = (mask_coverage > 0.01).float()
                
                if weighted_error.dim() == 5:
                    loss_per_sample = weighted_error.mean(dim=[1, 2, 3, 4])
                else:
                    loss_per_sample = weighted_error.mean(dim=[1, 2, 3])
                
                masked_loss = loss_per_sample * valid_samples
                num_valid = valid_samples.sum() + 1e-8
                loss_G_total = masked_loss.sum() / num_valid
            
            # AMP backward pass
            self.scaler.scale(loss_G_total).backward()
            self.scaler.step(self.G_optimizer)
            self.scaler.update()
        else:
            # Original FP32 path
            self.E = self.netG(self.L)
            
            # 1. Load Mask (Range 0 to 1)
            mask = self.data['M'].to(self.device)

            # 2. Calculate Thesis Weights
            thesis_weights = mask * 1.0 + (1.0 - mask) * 0.01

            # 3. Calculate Basic Pixel Loss
            if self.opt['train']['G_lossfn_type'] == 'charbonnier':
                eps = 1e-12
                diff = torch.add(self.E, -self.H)
                error = torch.sqrt(diff * diff + eps)
            elif self.opt['train']['G_lossfn_type'] == 'l1':
                error = torch.abs(self.E - self.H)
            else:
                error = torch.abs(self.E - self.H)

            # 4. Apply Thesis Weights
            weighted_error = error * thesis_weights
            
            # 5. False Negative Mitigation
            if mask.dim() == 5:
                mask_coverage = mask.sum(dim=[1, 2, 3, 4]) / mask[0].numel()
            else:
                mask_coverage = mask.sum(dim=[1, 2, 3]) / mask[0].numel()
            
            valid_samples = (mask_coverage > 0.01).float()
            
            if weighted_error.dim() == 5:
                loss_per_sample = weighted_error.mean(dim=[1, 2, 3, 4])
            else:
                loss_per_sample = weighted_error.mean(dim=[1, 2, 3])
            
            masked_loss = loss_per_sample * valid_samples
            num_valid = valid_samples.sum() + 1e-8
            loss_G_total = masked_loss.sum() / num_valid

            loss_G_total.backward()
            self.G_optimizer.step()
        
        # Logging
        self.log_dict['G_loss'] = loss_G_total.item()
        self.log_dict['valid_samples'] = valid_samples.sum().item()
        self.log_dict['total_samples'] = valid_samples.numel()
        
        current_step += 1
        return current_step

    def test(self):
        self.netG.eval()
        with torch.no_grad(): self.netG_forward()
        self.netG.train()

    def current_log(self): return self.log_dict
    def current_visuals(self, need_H=True):
        out_dict = OrderedDict()
        out_dict['L'] = self.L.detach()[0].float().cpu()
        out_dict['E'] = self.E.detach()[0].float().cpu()
        if need_H: out_dict['H'] = self.H.detach()[0].float().cpu()
        return out_dict
    def print_network(self): pass