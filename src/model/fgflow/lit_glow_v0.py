import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.utils import make_grid

from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from ..common.lit_basemodel import LitBaseModel
from .glow_64x64_v0 import Glow64x64V0
from .glow_64x64_v1 import Glow64x64V1
from .glow_256x256_v0 import Glow256x256V0
from loss import NLLLoss, TripletLoss, MSELoss, L1Loss, PerceptualLoss, IDLoss, GANLoss
from metric import L1, PSNR, SSIM

import os
import numbers
import numpy as np
from PIL import Image
from collections import OrderedDict

import cv2

# NLL + quant_randomness
class LitGlowV0(LitBaseModel):
    def __init__(self,
                 opt: dict,
                 pretrained=None):

        super().__init__()

        # opt
        if pretrained is True:
            opt['flow_net']['args']['pretrained'] = True
        self.opt = opt

        # network
        flow_nets = {
            'Glow64x64V0': Glow64x64V0,
            'Glow64x64V1': Glow64x64V1,
            'Glow256x256V0': Glow256x256V0,
        }

        self.opt = opt
        self.flow_net = flow_nets[opt['flow_net']['type']](**opt['flow_net']['args'])
        self.in_size = self.opt['in_size']
        self.n_bits = self.opt['n_bits']
        self.n_bins = 2.0**self.n_bits

        self.norm_mean = [0.5, 0.5, 0.5]
        self.norm_std = [1.0, 1.0, 1.0] #[0.5, 0.5, 0.5]
        # self.norm_mean = [0.485, 0.456, 0.406]
        # self.norm_std = [0.229, 0.224, 0.225]

        self.preprocess = transforms.Normalize(
            mean=self.norm_mean, 
            std=self.norm_std)
        self.reverse_preprocess = transforms.Normalize(
            mean=[-m/s for m,s in zip(self.norm_mean, self.norm_std)],
            std=[1/s for s in self.norm_std])

        # loss
        self._create_loss(opt['loss'])
        
        # metric
        self.sampled_images = []
        
        # log
        self.save_hyperparameters(ignore=[])

        # pretrained
        self.pretrained = pretrained
        
    def forward(self, x):
        pass

    def preprocess_batch(self, batch):
        # Data Preprocess
        im = batch        
        im = im * 255

        if self.n_bits < 8:
            im = torch.floor(im / 2 ** (8 - self.n_bits))
        im = im / self.n_bins
        im = self.preprocess(im)
        
        conditions = [None] * (len(self.flow_net.blocks) + len(self.flow_net.headers))

        return im, conditions

    def training_step(self, batch, batch_idx):
        im, conditions = self.preprocess_batch(batch)

        # Forward
        # quant_randomness = torch.zeros_like(im)
        quant_randomness = self.preprocess(torch.rand_like(im)/self.n_bins) - self.preprocess(torch.zeros_like(im)) # x = (0~1)/n_bins, \ (im-m)/s + (x-m)/s - (0-m)/s = (im+x-m)/s
        w, log_p, log_det, _, _ = self.flow_net.forward(im + quant_randomness, conditions)
        
        # Loss
        losses = dict()
        losses['loss_nll'], log_nll = self.loss_nll(log_p, log_det, n_pixel=3*self.in_size*self.in_size)
        loss_total_common = sum(losses.values())
        
        log_train = {
            'train/loss_nll': log_nll,
            'train/loss_total_common': loss_total_common,
        }
        
        # Log
        self.log_dict(log_train, logger=True, prog_bar=True)
        
        # Total Loss
        return loss_total_common

    def validation_step(self, batch, batch_idx):
        im, conditions = self.preprocess_batch(batch)

        # Forward
        w, log_p, log_det, splits, _ = self.flow_net.forward(im, conditions)

        # Reverse_function
        def compute_im_recon(w, conditions, splits):
            # Flow.reverse
            im_rec = self.flow_net.reverse(w, conditions, splits)
            # Range : (-0.5, 0.5) -> (0,1)
            im_rec = self.reverse_preprocess(im_rec)
            # Clamp : (0,1)
            im_rec = torch.clamp(im_rec, 0, 1)
            return im_rec
    
        # Reverse - Latent to Image
        w_rand =  self.flow_net.final_temp * torch.randn_like(w)
        w_rand_temp = 0.7 * w_rand
        splits_rand = [torch.randn_like(split) * self.flow_net.inter_temp if split is not None else None for split in splits]
        splits_rand_temp = [0.7 * split if split is not None else None for split in splits_rand]
        splits_zero = [torch.zeros_like(split) if split is not None else None for split in splits]

        # RECONS
        im = torch.clamp(self.reverse_preprocess(im), 0, 1)
        im_recs = compute_im_recon(w, conditions, splits)
        im_recr = compute_im_recon(w, conditions, splits_rand)
        im_recr_t = compute_im_recon(w, conditions, splits_rand_temp)
        im_recr_z = compute_im_recon(w, conditions, splits_zero)
        # GEN-Random
        im_genr = compute_im_recon(w_rand, conditions, splits_rand)
        im_gent = compute_im_recon(w_rand_temp, conditions, splits_rand_temp)
        
        
        # Metric - Image, CHW
        if batch_idx < 10:
            self.sampled_images.append(im[0].cpu())
            self.sampled_images.append(im_recs[0].cpu())
            
            self.sampled_images.append(im_recr[0].cpu())
            self.sampled_images.append(im_recr_t[0].cpu())
            self.sampled_images.append(im_recr_z[0].cpu())
            
            self.sampled_images.append(im_genr[0].cpu())
            self.sampled_images.append(im_gent[0].cpu())
            
        # Metric - PSNR, SSIM
        im = im[0].cpu().numpy().transpose(1,2,0)
        im_recr = im_recr[0].cpu().numpy().transpose(1,2,0)
        im_recr_t = im_recr_t[0].cpu().numpy().transpose(1,2,0)
        im_recr_z = im_recr_z[0].cpu().numpy().transpose(1,2,0)
        metric_psnr_r = PSNR(im_recr*255, im*255) 
        metric_ssim_r = SSIM(im_recr*255, im*255)
        metric_psnr_t = PSNR(im_recr_t*255, im*255) 
        metric_ssim_t = SSIM(im_recr_t*255, im*255)
        metric_psnr_z = PSNR(im_recr_z*255, im*255) 
        metric_ssim_z = SSIM(im_recr_z*255, im*255)

        # Metric - Objective Functions
        loss_nll, metric_nll = self.loss_nll(log_p, log_det, n_pixel=3*self.in_size*self.in_size)

        log_valid = {
            'val/metric/loss': loss_nll,
            'val/metric/nll': metric_nll,
            'val/metric/psnr_r': metric_psnr_r,
            'val/metric/ssim_r': metric_ssim_r,
            'val/metric/psnr_r_temp': metric_psnr_t,
            'val/metric/ssim_r_temp': metric_ssim_t,
            'val/metric/psnr_r_zero': metric_psnr_z,
            'val/metric/ssim_r_zero': metric_ssim_z,}
        self.log_dict(log_valid)

    def test_step(self, batch, batch_idx):
        # if batch_idx == 0:
        #     torch.save(self.flow_net.state_dict(), 'fgflow.ckpt')
        self.validation_step(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        # Log Qualative Result - Image
        grid = make_grid(self.sampled_images, nrow=7)
        if isinstance(self.logger, TensorBoardLogger):
            self.logger.experiment.add_image(
                f'val/visualization',
                grid, self.global_step+1, dataformats='CHW')
        self.sampled_images = []

        # Update hyper-params if necessary
        if self.current_epoch % 10 == 0:
            self.n_bits = min(self.n_bits+1, 8)
            self.n_bins = 2.0**self.n_bits
            self.loss_nll.n_bits = self.n_bits
            self.loss_nll.n_bins = self.n_bins

    def test_epoch_end(self, outputs):
        pass

    def configure_optimizers(self):
        trainable_parameters = [*self.flow_net.parameters()]

        optimizer = Adam(
            trainable_parameters, 
            lr=self.opt['optim']['lr'], 
            betas=self.opt['optim']['betas'])
    
        scheduler = {
            'scheduler': CosineAnnealingLR(
                optimizer, 
                T_max=self.opt['scheduler']['T_max'], 
                eta_min=self.opt['scheduler']['eta_min']),
            'name': 'learning_rate'}

        return [optimizer], [scheduler]
    
    def _create_loss(self, opt):
        losses = {
            'NLLLoss': NLLLoss,
            'TripletLoss': TripletLoss,
            'MSELoss': MSELoss,
            'L1Loss': L1Loss,
            'PerceptualLoss': PerceptualLoss,
            'IDLoss': IDLoss,
            'GANLoss': GANLoss
        }
        
        self.loss_nll = losses[opt['nll']['type']](**opt['nll']['args'])
       
