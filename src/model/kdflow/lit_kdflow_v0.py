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
from .glow_64x64_v1 import Glow64x64V1
from .glow_256x256_v0 import Glow256x256V0
from .vgg_header import get_vgg_header
from loss import NLLLoss, TripletLoss, MSELoss, L1Loss, PerceptualLoss, IDLoss, GANLoss
from metric import L1, PSNR, SSIM

import os
import numbers
import numpy as np
from PIL import Image
from collections import OrderedDict

import cv2

class LitKDFlowV0(LitBaseModel):
    def __init__(self,
                 opt: dict,
                 pretrained=None,
                 strict_load=True):

        super().__init__()

        # network
        flow_nets = {
            'Glow64x64V1': Glow64x64V1,
            'Glow256x256V0': Glow256x256V0,
        }

        self.opt = opt
        self.flow_net = flow_nets[opt['flow_net']['type']](**opt['flow_net']['args'])
        self.in_size = self.opt['in_size']
        self.n_bits = self.opt['n_bits']
        self.n_bins = 2.0**self.n_bits

        self.vgg_blocks = nn.Sequential(
            torchvision.models.vgg16(pretrained=True).features[:4].eval(),      # 64,64,64 
            torchvision.models.vgg16(pretrained=True).features[4:9].eval(),     # 128,32,32
            torchvision.models.vgg16(pretrained=True).features[9:16].eval(),    # 256,16,16
            torchvision.models.vgg16(pretrained=True).features[16:23].eval())   # 512,8,8
        self.vgg_headers = nn.Sequential(
            get_vgg_header(6,32,64,3),
            get_vgg_header(12,64,128,3),
            get_vgg_header(24,128,256,3),
            get_vgg_header(48,256,512,3),            
        )

        self.norm_mean = [0.5, 0.5, 0.5]
        self.norm_std = [1.0, 1.0, 1.0] #[0.5, 0.5, 0.5]
        self.vgg_norm_mean = [0.485, 0.456, 0.406]
        self.vgg_norm_std = [0.229, 0.224, 0.225]
                
        self.preprocess = transforms.Normalize(
            mean=self.norm_mean, 
            std=self.norm_std)
        self.reverse_preprocess = transforms.Normalize(
            mean=[-m/s for m,s in zip(self.norm_mean, self.norm_std)],
            std=[1/s for s in self.norm_std])

        self.vgg_preprocess = transforms.Normalize(
            mean=self.vgg_norm_mean, 
            std=self.vgg_norm_std)
        
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
        # Data Quantization
        im = batch        
        im = im * 255

        if self.n_bits < 8:
            im = torch.floor(im / 2 ** (8 - self.n_bits))
        im = im / self.n_bins

        # Image preprocess
        im_resized = T.Resize(self.in_size//2, interpolation=InterpolationMode.BICUBIC, antialias=True)(im)
        im = self.preprocess(im)

        # VGG Guidance
        vgg_features = []
        with torch.no_grad():
            feature = self.vgg_preprocess(im_resized)
            for block in self.vgg_blocks:
                feature = block(feature)
                vgg_features.append(feature)

        # Conditions for affine-coupling layers
        conditions = [None] * (len(self.flow_net.blocks) + len(self.flow_net.headers))

        return im, conditions, vgg_features

    def training_step(self, batch, batch_idx):
        im, conditions, vgg_features = self.preprocess_batch(batch)

        # Forward
        # quant_randomness = torch.zeros_like(im)
        quant_randomness = self.preprocess(torch.rand_like(im)/self.n_bins) - self.preprocess(torch.zeros_like(im)) # x = (0~1)/n_bins, \ (im-m)/s + (x-m)/s - (0-m)/s = (im+x-m)/s
        w, log_p, log_det, _splits, inter_features = self.flow_net.forward(im + quant_randomness, conditions)
        inter_features = [ vgg_header(inter_feature) for vgg_header, inter_feature in zip(self.vgg_headers, inter_features[:4]) ]

        # Loss
        losses = dict()
        losses['loss_nll'], log_nll = self.loss_nll(log_p, log_det, n_pixel=3*self.in_size*self.in_size)
        losses['loss_fg0'], log_fg0 = self.loss_fg(inter_features[0], vgg_features[0], weight=self.loss_fg_weights[0])
        losses['loss_fg1'], log_fg1 = self.loss_fg(inter_features[1], vgg_features[1], weight=self.loss_fg_weights[1])
        losses['loss_fg2'], log_fg2 = self.loss_fg(inter_features[2], vgg_features[2], weight=self.loss_fg_weights[2])
        losses['loss_fg3'], log_fg3 = self.loss_fg(inter_features[3], vgg_features[3], weight=self.loss_fg_weights[3])
        loss_total_common = sum(losses.values())
        
        log_train = {
            'train/loss_nll': log_nll,
            'train/loss_fg0': log_fg0,
            'train/loss_fg1': log_fg1,
            'train/loss_fg2': log_fg2,
            'train/loss_fg3': log_fg3,
            'train/loss_total_common': loss_total_common,
        }
        
        # Log
        self.log_dict(log_train, logger=True, prog_bar=True)
        
        # Total Loss
        return loss_total_common

    def validation_step(self, batch, batch_idx):
        im, conditions, vgg_features = self.preprocess_batch(batch)

        # Forward
        w, log_p, log_det, splits, inter_features = self.flow_net.forward(im, conditions)
        inter_features = [ vgg_header(inter_feature) for vgg_header, inter_feature in zip(self.vgg_headers, inter_features[:4]) ]

        # Reverse - Latent to Image
        w_rand = torch.randn_like(w)
        w_rand_temp = w_rand * 0.7
        splits_random = [torch.randn_like(split) if split is not None else None for split in splits]
        splits_random_temp = [0.7*split if split is not None else None for split in splits_random]

        im_recs = self.flow_net.reverse(w, conditions, splits=splits)
        im_recr = self.flow_net.reverse(w, conditions, splits=splits_random)
        im_recr_temp = self.flow_net.reverse(w, conditions, splits=splits_random_temp)
        im_gen = self.flow_net.reverse(w_rand, conditions, splits=splits_random)
        im_gen_temp = self.flow_net.reverse(w_rand_temp, conditions, splits=splits_random_temp)
        
        # Format - range (0~1)
        im = torch.clamp(self.reverse_preprocess(im), 0, 1)
        im_recs = torch.clamp(self.reverse_preprocess(im_recs), 0, 1)
        im_recr = torch.clamp(self.reverse_preprocess(im_recr), 0, 1)
        im_recr_temp = torch.clamp(self.reverse_preprocess(im_recr_temp), 0, 1)
        im_gen = torch.clamp(self.reverse_preprocess(im_gen), 0, 1)
        im_gen_temp = torch.clamp(self.reverse_preprocess(im_gen_temp), 0, 1)
        
        # Metric - Image, CHW
        if batch_idx < 10:
            self.sampled_images.append(im[0].cpu())
            self.sampled_images.append(im_recs[0].cpu())
            self.sampled_images.append(im_recr[0].cpu())
            self.sampled_images.append(im_recr_temp[0].cpu())
            self.sampled_images.append(im_gen[0].cpu())
            self.sampled_images.append(im_gen_temp[0].cpu())
            
        # Metric - PSNR, SSIM
        im = im[0].cpu().numpy().transpose(1,2,0)
        im_recs = im_recs[0].cpu().numpy().transpose(1,2,0)
        im_recr = im_recr[0].cpu().numpy().transpose(1,2,0)
        im_recr_temp = im_recr_temp[0].cpu().numpy().transpose(1,2,0)
        im_gen = im_gen[0].cpu().numpy().transpose(1,2,0)
        metric_psnr_r = PSNR(im_recr*255, im*255) 
        metric_ssim_r = SSIM(im_recr*255, im*255)
        metric_psnr_r_temp = PSNR(im_recr_temp*255, im*255) 
        metric_ssim_r_temp = SSIM(im_recr_temp*255, im*255)

        # Metric - Objective Functions
        loss_nll, metric_nll = self.loss_nll(log_p, log_det, n_pixel=3*self.in_size*self.in_size)
        loss_fg0, metric_fg0 = self.loss_fg(inter_features[0], vgg_features[0], weight=self.loss_fg_weights[0])
        loss_fg1, metric_fg1 = self.loss_fg(inter_features[1], vgg_features[1], weight=self.loss_fg_weights[1])
        loss_fg2, metric_fg2 = self.loss_fg(inter_features[2], vgg_features[2], weight=self.loss_fg_weights[2])
        loss_fg3, metric_fg3 = self.loss_fg(inter_features[3], vgg_features[3], weight=self.loss_fg_weights[3])
        loss_fg = loss_fg0 + loss_fg1 + loss_fg2 + loss_fg3
        metric_fg = loss_fg / self.loss_fg.weight
        loss_val = loss_nll + loss_fg

        log_valid = {
            'val/metric/loss': loss_val,
            'val/metric/nll': metric_nll,
            'val/metric/fg': metric_fg,
            'val/metric/psnr_r': metric_psnr_r,
            'val/metric/ssim_r': metric_ssim_r,
            'val/metric/psnr_r_temp': metric_psnr_r_temp,
            'val/metric/ssim_r_temp': metric_ssim_r_temp,}
        self.log_dict(log_valid)

    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        # Log Qualative Result - Image
        grid = make_grid(self.sampled_images, nrow=6)
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
        trainable_parameters = [*self.flow_net.parameters(), *self.vgg_headers.parameters()]

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
        self.loss_fg = losses[opt['feature_guide']['type']](**opt['feature_guide']['args'])
        self.loss_fg_weights = [1.0, 0.5, 0.25, 0.125]
       
