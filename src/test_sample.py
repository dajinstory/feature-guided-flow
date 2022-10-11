# Library
import os
import argparse
import torch
import torchvision.transforms as T
from torchvision.utils import make_grid
from PIL import Image
from model.fgflow import Glow64x64V0, LitFGFlowV0

ptt = T.ToTensor()
ttp = T.ToPILImage()


# Parse
parser = argparse.ArgumentParser(description='model_version')
parser.add_argument('--model', type=str, default='baseline')
args = parser.parse_args()


# Model
def load_glow_baseline(model='glow'):
    if model=='lit_glow':
        lit_ckpt_path = '/data/dajinhan/experiment/fgflow_v0_baseline/checkpoint/last.ckpt'
        ckpt_path = '/home/dajinhan/nas_dajinhan/experiments/fgflow/result/glow_baseline.ckpt'

        lit_net = LitFGFlowV0.load_from_checkpoint(lit_ckpt_path, pretrained=True, strict=False)
        torch.save(lit_net.flow_net.state_dict(), ckpt_path)
        return lit_net.flow_net.eval()
    else:
        # ckpt_path = '/home/dajinhan/nas_dajinhan/experiments/fgflow/result/glow_baseline.ckpt'
        ckpt_path = '/home/dajinhan/nas_dajinhan/experiments/glow/result/glow_64x64_celeba.ckpt'
        pretrained = {'ckpt_path': ckpt_path}
        net = Glow64x64V0(pretrained)
        return net.eval()

def load_glow_intertemp(model='glow'):
    if model=='lit_glow':
        lit_ckpt_path = '/data/dajinhan/experiment/fgflow_v0_intertemp/checkpoint/last.ckpt'
        ckpt_path = '/home/dajinhan/nas_dajinhan/experiments/fgflow/result/glow_intertemp.ckpt'

        lit_net = LitFGFlowV0.load_from_checkpoint(lit_ckpt_path, pretrained=True, strict=False)
        torch.save(lit_net.flow_net.state_dict(), ckpt_path)
        return lit_net.flow_net.eval()
    else:
        ckpt_path = '/home/dajinhan/nas_dajinhan/experiments/fgflow/result/glow_intertemp.ckpt'
        pretrained = {'ckpt_path': ckpt_path}
        net = Glow64x64V0(pretrained)
        return net.eval()

def load_glow_recon(model='glow'):
    if model=='lit_glow':
        lit_ckpt_path = '/data/dajinhan/experiment/fgflow_v0_recon/checkpoint/last.ckpt'
        ckpt_path = '/home/dajinhan/nas_dajinhan/experiments/fgflow/result/glow_recon.ckpt'

        lit_net = LitFGFlowV0.load_from_checkpoint(lit_ckpt_path, pretrained=True, strict=False)
        torch.save(lit_net.flow_net.state_dict(), ckpt_path)
        return lit_net.flow_net.eval()
    else:
        ckpt_path = '/home/dajinhan/nas_dajinhan/experiments/fgflow/result/glow_recon.ckpt'
        pretrained = {'ckpt_path': ckpt_path}
        net = Glow64x64V0(pretrained)
        return net.eval()

def load_glow_featureguidance(model='glow'):
    if model=='lit_glow':
        lit_ckpt_path = '/data/dajinhan/experiment/fgflow_v0_featureguidance/checkpoint/last.ckpt'
        ckpt_path = '/home/dajinhan/nas_dajinhan/experiments/fgflow/result/glow_featureguidance.ckpt'

        lit_net = LitFGFlowV0.load_from_checkpoint(lit_ckpt_path, pretrained=True, strict=False)
        torch.save(lit_net.flow_net.state_dict(), ckpt_path)
        return lit_net.flow_net.eval()
    else:
        ckpt_path = '/home/dajinhan/nas_dajinhan/experiments/fgflow/result/glow_featureguidance.ckpt'
        pretrained = {'ckpt_path': ckpt_path}
        net = Glow64x64V0(pretrained)
        return net.eval()

def load_glow_fg_recon(model='glow'):
    if model=='lit_glow':
        lit_ckpt_path = '/data/dajinhan/experiment/fgflow_v0_fg_recon/checkpoint/last.ckpt'
        ckpt_path = '/home/dajinhan/nas_dajinhan/experiments/fgflow/result/glow_fg_recon.ckpt'

        lit_net = LitFGFlowV0.load_from_checkpoint(lit_ckpt_path, pretrained=True, strict=False)
        torch.save(lit_net.flow_net.state_dict(), ckpt_path)
        return lit_net.flow_net.eval()
    else:
        ckpt_path = '/home/dajinhan/nas_dajinhan/experiments/fgflow/result/glow_fg_recon.ckpt'
        pretrained = {'ckpt_path': ckpt_path}
        net = Glow64x64V0(pretrained)
        return net.eval()

load_models = {
    'baseline': load_glow_baseline,
    'intertemp': load_glow_intertemp,
    'recon': load_glow_recon,
    'featureguidance': load_glow_featureguidance,
    'fg_recon': load_glow_fg_recon,
}

# Norm
norm_mean = [0.5, 0.5, 0.5]
norm_std = [1.0, 1.0, 1.0]
preprocess = T.Normalize(
    mean=norm_mean, 
    std=norm_std)
reverse_preprocess = T.Normalize(
    mean=[-m/s for m,s in zip(norm_mean, norm_std)],
    std=[1/s for s in norm_std])


# Sample
def sample_64(net, n_samples=1, final_temp=1.0, inter_temp=0.7):
    z_final = final_temp * torch.randn((n_samples,96,4,4)).cuda() * net.final_temp
    z_splits = [
        inter_temp * torch.randn((n_samples,6,32,32)).cuda() * net.inter_temp,
        inter_temp * torch.randn((n_samples,12,16,16)).cuda() * net.inter_temp,
        inter_temp * torch.randn((n_samples,24,8,8)).cuda() * net.inter_temp,
        None,
    ]
    conditions = [None] * len(net.blocks)
    x_sample = net.reverse(z_final, conditions, z_splits)
    x_sample = reverse_preprocess(x_sample)
    x_sample = torch.clamp(x_sample, 0,1)
    return x_sample

def sample_256(net, n_samples=1, temp=0.7):
    z_final = temp * torch.randn((n_samples,384,4,4)) * net.final_temp
    z_splits = [
        temp * torch.randn((n_samples,6,128,128)) * net.inter_temp,
        temp * torch.randn((n_samples,12,64,64)) * net.inter_temp,
        temp * torch.randn((n_samples,24,32,32)) * net.inter_temp,
        temp * torch.randn((n_samples,48,16,16)) * net.inter_temp,
        temp * torch.randn((n_samples,96,8,8)) * net.inter_temp,
        None,
    ]
    conditions = [None] * len(net.blocks)
    x_sample = net.reverse(z_final, conditions, z_splits)
    x_sample = reverse_preprocess(x_sample)
    x_sample = torch.clamp(x_sample, 0,1)
    return x_sample

def sample_grid(x, n_row):
    imgs = [img.cpu() for img in x]
    grid = make_grid(imgs, n_row)
    return ttp(grid)


# Sample Variations
def x_to_z(net, x):
    x = preprocess(x)
    conditions = [None] * len(net.blocks)
    z_final, _, _, z_splits = net.forward(x, conditions)
    return z_final, z_splits

def z_to_x(net, z_final, z_splits):
    conditions = [None] * len(net.blocks)
    x = net.reverse(z_final, conditions, z_splits)
    x = reverse_preprocess(x)
    return x

def sample_temperature(net, z_final, z_splits):
    temperatures = [0, 0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    def adjust_temperature(z):
        return torch.cat([torch.stack([z_i * temp for temp in temperatures], dim=0) for z_i in z], dim=0)
    z_final = adjust_temperature(z_final)
    z_splits = [adjust_temperature(z_split) if z_split is not None else None for z_split in z_splits]
    
    conditions = [None] * len(net.blocks)
    x_sample = net.reverse(z_final, conditions, z_splits)
    x_sample = reverse_preprocess(x_sample)
    x_sample = torch.clamp(x_sample, 0,1)
    return x_sample

def sample_interpolate(net, z1_, z2_):
    temperatures = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    z1_final, z1_splits = z1_
    z2_final, z2_splits = z2_
    def adjust_interpolate(z1, z2):
        return torch.cat([ torch.stack([z1_i*temp + z2_i*(1-temp) for temp in temperatures], dim=0) for z1_i, z2_i in zip(z1,z2)], dim=0)
    z_final = adjust_interpolate(z1_final, z2_final)
    z_splits = [adjust_interpolate(z1_split, z2_split) if z1_split is not None else None for z1_split, z2_split in zip(z1_splits, z2_splits)]

    conditions = [None] * len(net.blocks)
    x_sample = net.reverse(z_final, conditions, z_splits)
    x_sample = reverse_preprocess(x_sample)
    x_sample = torch.clamp(x_sample, 0,1)
    return x_sample
    

# Load Model
net = load_models[args.model]('glow')
net = net.cuda()

root_path = os.path.join('/data/dajinhan/outputs', args.model)
os.makedirs(root_path, exist_ok=True)


# Sample outputs
n_trains = 30000
n_gens = 0
with torch.no_grad():
    while n_gens < n_trains:
        x_samples = sample_64(net, n_samples=1024, final_temp=1.0, inter_temp=0.7)
        for x_sample in x_samples:
            if n_gens < n_trains:
                n_gens += 1
                im = ttp(x_sample.cpu().detach())
                im.save(os.path.join(root_path, '%.5d.png'%(n_gens)))
            else:
                break
    




