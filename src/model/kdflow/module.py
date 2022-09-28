import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

from ..facial_recognition.model_irse import Backbone as Backbone_ID_Loss

def get_header1(ch_in, ch_hidden, ch_out, kernel=3):
    pad = kernel // 2
    header = nn.Sequential(
        nn.Conv2d(ch_in, ch_hidden, kernel, padding=pad),
        nn.ReLU(),
        nn.Conv2d(ch_hidden, ch_hidden, kernel, padding=pad),
        nn.ReLU(),
        nn.Conv2d(ch_hidden, ch_out, kernel, padding=pad),
    )
    return header

def get_header2(ch_in, ch_hidden, ch_out, kernel=3):
    pad = kernel // 2
    header = nn.Sequential(
        nn.Conv2d(ch_in, ch_hidden, kernel, padding=pad),
        nn.ReLU(),
        nn.Conv2d(ch_hidden, ch_hidden, kernel, padding=pad),
        nn.ReLU(),
        nn.Conv2d(ch_hidden, ch_hidden, kernel, padding=pad),
        nn.ReLU(),
        nn.Conv2d(ch_hidden, ch_hidden, kernel, padding=pad),
        nn.ReLU(),
        nn.Conv2d(ch_hidden, ch_out, kernel, padding=pad),
    )
    return header

    
class VGG16ModuleV0(nn.Module):
    def __init__(self, pretrained=None):
        super(VGG16ModuleV0, self).__init__()
            
        # INPUT : (,3,32,32), resized to /2
        self.blocks = nn.Sequential(                                                 
            torchvision.models.vgg16(pretrained=True).features[:4].eval(),      # 64,H,W 
            torchvision.models.vgg16(pretrained=True).features[4:9].eval(),     # 128,/2,/2
            torchvision.models.vgg16(pretrained=True).features[9:16].eval(),    # 256,/4,/4
            torchvision.models.vgg16(pretrained=True).features[16:23].eval()   # 512,/8,/8
        )
        self.headers = nn.Sequential(
            get_header2(6,32,64,3),
            get_header2(12,64,128,3),
            get_header2(24,128,256,3),
            get_header2(96,256,512,3),            
        )
        self.require_resize = True
        self.norm_mean = [0.485, 0.456, 0.406]
        self.norm_std = [0.229, 0.224, 0.225]
        self.preprocess = transforms.Normalize(
            mean=self.norm_mean, 
            std=self.norm_std)
    
    def forward(self):
        pass


class VGG16ModuleV1(nn.Module):
    def __init__(self, pretrained=None):
        super(VGG16ModuleV1, self).__init__()
            
        # INPUT : (,3,32,32), resized to /2
        self.blocks = nn.Sequential(                                                 
            torchvision.models.vgg16(pretrained=True).features[:4].eval(),      # 64,H,W 
            torchvision.models.vgg16(pretrained=True).features[4:9].eval(),     # 128,/2,/2
            torchvision.models.vgg16(pretrained=True).features[9:16].eval(),    # 256,/4,/4
            torchvision.models.vgg16(pretrained=True).features[16:23].eval()   # 512,/8,/8
        )
        self.headers = nn.Sequential(
            get_header2(6,32,64,3),
            get_header2(12,64,128,3),
            get_header2(24,128,256,3),
            get_header2(48,256,512,3),            
        )
        self.require_resize = True
        self.norm_mean = [0.485, 0.456, 0.406]
        self.norm_std = [0.229, 0.224, 0.225]
        self.preprocess = transforms.Normalize(
            mean=self.norm_mean, 
            std=self.norm_std)
    
    def forward(self):
        pass

class InsightFaceModuleV0(nn.Module):
    def __init__(self, pretrained):
        super(InsightFaceModuleV0, self).__init__()

        facenet = Backbone_ID_Loss(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')  
        facenet.load_state_dict(torch.load(pretrained))  
        self.blocks = nn.Sequential(
            nn.Sequential(
                facenet.input_layer,
                facenet.body[:3]),      # 64,/2,/2
            facenet.body[3:7],          # 128,/4,/4
            facenet.body[7:21],         # 256,8,8
            facenet.body[21:],        # 512,4,4
        )
        self.headers = nn.Sequential(
            get_header2(6,32,64,3),
            get_header2(12,64,128,3),
            get_header2(24,128,256,3),
            get_header2(96,256,512,3),            
        )
        self.require_resize = False
        self.norm_mean = [0.5,0.5,0.5]
        self.norm_std = [0.5,0.5,0.5]
        self.preprocess = transforms.Normalize(
            mean=self.norm_mean, 
            std=self.norm_std)
            
    def forward(self):
        pass

class InsightFaceModuleV1(nn.Module):
    def __init__(self, pretrained):
        super(InsightFaceModuleV1, self).__init__()

        facenet = Backbone_ID_Loss(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')  
        facenet.load_state_dict(torch.load(pretrained))  
        self.blocks = nn.Sequential(
            nn.Sequential(
                facenet.input_layer,
                facenet.body[:3]),      # 64,/2,/2
            facenet.body[3:7],          # 128,/4,/4
            facenet.body[7:21],         # 256,8,8
            facenet.body[21:],        # 512,4,4
        )
        self.headers = nn.Sequential(
            get_header2(6,32,64,3),
            get_header2(12,64,128,3),
            get_header2(24,128,256,3),
            get_header2(48,256,512,3),            
        )
        self.require_resize = False
        self.norm_mean = [0.5,0.5,0.5]
        self.norm_std = [0.5,0.5,0.5]
        self.preprocess = transforms.Normalize(
            mean=self.norm_mean, 
            std=self.norm_std)
            
    def forward(self):
        pass