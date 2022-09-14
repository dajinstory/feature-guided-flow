import torch
from importlib import import_module

from .kdflow import LitKDFlowV0, LitKDFlowV1


def build_model(opt, is_train=True):

    models={
        'LitKDFlowV0': LitKDFlowV0,
        'LitKDFlowV1': LitKDFlowV1,
    }

    try: 
        model_type = opt['type']
        if opt['pretrained']:
            print("Load Checkpoint from ", opt['pretrained'])
            model = models[model_type].load_from_checkpoint(opt['pretrained'], pretrained=True, strict=False)
        else:
            model = models[model_type](opt)
    except:
        raise ValueError(f'Model [{model_type}] is not supported')
        
    return model
