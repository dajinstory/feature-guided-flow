# import torch
# from importlib import import_module

# from .fgflow import LitGlowV0, LitFGFlowV0, LitFGFlowV1


# def build_model(opt, is_train=True):

#     models={
#         'LitGlowV0': LitGlowV0,
#         'LitFGFlowV0': LitFGFlowV0,
#         'LitFGFlowV1': LitFGFlowV1,
#     }

#     try: 
#         model_type = opt['type']
#         if opt['pretrained']:
#             print("Load Checkpoint from ", opt['pretrained']['ckpt_path'])
#             model = models[model_type].load_from_checkpoint(opt['pretrained']['ckpt_path'], pretrained=True, strict=False)
#         else:
#             model = models[model_type](opt)
#     except:
#         raise ValueError(f'Model [{model_type}] is not supported')
        
#     return model
