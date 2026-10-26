"""Use CPU EDT and upstream non-perflib paths for single-image inference.

The neural model and its weights are unmodified. Triton's EDT import is
otherwise eager, even when the interactive/video tracker is not requested.
"""
import os
import sys
import types

os.environ['USE_PERFLIB']='0'

def edt(data):
    import cv2
    import numpy as np
    import torch
    masks=data.detach().cpu().numpy().astype('uint8')
    distances=np.stack([cv2.distanceTransform(mask,cv2.DIST_L2,cv2.DIST_MASK_PRECISE) for mask in masks])
    return torch.as_tensor(distances,device=data.device,dtype=torch.float32)

module=types.ModuleType('sam3.model.edt')
module.edt_triton=edt
sys.modules['sam3.model.edt']=module
