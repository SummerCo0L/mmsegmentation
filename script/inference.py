import os
import os.path as osp
from os import listdir
import glob
import cv2 as cv
import collections
import time 
import tqdm
from PIL import Image
from functools import partial
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.compose import ColumnTransformer

import torchvision
import torchvision.transforms as transforms
import torch

from nibabel.testing import data_path
import nibabel as nib
from nibabel.viewers import OrthoSlicer3D

from .utility import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help = 'Path to model file')
    parser.add_argument('--input', help = 'Path to input')
    parser.add_argument('--output', help = 'Path to output')
    args = parser.parse_args()

    loaded_module = torch.jit.load(args.model)
    loaded_module.eval()

    from mmseg.datasets.pipelines import Compose
    from mmseg.apis.inference import LoadImage
    from mmcv.parallel import collate, scatter
    import torch.nn.functional as F
    device = next(loaded_module.parameters()).device
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    x = dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
    test_pipeline = [LoadImage()] + [x]
    test_pipeline = Compose(test_pipeline)
    
    out_path = args.output
    
    res = []
    start_time = time.time()
    
    for name in glob.glob(args.input + '/*.nii.gz'):
        ct_img = nib.load(name)
        data_arr = ct_img.get_fdata()
        data_arr = convert(data_arr)
        res = []
        for i in range(data_arr.shape[2]):
            img =  data_arr[:,:,i]
            cv.imwrite('tmp.png', img)
            data = dict(img='tmp.png')
            data = test_pipeline(data)
            data = collate([data], samples_per_gpu=1)
            data = scatter(data, [device])[0]
            with torch.no_grad() and torch.jit.optimized_execution(True):
                result = F.softmax(loaded_module.forward(data['img'][0].half()), dim=1).argmax(dim=1)
                result = result.cpu().numpy()
            res.append(result[0].astype('uint8'))
        res = np.array(res)
        res = res.transpose(1,2,0)
        ct_img = nib.Nifti1Image(res.copy(), ct_img.affine, ct_img.header) 
        filename = name.split('/')[-1]
        filename = '_'.join(filename.split('.')[0].split('_')[:2]) + filename[-7:]
        nib.save(ct_img, osp.join(out_path, filename))
    print('Time elapsed: ', time.time() - start_time)