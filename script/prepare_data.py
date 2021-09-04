# !mkdir /content/drive/MyDrive/Contest_Folder/FLARE2021/dataset/AbdomenCT-1K_950-50/Training/SeparatedMask
import cv2 as cv
from os import listdir
import os.path as osp
import os
from tqdm import tqdm
import glob
import argparse

from nibabel.testing import data_path
import nibabel as nib
from nibabel.viewers import OrthoSlicer3D

from script.utility import *

def make_name(x):
  while len(x) < 4:
    x = '0' + x
  return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nii_path', type=str)
    parser.add_argument('--png_path', type=str)
    args = parser.parse_args()

    raw_path = args.nii_path + '/*.nii.gz'
    nii_list = glob.glob(raw_path)
    nii_list.sort()
    outpath = args.png_path
    
    for i, x in tqdm(enumerate(nii_list),total = len(nii_list)):
        case = x.split('/')[-1].split('.')[0].split('_')[1]
        case_dir = osp.join(outpath,case)
        if not osp.exists(case_dir):
            os.makedirs(case_dir)
        x = nib.load(x).get_fdata()
        for idx in range(x.shape[2]):
            outname = osp.join(case_dir, '{}_{}.png'.format(case,make_name(str(idx))))
            cv.imwrite(outname,convert(x[:,:,idx]))