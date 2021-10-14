<div align="center">
  <img src="resources/mmseg-logo.png" width="600"/>
</div>
<br />

# Introduction

This is our setup in the challenge FLARE2021. This work is inherited directly from the [MMSegmentation Repo](https://github.com/open-mmlab/mmsegmentation)

We exploit the robustness of HRNet + OCRNet in our pipeline. The slices of 3D image are splitted into 2D images and then they are fed to the trained model. Also, keeping in mind that the running speed and memory cost also matter, we convert the model into Torch Script format.

# Intalling
```
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install nibabel
pip install Pillow
pip install -U scikit-learn
pip install -e .
```

or pulling from DockerHub

```
docker pull quoccuongcs/uit_vnu
```



# Inference

Firstly, you need to download our [pretrained model](https://drive.google.com/file/d/1bAU4YvkViXv6rCUXwrMV4ydbbpaOWiih/view?usp=sharing) from Google Drive and put them anywhere.

In order to reproduce the result, run the following

```
python script/inference.py --model MODEL_PATH --input INPUT --output OUTPUT 
```

You will need to adapt the corresponding path to your TorchScript model, your folder that contains .nii files and folder that are supposed to contain the .nii output.

# Train

## Prepare data

Suppose you have a dataset whose format are .nii files, you need to convert them to 2D images. We have provided our structured FLARE2021 dataset [here](https://drive.google.com/file/d/1-mQ_FOzutCb2HK3GJm39Grfq_BAOUKb5/view?usp=sharing). 

<!-- NII_PATH and PNG_PATH represent for the folder contain .nii inputs and .png outputs, respectively.

```
python script/prepare_data.py --nii_path NII_PATH --png_path PNG_PATH 
```

Now the 2D image's path should have the following format `PNG_PATH/{SEQUENCE}/{FRAME}.png`. This step should be applied to both the image folder and label mask folder. After that, 2 text files named `train.txt` and `val.txt` need to be created. These files contains the images used for training and validation.  -->

After downloading the zip file above, as well as unzipping it into `data` folder, you should have something like following:

```
data
│   train.txt
│   val.txt    
│
└───separated_img
│   │   001_0000.png
│   │   ...
│   
└───separated_mask
│   │   001_0000.png
│   │   ...
│
│___ ...
```

## Run training
You also need to modify the `script/config.py` as follows:
- Line 43 represents image's path
- Line 44 represents label mask's path
- Line 77 represents data folder's path
- Line 141 represents path where the model will be saved.
The current file has been hard-coded for your convenience. 

Finally, you can run the training script:
```
python script/train.py 
```

To convert trained models to Torch Script format, use the following command with the corresponding paths
```
python script/create_torchscript.py --in_model INPUT_MODEL --out_model OUTPUT_MODEL
```

In the command above, `INPUT_MODEL` is the path to .pth file and `OUTPUT_MODEL` is the path to .pt file. For example: `python script/create_torchscript.py --in_model ./weight/latest.pth --out_model ./weight/latest.pt`