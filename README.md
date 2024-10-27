This project aims to implement the pretrained HRNet-OCR model to relieaze semantic segmentation for Google street view images. The original project comes from [github:HRNet/HRNet-Semantic-Segmentation](https://github.com/HRNet/HRNet-Semantic-Segmentation)

# 1. Package requirements

Besides packages listed in `requirements.txt`, also requires

```shell
conda install py-opencv matplotlib
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

# 2. Modified list

change `np.int()` into `int()`

add cpu compatibility, `torch.cuda.is_available()`


# 3. Model descriptions

Required pretrained model:

- Pretrained HRNet Image Classification model, [github](https://github.com/HRNet/HRNet-Image-Classification)

    - HRNet-W48-C: `hrnetv2_w48_imagenet_pretrained.pth`
    
    - configured by argument `MODEL.PRETRAINED` in `config` file
    
- Pretrained HRNet OCR model for Cityscapes:

    - HRNetV2-W48 + OCR: `hrnet_ocr_cs_trainval_8227_torch11.pth`

    - configured by argument `TEST.MODEL_FILE` in `config` file


## 2.1 Output categorise

| trainId      | id                                                                | name            |
| ------------ |-------------------------------------------------------------------| --------------- |
| ignore_label | \-1,<br/>0,1,2,3,4,5,6,<br/>9,10,<br/>14,15,16,<br/>18,<br/>29,30 |
| 0            | 7                                                                 | 'road'          |
| 1            | 8                                                                 | 'sidewalk'      |
| 2            | 11                                                                | 'building'      |
| 3            | 12                                                                | 'wall'          |
| 4            | 13                                                                | 'fence'         |
| 5            | 17                                                                | 'pole'          |
| 6            | 19                                                                | 'traffic light' |
| 7            | 20                                                                | 'traffic sign'  |
| 8            | 21                                                                | 'vegetation'    |
| 9            | 22                                                                | 'terrain'       |
| 10           | 23                                                                | 'sky'           |
| 11           | 24                                                                | 'person'        |
| 12           | 25                                                                | 'rider'         |
| 13           | 26                                                                | 'car'           |
| 14           | 27                                                                | 'truck'         |
| 15           | 28                                                                | 'bus'           |
| 16           | 31                                                                | 'train'         |
| 17           | 32                                                                | 'motorcycle'    |
| 18           | 33                                                                | 'bicycle'       |