# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import os

import cv2
import numpy as np
from PIL import Image

import torch
from torch.nn import functional as F

from .base_dataset import BaseDataset

class Cityscapes(BaseDataset):
    def __init__(self, 
                 root, 
                 list_path, 
                 num_samples=None, 
                 num_classes=19,
                 multi_scale=True, 
                 flip=True, 
                 ignore_label=-1, 
                 base_size=2048, 
                 crop_size=(512, 1024), 
                 downsample_rate=1,
                 scale_factor=16,
                 mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225]):

        super(Cityscapes, self).__init__(ignore_label, base_size,
                crop_size, downsample_rate, scale_factor, mean, std,)

        self.root = root
        self.list_path = list_path
        self.num_classes = num_classes

        self.multi_scale = multi_scale
        self.flip = flip
        
        self.img_list = [line.strip().split() for line in open(root+list_path)]

        self.files = self.read_files()
        if num_samples:
            self.files = self.files[:num_samples]

        self.label_mapping = {-1: ignore_label, 0: ignore_label, 
                              1: ignore_label, 2: ignore_label, 
                              3: ignore_label, 4: ignore_label, 
                              5: ignore_label, 6: ignore_label, 
                              7: 0, 8: 1, 9: ignore_label, 
                              10: ignore_label, 11: 2, 12: 3, 
                              13: 4, 14: ignore_label, 15: ignore_label, 
                              16: ignore_label, 17: 5, 18: ignore_label, 
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
                              25: 12, 26: 13, 27: 14, 28: 15, 
                              29: ignore_label, 30: ignore_label, 
                              31: 16, 32: 17, 33: 18}

        if torch.cuda.is_available():
            self.class_weights = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345,
                                            1.0166, 0.9969, 0.9754, 1.0489,
                                            0.8786, 1.0023, 0.9539, 0.9843,
                                            1.1116, 0.9037, 1.0865, 1.0955,
                                            1.0865, 1.1529, 1.0507]).cuda()
        else:
            self.class_weights = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345,
                                            1.0166, 0.9969, 0.9754, 1.0489,
                                            0.8786, 1.0023, 0.9539, 0.9843,
                                            1.1116, 0.9037, 1.0865, 1.0955,
                                            1.0865, 1.1529, 1.0507])
    
    def read_files(self):
        files = []
        if 'test' in self.list_path:
            for item in self.img_list:
                image_path = item
                name = os.path.splitext(os.path.basename(image_path[0]))[0]
                files.append({
                    "img": image_path[0],
                    "name": name,
                })
        else:
            for item in self.img_list:
                image_path, label_path = item
                name = os.path.splitext(os.path.basename(label_path))[0]
                files.append({
                    "img": image_path,
                    "label": label_path,
                    "name": name,
                    "weight": 1
                })
        return files
        
    def convert_label(self, label, inverse=False):
        temp = label.copy()
        if inverse:
            for v, k in self.label_mapping.items():
                label[temp == k] = v
        else:
            for k, v in self.label_mapping.items():
                label[temp == k] = v
        return label

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        # image = cv2.imread(os.path.join(self.root,'cityscapes',item["img"]),
        #                    cv2.IMREAD_COLOR)
        image = cv2.imread(os.path.join(self.root, item["img"]),
                           cv2.IMREAD_COLOR)
        size = image.shape

        if 'test' in self.list_path:
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))

            return image.copy(), np.array(size), name

        # label = cv2.imread(os.path.join(self.root,'cityscapes',item["label"]),
        #                    cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(os.path.join(self.root, item["label"]),
                           cv2.IMREAD_GRAYSCALE)
        label = self.convert_label(label)

        image, label = self.gen_sample(image, label, 
                                self.multi_scale, self.flip)

        return image.copy(), label.copy(), np.array(size), name

    def multi_scale_inference(self, config, model, image, scales=[1], flip=False):
        batch, _, ori_height, ori_width = image.size()
        assert batch == 1, "only supporting batchsize 1."
        image = image.numpy()[0].transpose((1,2,0)).copy()
        stride_h = int(self.crop_size[0] * 1.0)
        stride_w = int(self.crop_size[1] * 1.0)

        # add cpu compatible
        if torch.cuda.is_available():
            final_pred = torch.zeros([1, self.num_classes,
                                        ori_height,ori_width]).cuda()
        else:
            final_pred = torch.zeros([1, self.num_classes,
                                        ori_height,ori_width]).cpu()

        for scale in scales:
            new_img = self.multi_scale_aug(image=image,
                                           rand_scale=scale,
                                           rand_crop=False)
            height, width = new_img.shape[:-1]
                
            if scale <= 1.0:
                new_img = new_img.transpose((2, 0, 1))
                new_img = np.expand_dims(new_img, axis=0)
                new_img = torch.from_numpy(new_img)
                preds = self.inference(config, model, new_img, flip)
                preds = preds[:, :, 0:height, 0:width]
            else:
                new_h, new_w = new_img.shape[:-1]
                rows = int(np.ceil(1.0 * (new_h -
                            self.crop_size[0]) / stride_h)) + 1
                cols = int(np.ceil(1.0 * (new_w -
                            self.crop_size[1]) / stride_w)) + 1

                # add cpu compatible
                if torch.cuda.is_available():
                    preds = torch.zeros([1, self.num_classes,
                                               new_h,new_w]).cuda()
                    count = torch.zeros([1,1, new_h, new_w]).cuda()
                else:
                    preds = torch.zeros([1, self.num_classes,
                                         new_h,new_w]).cpu()
                    count = torch.zeros([1,1, new_h, new_w]).cpu()

                for r in range(rows):
                    for c in range(cols):
                        h0 = r * stride_h
                        w0 = c * stride_w
                        h1 = min(h0 + self.crop_size[0], new_h)
                        w1 = min(w0 + self.crop_size[1], new_w)
                        h0 = max(int(h1 - self.crop_size[0]), 0)
                        w0 = max(int(w1 - self.crop_size[1]), 0)
                        crop_img = new_img[h0:h1, w0:w1, :]
                        crop_img = crop_img.transpose((2, 0, 1))
                        crop_img = np.expand_dims(crop_img, axis=0)
                        crop_img = torch.from_numpy(crop_img)
                        pred = self.inference(config, model, crop_img, flip)
                        preds[:,:,h0:h1,w0:w1] += pred[:,:, 0:h1-h0, 0:w1-w0]
                        count[:,:,h0:h1,w0:w1] += 1
                preds = preds / count
                preds = preds[:,:,:height,:width]

            preds = F.interpolate(
                preds, (ori_height, ori_width), 
                mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
            )            
            final_pred += preds
        return final_pred

    def get_palette(self, n):
        palette = [0] * (n * 3)
        for j in range(0, n):
            lab = j
            palette[j * 3 + 0] = 0
            palette[j * 3 + 1] = 0
            palette[j * 3 + 2] = 0
            i = 0
            while lab:
                palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
                palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
                palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
                i += 1
                lab >>= 3
        return palette

    # add a function mapping the label to the color aligning with Cityscapes
    def convert_label_to_color(self, arr_2d):

        color_dict = {
            0:  (  0,   0,   0),
            1:  (  0,   0,   0),
            2:  (  0,   0,   0),
            3:  (  0,   0,   0),
            4:  (  0,   0,   0),
            5:  (111,  74,   0),
            6:  ( 81,   0,  81),
            7:  (128,  64, 128),
            8:  (244,  35, 232),
            9:  (250, 170, 160),
            10: (230, 150, 140),
            11: ( 70,  70,  70),
            12: (102, 102, 156),
            13: (190, 153, 153),
            14: (180, 165, 180),
            15: (150, 100, 100),
            16: (150, 120, 90),
            17: (153, 153, 153),
            18: (153, 153, 153),
            19: (250, 170,  30),
            20: (220, 220,   0),
            21: (107, 142,  35),
            22: (152, 251, 152),
            23: (70,  130, 180),
            24: (220,  20,  60),
            25: (255,   0,   0),
            26: (  0,   0, 142),
            27: (  0,   0,  70),
            28: (  0,  60, 100),
            29: (  0,   0,  90),
            30: (  0,   0, 110),
            31: (  0,  80, 100),
            32: (  0,   0, 230),
            33: (119,  11,  32),
            -1: (  0,   0, 142)
        }

        unique_ids = np.unique(arr_2d)

        # Map each unique ID to its color using the dictionary
        color_map = {id_: color_dict[id_] for id_ in unique_ids}

        # Use np.vectorize to create a 3D RGB array from the 2D array of IDs
        color_array = np.vectorize(color_map.get, otypes=[np.uint8, np.uint8, np.uint8])

        # Stack the resulting 3 separate RGB channels to get the final 3D array
        arr_3d = np.stack(color_array(arr_2d), axis=-1)

        return arr_3d

    def save_pred(self, preds, sv_path, name):
        palette = self.get_palette(256)
        preds = np.asarray(np.argmax(preds.cpu(), axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            pred = self.convert_label(preds[i], inverse=True)
            save_img = Image.fromarray(pred)
            # save_img.putpalette(palette)
            # save_img = Image.fromarray(self.convert_label_to_color(pred))

            save_img.save(os.path.join(sv_path, name[i]+'.png'))

        
        
