import os
import random
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms
from datasets.data_io import get_transform, read_all_lines, readPFM

class EMDataset(Dataset):
    def __init__(self, datapath, list_filename, training):
        self.datapath = datapath
        self.left_filenames, self.right_filenames, self.disp_filenames = self.load_path(list_filename)
        self.training = training
       
        self.brightness = (0.8, 1.2)
        self.contrast = (0.8, 1.2)
        self.saturation = (0.8, 1.2)
        self.hue = (-0.1, 0.1)

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        left_images = [x[0] for x in splits]
        right_images = [x[1] for x in splits]
        disp_images = [x[2] for x in splits]
        return left_images, right_images, disp_images

    def load_image(self, filename):
        return Image.open(filename).convert('RGB')

    def load_disp(self, filename):
        if '.png' in filename:
            data = Image.open(filename)
            data = np.array(data, dtype=np.float32) / 256.
        else:
            data, scale = readPFM(filename)
            data = np.ascontiguousarray(data, dtype=np.float32)
        return data

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):
        if '2015' in self.left_filenames[index] or '2012' in self.left_filenames[index]:
            left_img = self.load_image(os.path.join('/data/yyx/data/kitti', self.left_filenames[index]))
            right_img = self.load_image(os.path.join('/data/yyx/data/kitti', self.right_filenames[index]))
            disparity = self.load_disp(os.path.join('/data/yyx/data/kitti', self.disp_filenames[index]))
        else:
            left_img = self.load_image(os.path.join(self.datapath, self.left_filenames[index]))
            right_img = self.load_image(os.path.join(self.datapath, self.right_filenames[index]))
            disparity = self.load_disp(os.path.join(self.datapath, self.disp_filenames[index]))

        if self.training:
            do_color_aug_l = random.random() > 0.5
            do_color_aug_r = random.random() > 0.5
            do_flip = random.random() > 0.5
            if do_flip:
                left_img = left_img.transpose(Image.FLIP_TOP_BOTTOM)
                right_img = right_img.transpose(Image.FLIP_TOP_BOTTOM)
                disparity = np.flipud(disparity).copy() # c*h*w
            
            if do_color_aug_l:
                color_aug = transforms.ColorJitter.get_params(
                    self.brightness, self.contrast, self.saturation, self.hue)
                left_img = color_aug(left_img)
            if do_color_aug_r:
                color_aug = transforms.ColorJitter.get_params(
                    self.brightness, self.contrast, self.saturation, self.hue)
                right_img = color_aug(right_img)

            w, h = left_img.size
            crop_w, crop_h = 512, 240#

            x1 = random.randint(0, w - crop_w)
            y1 = random.randint(0, h - crop_h)

            # random crop
            left_img = left_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            right_img = right_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            disparity = disparity[y1:y1 + crop_h, x1:x1 + crop_w]
            disparity[disparity == np.inf] = 0

            # to tensor, normalize
            processed = get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)
            disparity = np.ascontiguousarray(disparity, dtype=np.float32)

            return {"left": left_img,
                    "right": right_img,
                    "disparity": disparity}
        else:
            w, h = left_img.size

            # normalize
            processed = get_transform()
            left_img = processed(left_img).numpy()
            right_img = processed(right_img).numpy()

            # pad to size 1248x384
            if h % 16 == 0:
                top_pad = 0
            else:
                top_pad = 16 - (h % 16)

            if w % 16 == 0:
                right_pad = 0
            else:
                right_pad = 16 - (w % 16)
            assert top_pad >= 0 and right_pad >= 0
            # pad images
            left_img = np.lib.pad(left_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
            right_img = np.lib.pad(right_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant',
                                   constant_values=0)
            # pad disparity gt
            if disparity is not None:
                assert len(disparity.shape) == 2
                disparity[disparity == np.inf] = 0
                disparity = np.lib.pad(disparity, ((top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
                disparity = np.ascontiguousarray(disparity, dtype=np.float32)

            if disparity is not None:
                
                return {"left": left_img,
                        "right": right_img,
                        "disparity": disparity,
                        "top_pad": top_pad,
                        "right_pad": right_pad}
            else:
                return {"left": left_img,
                        "right": right_img,
                        "top_pad": top_pad,
                        "right_pad": right_pad,
                        "left_filename": self.left[index],
                        "right_filename": self.right[index]}
