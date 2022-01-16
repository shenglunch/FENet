import os
import random
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from datasets.data_io import get_transform, read_all_lines
bf = {} # pixel*m
bf['2018-07-11-14-48-52'] = (2063.200+2062.400)/2 * 0.5446076
bf['2018-08-01-11-13-14'] = (2069.500+2068.300)/2 * 0.5449333
bf['2018-08-07-13-46-08'] = (2069.500+2068.300)/2 * 0.5449333
bf['2018-10-11-16-03-19'] = (2061.940+2060.674)/2 * 0.5449150

class DrivingStereoDataset(Dataset):
    def __init__(self, datapath, list_filename, training):
        self.datapath = datapath
        self.left_filenames, self.right_filenames, self.disp_filenames, self.depth_filenames = self.load_path(list_filename)
        self.training = training
        if self.training:
            assert self.disp_filenames is not None

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        left_images = [x[0] for x in splits]
        right_images = [x[1] for x in splits]
        disp_images = [x[2] for x in splits]
        depth_images = [x[3] for x in splits]
        return left_images, right_images, disp_images, depth_images

    def load_image(self, filename):
        return Image.open(filename).convert('RGB')

    def load_disp(self, filename, training):
        data = Image.open(filename)
        if training:
            data = np.array(data, dtype=np.float32) / 256.
        else:
            data = np.array(data, dtype=np.float32) / 128.
        return data
   
    def load_depth(self, filename):
        data = Image.open(filename)
        data = np.array(data, dtype=np.float32) / 256.
        return data

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):
        left_img = self.load_image(os.path.join(self.datapath, self.left_filenames[index]))
        right_img = self.load_image(os.path.join(self.datapath, self.right_filenames[index]))
        disparity = self.load_disp(os.path.join(self.datapath, self.disp_filenames[index]), self.training)
               
        if self.training:
            w, h = left_img.size
            crop_w, crop_h = 512, 240 # 512, 256

            x1 = random.randint(0, w - crop_w)
            y1 = random.randint(0, h - crop_h)

            # random crop
            left_img = left_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            right_img = right_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            disparity = disparity[y1:y1 + crop_h, x1:x1 + crop_w]

            # to tensor, normalize
            processed = get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)

            return {"left": left_img,
                    "right": right_img,
                    "disparity": disparity}
        else:
            depth = self.load_depth(os.path.join(self.datapath, self.depth_filenames[index]))
            seq = self.depth_filenames[index].split('/')[2]

            w, h = left_img.size

            # normalize
            processed = get_transform()
            left_img = processed(left_img).numpy()
            right_img = processed(right_img).numpy()

            # pad to size 1776*800
            top_pad = 800 - h
            right_pad = 1776 - w
            # assert top_pad > 0 and right_pad > 0
            # pad images
            left_img = np.lib.pad(left_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
            right_img = np.lib.pad(right_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
            # pad disparity gt
            assert len(disparity.shape) == 2
            disparity = np.lib.pad(disparity, ((top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
            depth = np.lib.pad(depth, ((top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)

            return {"left": left_img,
                    "right": right_img,
                    "disparity": disparity,
                    "depth": depth, 
                    "bf": bf[seq],
                    "top_pad": top_pad,
                    "right_pad": right_pad}
