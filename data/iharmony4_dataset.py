import os.path
import torch
import random
from PIL import Image
import numpy as np
import torchvision.transforms.functional as F
from torch.utils.data import Dataset


class Iharmony4Dataset(Dataset):
    """A template dataset class for you to implement custom datasets."""
    def __init__(self, dataset_root, is_for_train=True):
        """Initialize this dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        super().__init__()
        self.root = dataset_root
        self.isTrain = is_for_train
        self.image_paths, self.mask_paths, self.gt_paths = [], [], []
        self._load_images_paths()
        self.transform = HarmonyTransform()

    def _load_images_paths(self,):
        if self.isTrain:
            print('loading training file...')
            trainfile = os.path.join(self.root, 'train.txt')
            with open(trainfile,'r') as f:
                for line in f.readlines():
                    line = line.rstrip()
                    name_parts = line.split('_')
                    # data_parts = line.split('/')
                    mask_path = line.replace('composite_images', 'masks')
                    mask_path = mask_path.replace(('_'+name_parts[-1]),'.png')
                    
                    gt_path = line.replace('composite_images', 'real_images')
                    gt_path = gt_path.replace('_'+name_parts[-2]+'_'+name_parts[-1], '.jpg')
                    
                    self.image_paths.append(os.path.join(self.root, line))
                    self.mask_paths.append(os.path.join(self.root, mask_path))
                    self.gt_paths.append(os.path.join(self.root, gt_path))
        else:
            # if self.opt.accelerator.is_main_process:
            print('loading test file...')
            trainfile = os.path.join(self.root, 'test.txt')
            with open(trainfile,'r') as f:
                for line in f.readlines():
                    line = line.rstrip()
                    name_parts = line.split('_')
                    mask_path = line.replace('composite_images', 'masks')
                    mask_path = mask_path.replace(('_'+name_parts[-1]),'.png')
                    gt_path = line.replace('composite_images', 'real_images')
                    gt_path = gt_path.replace('_'+name_parts[-2]+'_'+name_parts[-1], '.jpg')
                    
                    self.image_paths.append(os.path.join(self.root, line))
                    self.mask_paths.append(os.path.join(self.root, mask_path))
                    self.gt_paths.append(os.path.join(self.root, gt_path))

    def __getitem__(self, index):
        comp = Image.open(self.image_paths[index]).convert('RGB')
        real = Image.open(self.gt_paths[index]).convert('RGB')
        mask = Image.open(self.mask_paths[index]).convert('1')
        
        #apply the same transform to composite and real images
        comp, real, mask = self.transform(comp, real, mask)
        
        return {'comp': comp, 'mask': mask, 'real': real, 'img_path':self.image_paths[index]}

    def __len__(self):
        """Return the total number of images."""
        return len(self.image_paths)


class HarmonyTransform():
    def __init__(
        self, 
        mean_img=[0.5, 0.5, 0.5],
        mean_mask=[0.5],
        std_img=[0.5, 0.5, 0.5],
        std_mask=[0.5],
        resize=None
    ):
        self.mean_img = mean_img
        self.mean_mask = mean_mask
        self.std_img = std_img
        self.std_mask = std_mask
        self.resize = resize

    def __call__(self, comp=None, real=None, mask=None):
        tf_comp = None
        tf_real = None
        tf_mask = None
        if comp is not None:
            if self.resize is not None:
                tf_comp = F.resize(comp, self.resize)
            else:
                tf_comp = comp
            tf_comp = F.to_tensor(comp)
            tf_comp = F.normalize(tf_comp, self.mean_img, self.std_img)
        if real is not None:
            if self.resize is not None:
                tf_real = F.resize(real, self.resize)
            else:
                tf_real = real
            tf_real = F.to_tensor(real)
            tf_real = F.normalize(tf_real, self.mean_img, self.std_img)
        if mask is not None:
            if self.resize is not None:
                tf_mask = F.resize(mask, self.resize)
            else: 
                tf_mask = mask
            tf_mask = F.to_tensor(mask)
            tf_mask = F.normalize(tf_mask, self.mean_mask, self.std_mask)

        tf_comp = tf_comp * tf_mask + tf_real * (1 - tf_mask)
        return tf_comp, tf_real, tf_mask