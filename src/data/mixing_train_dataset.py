import cv2
import math
import numpy as np
import os.path as osp
import torch
import torch.utils.data as data
from basicsr.data import degradations as degradations
from basicsr.data.data_util import paths_from_folder
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from torchvision.transforms.functional import (adjust_brightness, adjust_contrast, adjust_hue, adjust_saturation,
                                               normalize)
import imgaug.augmenters as iaa
from PIL import Image
from torchvision.transforms import transforms


@DATASET_REGISTRY.register()
class MixingDegradationDataset(data.Dataset):
    """FFHQ dataset for GFPGAN.

    It reads high resolution images, and then generate low-quality (LQ) images on-the-fly.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            io_backend (dict): IO backend type and other kwarg.
            mean (list | tuple): Image mean.
            std (list | tuple): Image std.
            use_hflip (bool): Whether to horizontally flip.
            Please see more options in the codes.
    """

    def __init__(self, opt):
        super(MixingDegradationDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        self.gt_folder = opt['dataroot_gt']
        self.mean = opt['mean']
        self.std = opt['std']
        self.out_size = opt['out_size']

        # file client (lmdb io backend)
        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = self.gt_folder
            if not self.gt_folder.endswith('.lmdb'):
                raise ValueError(f"'dataroot_gt' should end with '.lmdb', but received {self.gt_folder}")
            with open(osp.join(self.gt_folder, 'meta_info.txt')) as fin:
                self.paths = [line.split('.')[0] for line in fin]
        else:
            # disk backend: scan file list from a folder
            self.paths = paths_from_folder(self.gt_folder)

        # degradation configurations
        self.to_tensor = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])

    def __getitem__(self, index):
        # load gt image
        gt_path = self.paths[index]
        img_gt = Image.open(gt_path).convert('RGB') # image range: [0, 255]

        # ------------------------ generate lq image ------------------------ #
        img_lq = complex_imgaug(img_gt)

        # numpy to tensor
        img_gt = self.to_tensor(img_gt)
        img_lq = self.to_tensor(img_lq)

        return {'lq': img_lq, 'gt': img_gt, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)

def complex_imgaug(x):
    """input single RGB PIL Image instance.
    From PSFRGAN.
    """
    x = np.array(x)
    x = x[np.newaxis, :, :, :]
    aug_seq = iaa.Sequential([
        iaa.Sometimes(0.5, iaa.MotionBlur((3, 15))),
        iaa.Sometimes(0.5, iaa.GaussianBlur((3, 15))),
        iaa.Sometimes(1, iaa.Resize((0.25, 1), interpolation="cubic")), # downsample
        iaa.Sometimes(1, iaa.Resize({"height": 256, "width": 256}, interpolation="cubic")) # resize to original size
    ])
    aug_img = aug_seq(images=x)

    return aug_img[0]
