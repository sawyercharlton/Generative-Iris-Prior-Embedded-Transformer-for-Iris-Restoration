# flake8: noqa
import os.path as osp
from basicsr.test import test_pipeline

import src.archs
import src.data
import src.models

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)
