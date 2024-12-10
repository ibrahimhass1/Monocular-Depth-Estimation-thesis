import numpy as np
import math

import torch

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_

from einops import rearrange
from functools import partial
from torch import nn, einsum
from torch.nn.modules.batchnorm import _BatchNorm

from mmcv.cnn import build_norm_layer

from mmseg.utils import get_root_logger
from mmseg.models.builder import BACKBONES