# Copyright (c) OpenMMLab. All rights reserved.
from torch import nn


MODELS.register_module("Deconv", module=nn.ConvTranspose2d)
# TODO: octave conv
