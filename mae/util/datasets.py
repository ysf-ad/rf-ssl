# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL
import torch

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)

    print(dataset)

    return dataset


def build_transform(is_train, args):
    # Dataset is already normalized
    # mean = IMAGENET_DEFAULT_MEAN
    # std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),  # Convert single-channel to 3-channel
            # transforms.Resize(args.input_size, interpolation=transforms.InterpolationMode.BICUBIC),
            # transforms.CenterCrop(args.input_size),
            transforms.ToTensor(),
        ])
        return transform


    # eval transform
    t = [
        transforms.Resize(args.input_size, interpolation=PIL.Image.BICUBIC),
        transforms.CenterCrop(args.input_size),
        transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels
        transforms.ToTensor()
    ]
    return transforms.Compose(t)

