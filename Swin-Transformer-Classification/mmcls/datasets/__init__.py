# Copyright (c) OpenMMLab. All rights reserved.
from .base_dataset import BaseDataset
from .builder import (DATASETS, PIPELINES, SAMPLERS, build_dataloader,
                      build_dataset, build_sampler)
from .cifar import CIFAR10, CIFAR100
from .cub import CUB
from .custom import CustomDataset
from .dataset_wrappers import (ClassBalancedDataset, ConcatDataset,
                               KFoldDataset, RepeatDataset)
from .imagenet import ImageNet
from .imagenet21k import ImageNet21k
from .mnist import MNIST, FashionMNIST
from .multi_label import MultiLabelDataset
from .samplers import DistributedSampler, RepeatAugSampler
from .stanford_cars import StanfordCars
from .voc import VOC

from .caltech101_new import Caltech101
from .country211_new import Country211
from .dtd_new import DTD
from .eurosat_clip import EurosatClip
from .fer_new import Fer
from .fgvc_new import FGVC
from .gtsrb_new import Gtsrb

from .cifar10_new import CIFAR10N
from .cifar100_new import CIFAR100N
from .mnist_new import MNISTN
from .oxford_flower_new import OXFORDFLOWERN
from .oxford_pet_new import OXFORDPETN
from .stanford_cars_new import STANFORDCARSN
from .voc_new import VOCN

__all__ = [
    'BaseDataset', 'ImageNet', 'CIFAR10', 'CIFAR100', 'MNIST', 'FashionMNIST',
    'VOC', 'MultiLabelDataset', 'build_dataloader', 'build_dataset',
    'DistributedSampler', 'ConcatDataset', 'RepeatDataset',
    'ClassBalancedDataset', 'DATASETS', 'PIPELINES', 'ImageNet21k', 'SAMPLERS',
    'build_sampler', 'RepeatAugSampler', 'KFoldDataset', 'CUB',
    'CustomDataset', 'StanfordCars',

    'Caltech101', 'Country211', 'DTD', 'EurosatClip', 'Fer', 'FGVC', 'Gtsrb'
]
