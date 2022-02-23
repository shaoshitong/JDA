# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------'

import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']
_C.EPOCH = 10000
# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Path to dataset, could be overwritten by command line argument
_C.DATA.DEAP_DATA_PATH = '/data/EEG/Channel_DE_sample/Binary/'
_C.DATA.SEED_DATA_PATH = '/data/EEG/DE_4D/'
# _C.DATA.SEED_DATA_PATH = 'G:/Alex/SEED_experiment/Three sessions sample/Channel DE/'
# number of classes
# Dataset name
_C.DATA.DATASET = 'deap'
# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.PRETRAINED = ''
_C.MODEL.RESUME = ''
_C.MODEL.IF_TURN_LR=True

_C.MODEL.DEAP = CN()
_C.MODEL.DEAP.BATCH_SIZE=48
_C.MODEL.DEAP.OUT=128
_C.MODEL.DEAP.TIME_DIM=120
_C.MODEL.DEAP.NUM_CLASSES=2
_C.MODEL.SEED = CN()
_C.MODEL.SEED.BATCH_SIZE=48
_C.MODEL.SEED.OUT=310
_C.MODEL.SEED.TIME_DIM=180
_C.MODEL.SEED.NUM_CLASSES=3

def get_config():
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    return config


