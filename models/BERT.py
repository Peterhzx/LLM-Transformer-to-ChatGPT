import math

import torch
from torch import nn

import torch.nn.functional as F


class BERT(nn.Module):
    def __init__(self):
        super(BERT).__init__()
