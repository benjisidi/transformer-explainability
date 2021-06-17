# %%
import os
import sys

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from transformers import BertTokenizer, BertForQuestionAnswering, BertConfig

from captum.attr import visualization as viz
from captum.attr import IntegratedGradients, LayerConductance, LayerIntegratedGradients
from captum.attr import (
    configure_interpretable_embedding_layer,
    remove_interpretable_embedding_layer,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

device
