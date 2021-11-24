# coding: utf-8
from ./models/distilbert_finetuned.py import get_distilbert_finetuned
from models/distilbert_finetuned.py import get_distilbert_finetuned
from models.distilbert_finetuned import get_distilbert_finetuned
model, tokenizer, layers = get_distilbert_finetuned()
model
layers
layers[0]
layers[0][0]
layers[0]["attention"]
layers[0],ffn
layers[0],ffn
layers[0].ffn
__dir__(layers[0])
layers[0].__dir__()
layers[0].__dir__()
layers[0].named_children
layers[0].named_children()
layers[0].modules()
layers[0].modules
[x for x in layers[0].modules()]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import pickle
from os import path

# Ipython debugger
# import ipdb
import numpy as np
import torch
from datasets import load_dataset
from matplotlib import pyplot as plt
from scipy import stats
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from utils.compare_gradients import get_cos_similarites
from utils.compute_gradients import get_layer_gradients, get_layer_integrated_gradients
from utils.process_data import encode, pad_to_equal_length
from models.distilbert_finetuned import get_distilbert_finetuned
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_dataset = load_dataset("glue", "sst2", split="train")
test_examples = load_dataset("glue", "sst2", split="test[:10]")
def fwd(inputs, mask):
    return model(inputs, attention_mask=mask).logits
    
ds = train_dataset.map(encode, batched=True, fn_kwargs={"tokenizer": tokenizer})
ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
dataloader = torch.utils.data.DataLoader(
        ds, collate_fn=tokenizer.pad, batch_size=20
    )
layers_flat = [*x.modues() for x in layers]
layers_flat = [*x.modules() for x in layers]
layers_flat = [layer for module in x.modules() for x in layers]
layers_flat = [list(x.modules()) for x in layers]
layers_flat
import itertoools
import itertools
layers_flat = list(itertools.chain(*layers_flat))
layers_flat
len(layers_flat)
gradients = get_layer_gradients(dataloader, layers_flat, fwd)
device
model.to(device)
gradients = get_layer_gradients(dataloader, layers_flat, fwd)
gradients = get_layer_gradients(dataloader, layers_flat, fwd, device=device)
gradients = get_layer_gradients(dataloader, layers_flat, fwd, device=device, sparse=False)
loader
dataloader
dataloader.__next__()
dataloader.__dir__()
dataloader.generator.__next__()
dataloader.generator.__dir__()
batch = dataloader._iterator.__next__()
dataloader._iterator
tokenizer
get_ipython().run_line_magic('save', 'explore.py')
get_ipython().run_line_magic('save', 'explore')
get_ipython().run_line_magic('save>', '')
get_ipython().run_line_magic('pinfo', '%save')
get_ipython().run_line_magic('save', 'explore.py 1-9999999')
