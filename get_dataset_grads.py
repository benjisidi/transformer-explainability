import numpy as np
from utils.process_data import get_sst2
from utils.compute_gradients import get_layer_gradients_independent
import torch
from models.distilbert_finetuned import get_distilbert_finetuned

# Ipython debugger
import ipdb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model, tokenizer, layers = get_distilbert_finetuned()

train_ds, test_ds = get_sst2(tokenizer)
loader = torch.utils.data.DataLoader(train_ds, collate_fn=tokenizer.pad, batch_size=25)

model = model.to(device)


def fwd(inputs, mask):
    return model(inputs, attention_mask=mask).logits


grads = get_layer_gradients_independent(
    loader, layers, fwd, sparse=False, device=device
)
np.save(grads.numpy(), "./data/fullsize_grads_dense.npy")
