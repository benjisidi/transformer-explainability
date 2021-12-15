# %%
# Imports
from datasets import Dataset
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from captum.attr import LayerIntegratedGradients

from utils.compute_gradients import (
    get_layer_gradients,
    get_layer_integrated_gradients,
    get_all_layer_gradients,
)
from utils.compare_gradients import get_cos_similarites
from utils.process_data import make_input, encode
from pprint import pprint

from datasets import load_dataset

# %%
dataset = load_dataset("glue", "sst2")
#  Get test samples
test_sample = dataset["validation"][0]["sentence"]
test_label = dataset["validation"][0]["label"]
train_samples = dataset["train"][:500]["sentence"]
train_labels = dataset["train"][:500]["label"]

# %%
# Define Model
tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)
model.eval()
model.zero_grad()
layers = model.distilbert.transformer.layer

# %%
ds = dataset["train"]
ds = ds.map(encode, batched=True, fn_kwargs={"tokenizer": tokenizer})
ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
dataloader = torch.utils.data.DataLoader(ds, collate_fn=tokenizer.pad, batch_size=50)


def fwd(inputs, mask):
    return model(inputs, attention_mask=mask).logits


# %%
grads = get_layer_gradients(dataloader, layers, fwd)

# %%
test_input, test_mask = make_input([test_sample], tokenizer)
baseline = torch.zeros_like(test_input)
test_attributions = []
for layer in layers:
    ligs = LayerIntegratedGradients(forward_func=fwd, layer=layer)
    layer_grads = ligs.attribute(
        test_input,
        baselines=baseline,
        target=test_label,
        additional_forward_args=test_mask,
    )
    test_attributions.append(layer_grads)
test_attributions = torch.stack(test_attributions).squeeze().sum(1)
# %%

# %%
simils = get_cos_similarites(test_attr=test_attributions, training_grads=grads)
# %%
sorted_scores, sorted_candidates = list(
    zip(*sorted(zip(simils, train_samples), reverse=True))
)
# %%
list(zip(sorted_candidates, sorted_scores))
# %%
# from functools import partial
# pad = partial(tokenizer.pad, padding="longest")
# %%
ds = dataset["train"]
ds = ds.map(encode, batched=True, fn_kwargs={"tokenizer": tokenizer})
dataloader = torch.utils.data.DataLoader(ds, collate_fn=tokenizer.pad, batch_size=10)

for i, batch in enumerate(dataloader):
    if i < 5:
        print(batch)
    else:
        break

# %%

from models.distilbert_finetuned import get_distilbert_finetuned
from utils.compare_gradients import get_embedding_scores
from utils.compute_gradients import get_embeddings
from utils.process_data import get_layer_output_size, get_sst2
import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
model, tokenizer, layers = get_distilbert_finetuned()
train_ds, test_ds = get_sst2(tokenizer, return_sentences=True)

# %%
np.unique(train_ds["label"], return_counts=True)

# %%
train_ds[2]
# %%
test_labels = [
    torch.argmax(
        model(
            x["input_ids"].unsqueeze(0).to(device),
            attention_mask=x["attention_mask"].unsqueeze(0).to(device),
        ).logits[0]
    )
    .cpu()
    .item()
    for x in test_ds
]
# %%
np.save("./data/test_labels", test_labels)
# %%
