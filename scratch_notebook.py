# %%
# Imports
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from captum.attr import LayerIntegratedGradients

from utils.compute_gradients import (
    get_all_layer_gradients,
    get_all_layer_integrated_gradients,
    get_all_layer_gradients_2,
)
from utils.compare_gradients import get_cos_similarities, get_n_best_matches
from utils.process_data import make_input_batch
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
tokenized_data, attention_masks = make_input_batch(train_samples, tokenizer)
dataloader = torch.utils.data.DataLoader(
    list(zip(tokenized_data, attention_masks, train_labels)), batch_size=50
)
fwd = lambda x, y: model(x, attention_mask=y).logits

# %%
grads = get_all_layer_gradients_2(dataloader, layers, fwd)

# %%
test_input, test_mask = make_input_batch([test_sample], tokenizer)
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
from utils.compare_gradients import get_cos_similarites_2

# %%
simils = get_cos_similarites_2(test_attr=test_attributions, training_grads=grads)
# %%
sorted_scores, sorted_candidates = list(
    zip(*sorted(zip(simils, train_samples), reverse=True))
)
# %%
list(zip(sorted_candidates, sorted_scores))
# %%
