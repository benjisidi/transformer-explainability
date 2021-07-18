# %%
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from captum.attr import IntegratedGradients, LayerConductance, LayerIntegratedGradients
from captum._utils.gradient import compute_layer_gradients_and_eval
from functools import partial
from models.distilbert_sst2 import (
    get_all_layer_gradients,
    get_all_layer_conductance,
    get_all_layer_integrated_gradients,
    get_single_layer_conductance,
    get_single_layer_gradients,
)

# %%
tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)

model.eval()
# %%
test_sample = "Absolutely, positively tremendous"
training_examples = [
    "Absolutely terrible",
    "Absolutely wonderful",
    "Positively brilliant",
    "Absolutely tremendous",
]

# %%
# conductance = get_all_layer_conductance(test_sample, 1, model, tokenizer)
# # Conductance/IG shape: 1x4x768
# %%
# grads = [get_all_layer_gradients(x, 1, model, tokenizer) for x in training_examples]
# Grads shape: 1x6x768
# %%
conductance = get_single_layer_conductance(test_sample, 1, model, tokenizer, 0)
grads = get_single_layer_gradients(test_sample, 1, model, tokenizer, 0)

# %%
from datasets import load_dataset

# %%
dataset = load_dataset("glue", "sst2")
# %%
