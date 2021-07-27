# Imports
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils.compute_gradients import (
    get_all_layer_gradients,
    get_all_layer_integrated_gradients,
)
from utils.compare_gradients import get_cos_similarities, get_n_best_matches
from utils.process_data import make_input_batch
from pprint import pprint

from datasets import load_dataset

if __name__ == "__main__":

    dataset = load_dataset("glue", "sst2")
    #  Get test samples
    test_sample = dataset["validation"][0]["sentence"]
    test_label = dataset["validation"][0]["label"]
    train_samples = dataset["train"][:200]["sentence"]
    train_labels = dataset["train"][:200]["label"]
    # Define Model
    tokenizer = AutoTokenizer.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english"
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english"
    )
    model.eval()
    model.zero_grad()
    test_input = make_input_batch([test_sample], tokenizer)
    test_output = model(test_input).logits[:, test_label]
    test_output.backward()
    params = list(model.parameters())
    grads = [x.grad for x in params]
    grads_flat = [x.flatten() for x in grads]
    grads_big = torch.cat(grads_flat).flatten()
    print(grads_big.shape)
