from matplotlib import pyplot as plt
from utils.compare_gradients import get_cos_similarites_batch, get_n_best_matches
from utils.process_data import encode, pad_to_equal_length
from utils.compute_gradients import (
    get_layer_gradients,
    get_layer_integrated_gradients,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
import torch
import pickle
from pprint import pprint
from os import path
from scipy import stats
from tqdm import tqdm
import pandas as pd
from models.distilbert_finetuned import get_distilbert_finetuned

# Ipython debugger
import ipdb
import numpy as np


def get_attr_scores():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model, tokenizer, layers = get_distilbert_finetuned()

    def fwd(inputs, mask):
        return model(inputs, attention_mask=mask).logits

    def fwd_return_best(inputs, mask):
        results = model(inputs, attention_mask=mask).logits
        best = torch.argmax(results, dim=1)
        return torch.gather(results, 1, best.unsqueeze(1))

    n_samples = 800

    train_dataset = load_dataset("glue", "sst2", split="train")
    test_ds = load_dataset("glue", "sst2", split=f"test[:{n_samples}]")

    # Define Dataloaders
    train_ds = train_dataset.map(
        encode, batched=True, fn_kwargs={"tokenizer": tokenizer}
    )
    train_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    train_dataloader = torch.utils.data.DataLoader(
        train_ds, collate_fn=tokenizer.pad, batch_size=20
    )

    test_ds = test_ds.map(
        encode,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer},
        load_from_cache_file=False,
    )
    test_ds.set_format(
        "torch",
        columns=["input_ids", "attention_mask", "label"],
        # output_all_columns=True,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_ds, collate_fn=tokenizer.pad, batch_size=5
    )

    # Get Gradients
    pickled_grads = "./dense_gradients.pkl"
    if not path.isfile(pickled_grads):
        print("Calculating gradients...")
        grads = get_layer_gradients(train_dataloader, layers, fwd)
        print("Saving gradients...")
        with open(pickled_grads, "wb") as f:
            pickle.dump(grads, f)
    else:
        print("Loading saved gradients...")
        with open(pickled_grads, "rb") as f:
            grads = pickle.load(f)

    example_scores = torch.empty((len(test_ds), grads.shape[1]))
    counter = 0
    for i, batch in enumerate(tqdm(test_dataloader)):
        activations = get_layer_integrated_gradients(
            inputs=batch["input_ids"].to(device),
            mask=batch["attention_mask"].to(device),
            # target=batch["label"].to(device),
            layers=layers.to(device),
            fwd=fwd_return_best,
            device=device,
        )
        simils = get_cos_similarites_batch(activations, grads, sparse=False)
        # Simils is batch_size x n_train
        batch_size = len(batch["label"])
        example_scores[counter : counter + batch_size] = simils
        counter += batch_size
    # mean_scores = torch.mean(example_scores, dim=0)
    np.save(f"./attr_scores_{n_samples}.npy", example_scores)
    return example_scores


if __name__ == "__main__":
    scores = np.load("./data/attr_scores_800.npy")
    mean_scores = np.mean(scores, axis=0)
    max_scores = np.max(scores, axis=0)
    min_scores = np.min(scores, axis=0)
    rdm_test_A = scores[42]
    rdm_test_B = scores[742]
    counts, bins = np.histogram(mean_scores)
    plt.ylabel("Count")
    plt.hist(bins[:-1], bins=bins, weights=counts)
    plt.xlabel("Cos Similarity")
    plt.title("Mean score (n=800)")
    plt.figure()
    counts2, bins2 = np.histogram(max_scores)
    plt.hist(bins2[:-1], bins=bins2, weights=counts2)
    plt.ylabel("Count")
    plt.xlabel("Cos Similarity")
    plt.title("Max score (n=800)")
    plt.figure()
    counts3, bins3 = np.histogram(min_scores)
    plt.hist(bins3[:-1], bins=bins3, weights=counts3)
    plt.ylabel("Count")
    plt.xlabel("Cos Similarity")
    plt.title("Min score (n=800)")
    plt.figure()
    counts4, bins4 = np.histogram(rdm_test_A)
    plt.hist(bins4[:-1], bins=bins4, weights=counts4)
    plt.ylabel("Count")
    plt.xlabel("Cos Similarity")
    plt.title("Random Test Example A")
    plt.figure()
    counts4, bins4 = np.histogram(rdm_test_B)
    plt.hist(bins4[:-1], bins=bins4, weights=counts4)
    plt.ylabel("Count")
    plt.xlabel("Cos Similarity")
    plt.title("Random Test Example B")
    plt.show()
