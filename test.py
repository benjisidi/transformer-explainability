from utils.compare_gradients import get_cos_similarites_2, get_n_best_matches
from utils.process_data import encode, pad_to_equal_length
from utils.compute_gradients import (
    get_all_layer_gradients_2,
    get_all_layer_integrated_gradients,
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

# Ipython debugger
# import ipdb
import numpy as np

if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_dataset = load_dataset("glue", "sst2", split="train")
    test_examples = load_dataset("glue", "sst2", split="test[:10]")

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

    def fwd(inputs, mask):
        return model(inputs, attention_mask=mask).logits

    # Define Dataloader
    ds = train_dataset.map(encode, batched=True, fn_kwargs={"tokenizer": tokenizer})
    ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    dataloader = torch.utils.data.DataLoader(
        ds, collate_fn=tokenizer.pad, batch_size=20
    )

    # Get Gradients
    pickled_grads = "./dense_gradients.pkl"
    if not path.isfile(pickled_grads):
        print("Calculating gradients...")
        grads = get_all_layer_gradients_2(dataloader, layers, fwd)
        print("Saving gradients...")
        with open(pickled_grads, "wb") as f:
            pickle.dump(grads, f)
    else:
        print("Loading saved gradients...")
        with open(pickled_grads, "rb") as f:
            grads = pickle.load(f)
    # # Get activations from test example
    test_examples = test_examples.map(encode, fn_kwargs={"tokenizer": tokenizer})
    test_examples.set_format(
        "torch",
        columns=["input_ids", "attention_mask", "label"],
        output_all_columns=True,
    )
    ds = ds.map(
        lambda x: {
            "embeddings": model.distilbert.embeddings(
                x["input_ids"].unsqueeze(0)
            ).flatten()
        },
        cache_file_name="/home/benji/.cache/huggingface/datasets/glue/sst2/1.0.0/dacbe3125aa31d7f70367a07a8a9e72a5a0bfeb5fc42e75c9db75b96da6053ad/cache-239a3d208660a33a.arrow",
    )
    for test_example in test_examples:
        activations = get_all_layer_integrated_gradients(
            inputs=test_example["input_ids"],
            mask=test_example["attention_mask"],
            target=test_example["label"],
            layers=layers,
            fwd=fwd,
        )
        # activations.to(device)
        # grads.to(device)
        activations = activations.squeeze().sum(dim=1)
        simils = get_cos_similarites_2(activations, grads, sparse=False).unsqueeze(0)

        example_embedding = model.distilbert.embeddings(
            test_example["input_ids"].unsqueeze(0)
        ).flatten()
        cos = torch.nn.CosineSimilarity(dim=0)

        emb_simils = [
            cos(*pad_to_equal_length(x["embeddings"], example_embedding)).item()
            for x in tqdm(ds)
        ]
        print("Grad similarity distribution:")
        print(stats.describe(simils.squeeze().numpy()))
        print("Emb similarity distribution:")
        print(stats.describe(np.array(emb_simils)))
        grad_sentences, grad_scores, emb_grad_scores = get_n_best_matches(
            simils, ds["sentence"], emb_simils, n=20
        )[0]
        print("Test sentence: ", test_example["sentence"])
        print("Best train sentences (grads):")
        pprint(list(zip(grad_sentences, grad_scores, emb_grad_scores)), width=160)
        emb_sentences, emb_scores, grad_emb_scores = get_n_best_matches(
            torch.tensor(emb_simils).unsqueeze(0), ds["sentence"], simils[0], n=20
        )[0]
        print("Best train sentences (embs):")
        pprint(list(zip(emb_sentences, grad_emb_scores, emb_scores)), width=160)
