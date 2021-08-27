from utils.process_data import encode
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

# Ipython debugger
# import ipdb


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_dataset = load_dataset("glue", "sst2", split="train")
    test_example = load_dataset("glue", "sst2", split="test[:10]")

    # Define Model
    tokenizer = AutoTokenizer.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english"
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english"
    ).to(device)
    model.eval()
    model.zero_grad()
    layers = model.distilbert.transformer.layer

    def fwd(inputs, mask):
        return model(inputs, attention_mask=mask).logits

    # Define Dataloader
    ds = train_dataset.map(encode, batched=True, fn_kwargs={"tokenizer": tokenizer})
    ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    dataloader = torch.utils.data.DataLoader(
        ds, collate_fn=tokenizer.pad, batch_size=30
    )
    # Get Gradients
    pickled_grads = "./dense_gradients.pkl"
    print("Calculating gradients...")
    grads = get_all_layer_gradients_2(
        dataloader, layers, fwd, sparse=False, device=device
    )
    print("Saving gradients...")
    with open(pickled_grads, "wb") as f:
        pickle.dump(grads, f)
