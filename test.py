from utils.compare_gradients import get_cos_similarites_2, get_n_best_matches
from utils.process_data import encode
from utils.compute_gradients import get_all_layer_gradients_2, get_all_layer_integrated_gradients
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
import torch
import pickle
from pprint import pprint
from os import path
# Ipython debugger
import ipdb


if __name__ == "__main__":

    train_dataset = load_dataset("glue", "sst2", split="train[:2%]")
    test_example = load_dataset("glue", "sst2", split="test[:1]")

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

    def fwd(inputs, mask): return model(inputs,
                                        attention_mask=mask).logits

    # Define Dataloader
    ds = train_dataset.map(encode, batched=True, fn_kwargs={
        "tokenizer": tokenizer})
    ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    dataloader = torch.utils.data.DataLoader(
        ds, collate_fn=tokenizer.pad, batch_size=20)

    # Get Gradients
    pickled_grads = "./gradients.pkl"
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
    test_example = test_example.map(encode, fn_kwargs={"tokenizer": tokenizer})
    test_example.set_format(
        "torch", columns=["input_ids", "attention_mask", "label"])

    activations = get_all_layer_integrated_gradients(
        inputs=test_example["input_ids"],
        mask=test_example["attention_mask"],
        target=test_example["label"],
        layers=layers,
        fwd=fwd
    )
    activations = activations.squeeze().sum(dim=1)
    simils = get_cos_similarites_2(activations, grads).unsqueeze(0)
    best_matches = get_n_best_matches(simils, ds["sentence"])
    print(len(best_matches[0]))
    print("Test sentence: ", test_example["sentence"])
    print("Best train sentences:")
    pprint(best_matches, width=160)
