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


def get_embeddings(datum, model):
    embeds = model.distilbert.embeddings(datum["input_ids"].unsqueeze(0))
    return {
        "embeddings_flat": embeds.flatten(),
        "embeddings_avg": embeds.squeeze().mean(0),
    }


if __name__ == "__main__":

    # Load data
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_dataset = load_dataset("glue", "sst2", split="train")
    test_examples = load_dataset("glue", "sst2", split="test[:10]")

    # Define Model
    model, tokenizer, layers = get_distilbert_finetuned()

    def fwd(inputs, mask):
        return model(inputs, attention_mask=mask).logits

    # Define Dataloader
    ds = train_dataset.map(encode, batched=True, fn_kwargs={"tokenizer": tokenizer})
    ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    dataloader = torch.utils.data.DataLoader(
        ds, collate_fn=tokenizer.pad, batch_size=20
    )

    # Get Gradients
    pickled_grads = "./data/dense_gradients.pkl"
    if not path.isfile(pickled_grads):
        print("Calculating gradients...")
        grads = get_layer_gradients(dataloader, layers, fwd)
        print("Saving gradients...")
        with open(pickled_grads, "wb") as f:
            pickle.dump(grads, f)
    else:
        print("Loading saved gradients...")
        with open(pickled_grads, "rb") as f:
            grads = pickle.load(f)

    # Get activations from test example
    test_examples = test_examples.map(encode, fn_kwargs={"tokenizer": tokenizer})
    test_examples.set_format(
        "torch",
        columns=["input_ids", "attention_mask", "label"],
        output_all_columns=True,
    )
    model.cpu()
    ds = ds.map(get_embeddings, fn_kwargs={"model": model})

    model.to(device)
    for i, test_example in enumerate(test_examples):
        activations = get_layer_integrated_gradients(
            inputs=test_example["input_ids"],
            mask=test_example["attention_mask"],
            target=test_example["label"],
            layers=layers,
            fwd=fwd,
            device=device,
        )
        activations = activations.squeeze().sum(dim=1)
        simils = get_cos_similarites(activations, grads, sparse=False).unsqueeze(0)

        example_embedding = get_embeddings(test_example, model)
        cos = torch.nn.CosineSimilarity(dim=0)

        emb_simils_flat = [
            cos(
                *pad_to_equal_length(
                    x["embeddings_flat"], example_embedding["embeddings_flat"]
                )
            ).item()
            for x in tqdm(ds)
        ]
        emb_simils_avg = [
            cos(
                *pad_to_equal_length(
                    x["embeddings_avg"], example_embedding["embeddings_avg"]
                )
            ).item()
            for x in tqdm(ds)
        ]
        print("Grad similarity distribution:")
        print(stats.describe(simils.squeeze().numpy()))
        print("Emb similarity distribution (flat):")
        print(stats.describe(np.array(emb_simils_flat)))
        print("Emb similarity distribution (avg):")
        print(stats.describe(np.array(emb_simils_avg)))

        plt.scatter(simils, emb_simils_flat)
        plt.xlabel("Attribution score")
        plt.ylabel("Embedding score")
        plt.title(f"Attribution vs Embedding score, Test Example #{i+1}")
        plt.figure()
        plt.scatter(simils, emb_simils_avg)
        plt.xlabel("Attribution score")
        plt.ylabel("Embedding score")
        plt.title(f"Attribution vs Embedding score, Test Example #{i+1}")
        plt.show()
        # grad_sentences, grad_scores, emb_grad_scores = get_n_best_matches(
        #     simils, ds["sentence"], emb_simils, n=20
        # )[0]
        # print("Test sentence: ", test_example["sentence"])
        # print("Best train sentences (grads):")
        # pprint(list(zip(grad_sentences, grad_scores, emb_grad_scores)), width=160)
        # emb_sentences, emb_scores, grad_emb_scores = get_n_best_matches(
        #     torch.tensor(emb_simils).unsqueeze(0), ds["sentence"], simils[0], n=20
        # )[0]
        # print("Best train sentences (embs):")
        # pprint(list(zip(emb_sentences, grad_emb_scores, emb_scores)), width=160)
