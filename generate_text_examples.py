import json
import pickle
import pprint
from os import path
from pprint import pprint

# Ipython debugger
import ipdb
import numpy as np
import torch
from tqdm import tqdm

from models.distilbert_finetuned import get_distilbert_finetuned
from utils.compare_gradients import get_embedding_scores
from utils.compute_gradients import get_embeddings
from utils.process_data import get_layer_output_size, get_sst2

# ToDo: Average rank


def print_best_examples(test_idx, scores, train_ds, test_ds, test_labels):
    relevant_scores = scores[test_idx]
    sorted, idx = torch.sort(torch.tensor(relevant_scores), descending=True)
    test_sentence = test_ds[test_idx]["sentence"]
    best_train_matches = train_ds[idx[:10]]["sentence"]
    best_train_scores = sorted[:10]
    label_color_lookup = ["blue", "red"]
    test_label = test_labels[test_idx]
    test_color = label_color_lookup[test_label]
    with open("./tables.txt", "a") as f:
        f.write(
            """
\\begin{table}[htp]
\centering
\\resizebox{\\textwidth}{!}{%
\\begin{tabular}{lp{0.95\linewidth}l}
\hline
Id & Sentence & Score \\\\ \hline
"""
        )
        f.write(
            f"\\textcolor{{{test_color}}}{{\\textbf{{{test_idx}}}}} & \\textbf{{{test_sentence}}} & \\\\ \hline\n"
        )
        for rank in range(10):
            id_ = idx[rank].item()
            label = train_ds[id_]["label"].item()
            color = label_color_lookup[label]
            f.write(
                f"\\textcolor{{{color}}}{{{idx[rank]}}} & {best_train_matches[rank]} & {best_train_scores[rank]:.2f} \\\\\n"
            )
        f.write(
            f"""\hline
\end{{tabular}}
}}
\caption{{Top 10 scoring sentences for test example {test_idx}}}
\label{{tab:test_results_{test_idx}}}
\end{{table}}
"""
        )


if __name__ == "__main__":
    # Scores are test x train
    scores = np.load("./data/all_scores.npy")
    test_labels = np.load("./data/test_labels.npy")
    model, tokenizer, layers = get_distilbert_finetuned()
    train_ds, test_ds = get_sst2(tokenizer, return_sentences=True)
    np.random.seed(111)
    example_selection = np.random.randint(len(test_ds), size=20)
    for i in example_selection:
        print_best_examples(int(i), scores, train_ds, test_ds, test_labels)
