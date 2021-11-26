import pickle
import pprint
from os import path
from pprint import pprint

# Ipython debugger
import ipdb
import numpy as np
import pandas as pd
import seaborn as sb
import torch
from datasets import load_dataset
from matplotlib import pyplot as plt
from scipy import stats
from scipy.stats import describe
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from models.distilbert_finetuned import get_distilbert_finetuned
from utils.compare_gradients import get_cos_similarites, get_n_best_matches
from utils.compute_gradients import get_layer_gradients, get_layer_integrated_gradients
from utils.process_data import encode, get_sst2, pad_to_equal_length


def plot_similarity_histograms(scores):
    mean_scores = np.mean(scores, axis=0)
    max_scores = np.max(scores, axis=0)
    min_scores = np.min(scores, axis=0)
    fig, axs = plt.subplots(1, 3, sharey=True)
    sb.histplot(mean_scores, ax=axs[1], stat="probability")
    axs[1].set_ylabel("Proportion")
    axs[1].set_xlabel("Cos Similarity")
    axs[1].set_title("Mean")
    sb.histplot(max_scores, ax=axs[2], stat="probability")
    axs[2].set_ylabel("Proportion")
    axs[2].set_xlabel("Cos Similarity")
    axs[2].set_title("Max")
    sb.histplot(min_scores, ax=axs[0], stat="probability")
    axs[0].set_ylabel("Proportion")
    axs[0].set_xlabel("Cos Similarity")
    axs[0].set_title("Min")
    plt.suptitle("Cos similiarity score distribution, sst2 dataset")
    plt.tight_layout()


def plot_sentence_length_vs_mean(scores):
    mean_scores = np.mean(scores, axis=0)
    model, tokenizer, layers = get_distilbert_finetuned()
    train_ds, test_ds = get_sst2(tokenizer)
    sentence_lengths = [len(x) for x in train_ds["input_ids"]]
    bins = [0, 10, 20, 30, 40, 50, 60, 70]
    bin_data = np.digitize(sentence_lengths, bins, right=True)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    sb.boxplot(y=mean_scores, x=bin_data, ax=ax)
    ax.set_xticks(range(7))
    ax.set_xticklabels(["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-70"])
    ax.set_xlabel("Number of tokens")
    ax.set_ylabel("Mean similarity score")

    counts = np.bincount(bin_data)
    new_ax = ax.secondary_xaxis("top")
    new_ax.set_xticklabels(counts)
    new_ax.set_xlabel("Bin count")

    ax.set_title("Mean similarity score by sentence length")


def plot_embedding_correlation(scores):
    model, tokenizer, layers = get_distilbert_finetuned()
    train_ds, test_ds = get_sst2(tokenizer)
    random_points = np.random.randint(len(train_ds), size=9)
    print(random_points)


if __name__ == "__main__":
    # Scores are test x train
    scores = np.load("./data/all_scores.npy")
    plot_embedding_correlation(scores)
    # plt.show()
