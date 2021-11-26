import pickle
import pprint
from os import path
from pprint import pprint

# Ipython debugger
import ipdb
import numpy as np
import seaborn as sb
import torch
from matplotlib import pyplot as plt
from scipy.stats import describe, pearsonr, spearmanr
from sklearn.mixture import GaussianMixture

from tqdm import tqdm

from models.distilbert_finetuned import get_distilbert_finetuned
from utils.compare_gradients import get_embedding_scores
from utils.compute_gradients import get_embeddings
from utils.process_data import get_sst2


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
    np.random.seed(111)
    model, tokenizer, layers = get_distilbert_finetuned()
    train_ds, test_ds = get_sst2(tokenizer)
    random_points = np.random.randint(len(test_ds), size=9)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=200, collate_fn=tokenizer.pad
    )
    tiny = train_ds.select(list(range(400)))
    tiny_loader = torch.utils.data.DataLoader(
        tiny, batch_size=200, collate_fn=tokenizer.pad
    )
    # train_embeddings = get_embeddings(
    #     train_loader, model, model.get_input_embeddings().weight.size()[1]
    # )
    train_embeddings = torch.tensor(np.load("./data/train_embeddings.npy"))
    fig, axs = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True)
    for i, point in enumerate(random_points):
        simils = get_embedding_scores(test_ds[int(point)], train_embeddings, model)
        sb.scatterplot(y=simils, x=scores[point], ax=axs.flat[i], s=2)
        correlation, p = pearsonr(simils, scores[point])
        axs.flat[i].set_title(f"Test Point #{point} (r={correlation:.2f})")
    fig.add_subplot(111, frame_on=False)
    plt.tick_params(labelcolor="none", bottom=False, left=False)
    plt.xlabel("Attribution-Gradient similarity")
    plt.ylabel("Embedding similarity")
    plt.suptitle("Embedding Similarity vs Attribution-Gradient Similarity")


def plot_feature_magnitudes():
    all_attributions = torch.tensor(np.load("./data/all_attributions.npy"))
    attr_absmean = torch.mean(torch.abs(all_attributions), dim=0)
    fig, axs = plt.subplots(nrows=1, ncols=2)
    sb.histplot(attr_absmean, bins=20, log_scale=(False, True), ax=axs[1])
    axs[1].set_xlabel("Mean absolute attribution")
    axs[1].set_title("Test-set integrated gradients")
    all_gradients = torch.tensor(np.load("./data/fullsize_grads_dense.npy"))
    grads_absmean = torch.mean(torch.abs(all_gradients), dim=0)

    correlation, p = pearsonr(attr_absmean, grads_absmean)
    print(correlation, p)
    sb.histplot(grads_absmean, bins=20, log_scale=(False, True), ax=axs[0])
    axs[0].set_xlabel("Mean absolute gradient")
    axs[0].set_title("Train-set gradients")
    plt.suptitle("Mean absolute magnitudes of feature gradients")


# indices of top 10% of each
# merged indices
# merged data coeff and scatter


def plot_top_features():
    all_attributions = torch.tensor(np.load("./data/all_attributions.npy"))
    attr_absmean = torch.mean(torch.abs(all_attributions), dim=0)
    attr_mean = torch.mean(all_attributions, dim=0)
    all_gradients = torch.tensor(np.load("./data/fullsize_grads_dense.npy"))
    grads_absmean = torch.mean(torch.abs(all_gradients), dim=0)
    grads_mean = torch.mean(all_gradients, dim=0)
    attr_idx = np.asarray(attr_absmean >= 0.025).nonzero()[0]
    grads_idx = np.asarray(grads_absmean >= 0.005).nonzero()[0]
    merged_idx = torch.tensor(np.unique(np.concatenate((attr_idx, grads_idx), 0)))
    correlation, p = pearsonr(attr_absmean[merged_idx], grads_absmean[merged_idx])
    rank, rp = spearmanr(attr_absmean[merged_idx], grads_absmean[merged_idx])
    print(correlation, p)
    print(rank, rp)

    # Following https://machinelearningmastery.com/clustering-algorithms-with-python/
    zipped_data = torch.cat(
        (attr_absmean[merged_idx].unsqueeze(1), grads_absmean[merged_idx].unsqueeze(1)),
        dim=1,
    )
    model = GaussianMixture(n_components=2)
    model.fit(zipped_data)
    yhat = model.predict(zipped_data)
    clusters = np.unique(yhat)

    fig, axs = plt.subplots(nrows=1, ncols=2)

    for cluster in clusters:
        row_ix = np.where(yhat == cluster)
        sb.scatterplot(
            x=attr_absmean[merged_idx][row_ix],
            y=grads_absmean[merged_idx][row_ix],
            ax=axs[0],
        )
        sb.scatterplot(
            x=attr_mean[merged_idx][row_ix],
            y=grads_mean[merged_idx][row_ix],
            ax=axs[1],
        )
    axs[0].set_xlabel("Test set mean absolute attribution")
    axs[0].set_ylabel("Train set mean absolute gradient")
    axs[0].set_title("Mean absolute scores")
    axs[1].set_xlabel("Test set mean attribution")
    axs[1].set_ylabel("Train set mean gradient")
    axs[1].set_title("Mean scores")
    plt.suptitle(
        "Mean absolute gradients and attributions, top scoring points (r=0.33)"
    )
    ipdb.set_trace()
    # ToDo Crit: Colour by distilbert layer!


if __name__ == "__main__":
    # Scores are test x train
    # scores = np.load("./data/all_scores.npy")
    # plot_feature_magnitudes()
    plot_top_features()
    plt.show()
