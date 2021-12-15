import json
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
import matplotlib
from matplotlib import colors
from matplotlib import pyplot as plt
from scipy.stats import describe, pearsonr, spearmanr
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

from models.distilbert_finetuned import get_distilbert_finetuned
from utils.compare_gradients import get_embedding_scores
from utils.compute_gradients import get_embeddings
from utils.process_data import get_layer_output_size, get_sst2


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


def plot_similarity_histograms_by_label(scores):
    mean_scores = np.mean(scores, axis=0)
    max_scores = np.max(scores, axis=0)
    min_scores = np.min(scores, axis=0)
    model, tokenizer, layers = get_distilbert_finetuned()
    train_ds, test_ds = get_sst2(tokenizer)
    labels = torch.tensor(train_ds["label"]).squeeze()
    labels_str = ["Positive" if x == 1 else "Negative" for x in train_ds["label"]]

    # paired_mean = torch.cat(
    #     (torch.tensor(mean_scores).unsqueeze(dim=1), labels.unsqueeze(dim=1)), dim=1
    # ).numpy()
    # paired_max = torch.cat(
    #     (torch.tensor(max_scores).unsqueeze(dim=1), labels.unsqueeze(dim=1)), dim=1
    # ).numpy()
    # paired_min = torch.cat(
    #     (torch.tensor(min_scores).unsqueeze(dim=1), labels.unsqueeze(dim=1)), dim=1
    # ).numpy()
    df_mean = pd.DataFrame(
        zip(mean_scores, labels_str), columns=["Mean Score", "Sentiment Label"]
    )
    df_max = pd.DataFrame(
        zip(max_scores, labels_str), columns=["Max Score", "Sentiment Label"]
    )
    df_min = pd.DataFrame(
        zip(min_scores, labels_str), columns=["Min Score", "Sentiment Label"]
    )

    fig, axs = plt.subplots(1, 3, sharey=True)
    sb.histplot(
        df_mean, ax=axs[1], x="Mean Score", stat="probability", hue="Sentiment Label"
    )
    axs[1].set_ylabel("Proportion")
    axs[1].set_xlabel("Cos Similarity")
    axs[1].set_title("Mean")
    sb.histplot(
        df_max, ax=axs[2], x="Max Score", stat="probability", hue="Sentiment Label"
    )
    axs[2].set_ylabel("Proportion")
    axs[2].set_xlabel("Cos Similarity")
    axs[2].set_title("Max")
    sb.histplot(
        df_min, ax=axs[0], x="Min Score", stat="probability", hue="Sentiment Label"
    )
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
    sb.boxenplot(y=mean_scores, x=bin_data, ax=ax)
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
    plt.suptitle("Mean absolute magnitudes of neuron gradients")


# indices of top 10% of each
# merged indices
# merged data coeff and scatter


def plot_top_features():
    all_attributions = torch.tensor(np.load("./data/all_attributions.npy"))
    attr_absmean = torch.mean(torch.abs(all_attributions), dim=0)
    attr_absmean_mean = torch.mean(attr_absmean)
    attr_absmean_std = torch.std(attr_absmean)
    attr_absmean_threshold = attr_absmean_mean + 2 * attr_absmean_std
    attr_absmean_outliers = np.asarray(
        attr_absmean >= attr_absmean_threshold
    ).nonzero()[0]
    attr_mean = torch.mean(all_attributions, dim=0)
    attr_std = torch.std(all_attributions, dim=0)
    all_gradients = torch.tensor(np.load("./data/fullsize_grads_dense.npy"))
    grads_absmean = torch.mean(torch.abs(all_gradients), dim=0)
    grads_absmean_mean = torch.mean(grads_absmean)
    grads_absmean_std = torch.std(grads_absmean)
    grads_absmean_threshold = grads_absmean_mean + 2 * grads_absmean_std
    grads_absmean_outliers = np.asarray(
        grads_absmean >= grads_absmean_threshold
    ).nonzero()[0]
    grads_mean = torch.mean(all_gradients, dim=0)
    grads_std = torch.std(all_gradients, dim=0)
    merged_idx = torch.tensor(
        np.unique(np.concatenate((attr_absmean_outliers, grads_absmean_outliers), 0))
    )
    print(describe(grads_absmean))
    print(describe(grads_mean))
    correlation, p = pearsonr(attr_absmean[merged_idx], grads_absmean[merged_idx])
    rank, rp = spearmanr(attr_absmean[merged_idx], grads_absmean[merged_idx])
    print(correlation, p)
    print(rank, rp)

    # Following https://machinelearningmastery.com/clustering-algorithms-with-python/
    # zipped_data = torch.cat(
    #     (attr_absmean[merged_idx].unsqueeze(1), grads_absmean[merged_idx].unsqueeze(1)),
    #     dim=1,
    # )
    # model = GaussianMixture(n_components=2)
    # model.fit(zipped_data)
    # yhat = model.predict(zipped_data)
    # clusters = np.unique(yhat)

    fig, axs = plt.subplots(nrows=1, ncols=2)

    # for cluster in clusters:
    #     row_ix = np.where(yhat == cluster)
    #     sb.scatterplot(
    #         x=attr_absmean[merged_idx][row_ix],
    #         y=grads_absmean[merged_idx][row_ix],
    #         ax=axs[0],
    #     )
    #     sb.scatterplot(
    #         x=attr_mean[merged_idx][row_ix],
    #         y=grads_mean[merged_idx][row_ix],
    #         ax=axs[1],
    #     )

    model, tokenizer, layers = get_distilbert_finetuned()
    with open("./distilbert_layers.json", "r") as f:
        distilbert_layers = json.load(f)
    layer_names = distilbert_layers["layers"]
    sizes = [get_layer_output_size(x) for x in layers]
    bins = np.cumsum(sizes)
    layer_indices = np.digitize(merged_idx, bins=bins)
    for layer_idx in np.unique(layer_indices):
        row_ix = np.where(layer_indices == layer_idx)[0]
        sb.scatterplot(
            x=attr_absmean[merged_idx][row_ix],
            y=grads_absmean[merged_idx][row_ix],
            ax=axs[0],
            label=layer_names[layer_idx],
        )
        sb.scatterplot(
            x=attr_mean[merged_idx][row_ix],
            y=grads_mean[merged_idx][row_ix],
            ax=axs[1],
            label=layer_names[layer_idx],
        )
    axs[0].set_xlabel("Test set mean absolute attribution")
    axs[0].set_ylabel("Train set mean absolute gradient")
    axs[0].set_title("Mean absolute scores")
    axs[1].set_xlabel("Test set mean attribution")
    axs[1].set_ylabel("Train set mean gradient")
    axs[1].set_title("Mean scores")
    plt.legend()
    plt.suptitle("Mean absolute gradients and attributions, outlier points")

    layer_idx = np.unique(layer_indices)[-1]
    row_ix = np.where(layer_indices == layer_idx)[0]
    points = merged_idx[row_ix]
    rank, rp = pearsonr(attr_mean[merged_idx][row_ix], grads_mean[merged_idx][row_ix])
    plt.figure()
    # sb.scatterplot(
    #     x=attr_mean[merged_idx][row_ix],
    #     y=grads_mean[merged_idx][row_ix],
    #     label=layer_names[layer_idx],
    #     hue=[sb.color_palette()[1]],
    # )
    plt.errorbar(
        x=attr_mean[merged_idx][row_ix],
        y=grads_mean[merged_idx][row_ix],
        xerr=attr_std[merged_idx][row_ix],
        yerr=grads_std[merged_idx][row_ix],
        ecolor=["coral"],
        marker="o",
        color="coral",
        linestyle="",
    )
    plt.xlabel("Test set mean attribution")
    plt.ylabel("Train set mean gradient")
    plt.title(f"Mean scores, preclassifier layer only (r={rank:.2f})")
    print(rank, rp)


def plot_embedding_layer_heatmap():
    all_attributions = torch.tensor(np.load("./data/all_attributions.npy"))
    embedding_layer_attributions = all_attributions[:, :768]
    preclassifier_layer_attributions = all_attributions[:, -768:]
    attn_layer_attributions = all_attributions[:, -768 * 3 - 3072 : -768 * 2 - 3072]
    ffn_layer_attributions = all_attributions[:, -768 * 2 : -768]
    fig, axs = plt.subplots(2, 2)

    sb.heatmap(
        embedding_layer_attributions.T,
        cmap="seismic",
        ax=axs.flat[0],
        norm=colors.CenteredNorm(),
    )
    sb.heatmap(
        preclassifier_layer_attributions.T,
        cmap="seismic",
        ax=axs.flat[1],
        norm=colors.CenteredNorm(),
    )
    sb.heatmap(
        ffn_layer_attributions.T,
        cmap="seismic",
        ax=axs.flat[2],
        norm=colors.CenteredNorm(),
    )
    sb.heatmap(
        attn_layer_attributions.T,
        cmap="seismic",
        ax=axs.flat[3],
        norm=colors.CenteredNorm(),
    )
    axs.flat[0].set_title("Embedding Layer")
    axs.flat[1].set_title("Preclassifier Layer")
    axs.flat[3].set_title("Layer 5 Attention Output")
    axs.flat[2].set_title("Layer 5 FFN Linear 2")
    axs.flat[0].set_ylabel("Neuron")
    axs.flat[0].set_xlabel("Test Example")
    axs.flat[1].set_ylabel("Neuron")
    axs.flat[1].set_xlabel("Test Example")
    axs.flat[2].set_ylabel("Neuron")
    axs.flat[2].set_xlabel("Test Example")
    axs.flat[3].set_ylabel("Neuron")
    axs.flat[3].set_xlabel("Test Example")
    plt.suptitle("Neuron attribution heatmaps per layer, test set")

    plt.figure()
    sb.heatmap(
        embedding_layer_attributions.T,
        cmap="seismic",
        norm=colors.CenteredNorm(),
    )
    plt.xlabel("Test Example #")
    plt.ylabel("Neuron")
    plt.title("Embedding Layer Attribution Heatmap")
    plt.figure()
    sb.heatmap(
        preclassifier_layer_attributions.T,
        cmap="seismic",
        norm=colors.CenteredNorm(),
    )
    plt.title("Pre-Classifier Layer Attribution Heatmap")
    plt.xlabel("Test Example #")
    plt.ylabel("Neuron")
    plt.figure()
    sb.heatmap(
        ffn_layer_attributions.T,
        cmap="seismic",
        norm=colors.CenteredNorm(),
    )
    plt.title("Layer 5 Feed-Forward Linear 2 Attribution Heatmap")
    plt.xlabel("Test Example #")
    plt.ylabel("Neuron")
    plt.figure()
    sb.heatmap(
        attn_layer_attributions.T,
        cmap="seismic",
        norm=colors.CenteredNorm(),
    )
    plt.title("Layer 5 Attention Output Attribution Heatmap")
    plt.xlabel("Test Example #")
    plt.ylabel("Neuron")

    test_labels = np.load("./data/test_labels.npy")
    test_positive_indices = np.asarray(test_labels == 1).nonzero()[0]
    test_negative_indices = np.asarray(test_labels == 0).nonzero()[0]
    fig2, axs2 = plt.subplots(1, 2)
    sb.heatmap(
        preclassifier_layer_attributions[test_positive_indices].T,
        cmap="seismic",
        ax=axs2.flat[0],
        norm=colors.CenteredNorm(),
    )
    sb.heatmap(
        preclassifier_layer_attributions[test_negative_indices].T,
        cmap="seismic",
        ax=axs2.flat[1],
        norm=colors.CenteredNorm(),
    )
    axs2.flat[0].set_ylabel("Neuron")
    axs2.flat[0].set_xlabel("Test Example")
    axs2.flat[1].set_ylabel("Neuron")
    axs2.flat[1].set_xlabel("Test Example")
    axs2.flat[0].set_title("Positive Classification")
    axs2.flat[1].set_title("Negative Classification")
    plt.suptitle("Neuron attribution heatmaps, preclassifier layer")
    fig3, axs3 = plt.subplots(1, 2)
    sb.heatmap(
        embedding_layer_attributions[test_positive_indices].T,
        cmap="seismic",
        ax=axs3.flat[0],
        norm=colors.CenteredNorm(),
    )
    sb.heatmap(
        embedding_layer_attributions[test_negative_indices].T,
        cmap="seismic",
        ax=axs3.flat[1],
        norm=colors.CenteredNorm(),
    )
    axs3.flat[0].set_ylabel("Neuron")
    axs3.flat[0].set_xlabel("Test Example")
    axs3.flat[1].set_ylabel("Neuron")
    axs3.flat[1].set_xlabel("Test Example")
    axs3.flat[0].set_title("Positive Classification")
    axs3.flat[1].set_title("Negative Classification")
    plt.suptitle("Neuron attribution heatmaps, Embedding Layer")

    model, tokenizer, layers = get_distilbert_finetuned()
    train_ds, test_ds = get_sst2(tokenizer, return_sentences=True)
    example_selection = np.random.randint(len(train_ds), size=len(test_ds))
    all_gradients = torch.tensor(np.load("./data/fullsize_grads_dense.npy"))
    selected_gradients = all_gradients[example_selection]
    preclassifier_layer_gradients = selected_gradients[:, -768:]
    selected_labels = train_ds[example_selection]["label"]
    train_positive_indices = np.asarray(selected_labels == 1).nonzero()[0]
    train_negative_indices = np.asarray(selected_labels == 0).nonzero()[0]
    embedding_layer_gradients = selected_gradients[:, :768]

    fig4, axs4 = plt.subplots(1, 2)
    sb.heatmap(
        embedding_layer_gradients[train_positive_indices].T,
        cmap="seismic",
        ax=axs4.flat[0],
        norm=colors.CenteredNorm(),
    )
    sb.heatmap(
        embedding_layer_gradients[train_negative_indices].T,
        cmap="seismic",
        ax=axs4.flat[1],
        norm=colors.CenteredNorm(),
    )
    axs4.flat[0].set_ylabel("Neuron")
    axs4.flat[0].set_xlabel("Train Example")
    axs4.flat[1].set_ylabel("Neuron")
    axs4.flat[1].set_xlabel("Train Example")
    axs4.flat[0].set_title("Positive Classification")
    axs4.flat[1].set_title("Negative Classification")
    plt.suptitle("Neuron gradient heatmaps, Embedding Layer")


def plot_example_sentence_heatmaps(scores):
    model, tokenizer, layers = get_distilbert_finetuned()
    train_ds, test_ds = get_sst2(tokenizer, return_sentences=True)
    example_selection = np.random.randint(len(test_ds), size=20)
    test_idx = int(example_selection[0])
    relevant_scores = scores[test_idx]
    sorted, idx = torch.sort(torch.tensor(relevant_scores), descending=True)
    test_sentence = test_ds[test_idx]["sentence"]
    best_train_matches = train_ds[idx[:10]]["sentence"]
    best_train_scores = sorted[:10]
    all_attributions = torch.tensor(np.load("./data/all_attributions.npy"))
    all_gradients = torch.tensor(np.load("./data/fullsize_grads_dense.npy"))
    ipdb.set_trace()
    # Note: Single examples can be visualised as 200x192 rect heatmap


def plot_avg_rank(scores):
    pass


if __name__ == "__main__":
    np.random.seed(111)
    # matplotlib.rcParams.update({"font.size": 18})
    # Scores are test x train
    scores = np.load("./data/all_scores_no_preclassifier.npy")
    # plot_similarity_histograms(scores)
    # plot_similarity_histograms_by_label(scores)
    # plot_embedding_correlation(scores)
    # plot_feature_magnitudes()
    # plot_embedding_layer_heatmap()
    # plot_example_sentence_heatmaps(scores)
    # plot_top_features()
    # plot_sentence_length_vs_mean(scores)
    plt.tight_layout()
    plt.show()
