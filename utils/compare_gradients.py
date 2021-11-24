import torch
from torch import nn


def get_cos_similarities_batch(test_attributions, training_grads):
    """
    Takes a single list of test attributions and a batch of train gradients
    and returns cos similarities
    """
    # initialise cos similarity and output
    n_examples = training_grads.shape[0]
    cos = nn.CosineSimilarity(dim=1)
    tiled_attributions = torch.tile(test_attributions, (n_examples, 1))
    return cos(
        tiled_attributions,
        training_grads,
    )


def flatten_normalize_layers(params):
    summed = params.sum(axis=1)
    flat = summed.reshape(params.shape[0] * params.shape[2])
    return nn.functional.normalize(flat, dim=0)


def get_cos_similarites_batch_old(test_attr, training_grads, sparse=True):
    # First reshape and normalise test attributions
    layers, batch_size, tokens, neurons = test_attr.shape
    # Sum attributions over tokens
    test_attr_summed = torch.sum(test_attr, dim=2)
    # Swap dims so batch size is first
    test_attr_summed = torch.einsum("ijk -> jik", test_attr_summed)
    # Flatten layers
    test_attr_flat = torch.flatten(test_attr_summed, start_dim=1)
    test_attr_normalized = nn.functional.normalize(test_attr_flat, dim=1)

    # Next reshape and normalise training gradients
    layers, n_samples, neurons = training_grads.shape
    if sparse:
        training_grads = training_grads.to_dense()
    training_grads = torch.flatten(
        torch.einsum("ijk -> jik", training_grads), start_dim=1
    )
    training_grads = nn.functional.normalize(training_grads, dim=1)

    # initialise cos similarity and output
    cos = nn.CosineSimilarity(dim=1)
    output = torch.zeros((batch_size, n_samples))

    # calculate cos similarity for all training examples for each test sample in batch
    for i, test_attributions in enumerate(test_attr_normalized):
        tiled_attributions = torch.tile(test_attributions, (n_samples, 1))
        similarities = cos(tiled_attributions, training_grads)
        output[i] = similarities

    return output


def get_cos_similarites(test_attr, training_grads, sparse=True):
    _, n_samples, _ = training_grads.shape
    cos = nn.CosineSimilarity(dim=0)
    test_attr_normalized = nn.functional.normalize(test_attr.flatten(), dim=0)
    similarities = []
    for i in range(n_samples):
        if sparse:
            sample_grad_normalized = nn.functional.normalize(
                torch.select(training_grads, 1, i).to_dense().flatten(), dim=0
            )
        else:
            sample_grad_normalized = nn.functional.normalize(
                torch.select(training_grads, 1, i).flatten(), dim=0
            )
        similarities.append(cos(test_attr_normalized, sample_grad_normalized))

    return torch.stack(similarities)


def get_n_best_matches(scores, candidates, emb_simils, n=10):
    # Scores: n_test x n_train tensor of cos similarities
    # Candidates: n_train examples (str)
    results = []
    for sample_scores in scores:
        sorted_scores, sorted_candidates, sorted_emb_scores = list(
            zip(*sorted(zip(sample_scores, candidates, emb_simils), reverse=True))
        )
        best_matches = sorted_candidates[:n]
        best_scores = list(map(lambda x: x.item(), sorted_scores[:n]))
        results.append((best_matches, best_scores, sorted_emb_scores[:n]))
    return results
