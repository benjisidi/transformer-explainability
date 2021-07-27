import torch
from torch import nn


def flatten_normalize_layers(params):
    summed = params.sum(axis=1)
    flat = summed.reshape(params.shape[0] * params.shape[2])
    return nn.functional.normalize(flat, dim=0)


def get_cos_similarites_2(test_attr, training_grads):
    _, n_samples, _ = training_grads.shape
    cos = nn.CosineSimilarity(dim=0)
    test_attr_normalized = nn.functional.normalize(test_attr.flatten(), dim=0)
    similarities = []
    for i in range(n_samples):
        sample_grad_normalized = nn.functional.normalize(
            torch.select(training_grads, 1, i).to_dense().flatten(), dim=0
        )  # ToDo: Should this happen before sparse?
        similarities.append(cos(test_attr_normalized, sample_grad_normalized))
    return torch.stack(similarities)


def get_cos_similarities(test_attributions, training_gradients):
    # Inputs: list of n_layers x n_samples x n_tokens x n_params tensors
    test_attributions = torch.stack(test_attributions)
    training_gradients = torch.stack(training_gradients)
    n_layers, n_test_samples, n_tokens, n_params = test_attributions.shape
    n_training_examples = training_gradients.shape[2]
    results = []
    for i in range(n_test_samples):
        test_attribution = test_attributions[:, i, :, :]
        attributions_flat = flatten_normalize_layers(test_attribution)
        grads_flat = []
        for j in range(n_training_examples):
            grads_flat.append(flatten_normalize_layers(training_gradients[:, j, :, :]))
        cos = nn.CosineSimilarity(dim=0)
        simils = [cos(attributions_flat, x) for x in grads_flat]
        results.append(simils)
    return torch.tensor(results)


def get_n_best_matches(scores, candidates, n=10):
    # Scores: n_test x n_train tensor of cos similarities
    # Candidates: n_train examples (str)
    results = []
    for sample_scores in scores:
        sorted_candidates = list(
            zip(*sorted(zip(sample_scores, candidates), reverse=True))
        )[1]
        best_matches = sorted_candidates[:n]
        results.append(best_matches)
    return results
