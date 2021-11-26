import torch
from torch import nn
from .compute_gradients import get_layer_integrated_gradients


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


def get_scores(
    test_point, train_gradients, layers, fwd, device, return_attributions=False
):
    test_attributions = get_layer_integrated_gradients(
        test_point["input_ids"],
        test_point["attention_mask"],
        layers=layers,
        fwd=fwd,
        device=device,
        target=test_point["label"],
    )
    scores = torch.zeros(len(train_gradients))
    cos_batch_size = 500
    i = 0
    while i < len(train_gradients):
        batch = train_gradients[i : i + cos_batch_size]
        simils = get_cos_similarities_batch(
            test_attributions.to(device), batch.to(device)
        )
        scores[i : i + cos_batch_size] = simils.cpu()
        i += cos_batch_size
    if return_attributions:
        return scores, test_attributions
    return scores


def get_embedding_scores(test_point, train_embeddings, model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # all_embeddings =
    # train_embeddings = embed_fn(train_points["input_ids"].to(device))
    embeddings = model.get_input_embeddings()
    test_embeddings = (
        torch.mean(embeddings(test_point["input_ids"].unsqueeze(0).to(device)), dim=1)
        .squeeze()
        .detach()
    )
    # ToDo output
    cos_batch_size = 500
    i = 0
    scores = torch.zeros(len(train_embeddings))
    while i < len(train_embeddings):
        batch = train_embeddings[i : i + cos_batch_size]
        simils = get_cos_similarities_batch(
            test_embeddings.to(device), batch.to(device)
        )
        scores[i : i + cos_batch_size] = simils.cpu()
        i += cos_batch_size
    return scores
