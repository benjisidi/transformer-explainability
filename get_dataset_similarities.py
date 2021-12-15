import numpy as np
from tqdm import tqdm
from utils.process_data import get_sst2, get_total_features
from utils.compute_gradients import get_layer_gradients_independent
from utils.compare_gradients import get_scores
import torch
from models.distilbert_finetuned import get_distilbert_finetuned

# Ipython debugger
import ipdb


def calc_attrs_and_scores():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model, tokenizer, layers = get_distilbert_finetuned()

    train_ds, test_ds = get_sst2(tokenizer)
    loader = torch.utils.data.DataLoader(
        train_ds, collate_fn=tokenizer.pad, batch_size=25
    )

    model = model.to(device)

    def fwd(inputs, mask):
        return model(inputs, attention_mask=mask).logits

    def fwd_return_best(inputs, mask):
        model.zero_grad()
        results = model(inputs, attention_mask=mask).logits
        best = torch.argmax(results, dim=1)
        return torch.gather(results, 1, best.unsqueeze(1))

    grads = torch.tensor(np.load("data/fullsize_grads_dense.npy"))
    # all_attributions = torch.tensor(np.load("data/all_attributions.npy"))
    # all_attributions = all_attributions[:, :-768]
    # grads = grads[:, :-768]
    all_scores = torch.zeros(len(test_ds), len(train_ds))
    all_attributions = torch.zeros(len(test_ds), get_total_features(layers))
    for test_point in tqdm(test_ds):
        index = test_point["id"].item()
        scores, attributions = get_scores(
            test_point, grads, layers, device, fwd_return_best, return_attributions=True
        )
        all_scores[index] = scores
        all_attributions[index] = attributions

    np.save("./data/all_scores_fixed.npy", all_scores)
    np.save("./data/all_attributions_fixed.npy", all_attributions)


def calc_scores_no_preclassifier():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model, tokenizer, layers = get_distilbert_finetuned()

    train_ds, test_ds = get_sst2(tokenizer)
    loader = torch.utils.data.DataLoader(
        train_ds, collate_fn=tokenizer.pad, batch_size=25
    )

    model = model.to(device)
    grads = torch.tensor(np.load("data/fullsize_grads_dense.npy"))[:, :-768]
    all_attributions = torch.tensor(np.load("data/all_attributions.npy"))[:, :-768]
    all_scores = torch.zeros(len(test_ds), len(train_ds))
    for test_point in tqdm(test_ds):
        index = test_point["id"].item()
        scores = get_scores(
            test_point,
            grads,
            layers,
            device,
            return_attributions=False,
            test_attributions=all_attributions[index],
        )
        all_scores[index] = scores

    np.save("./data/all_scores_no_preclassifier.npy", all_scores)
    ipdb.set_trace()


if __name__ == "__main__":
    calc_scores_no_preclassifier()
