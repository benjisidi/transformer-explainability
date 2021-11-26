import numpy as np
from tqdm import tqdm
from utils.process_data import get_sst2, get_total_features
from utils.compute_gradients import get_layer_gradients_independent
from utils.compare_gradients import get_scores
import torch
from models.distilbert_finetuned import get_distilbert_finetuned

# Ipython debugger
import ipdb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model, tokenizer, layers = get_distilbert_finetuned()

train_ds, test_ds = get_sst2(tokenizer)
loader = torch.utils.data.DataLoader(train_ds, collate_fn=tokenizer.pad, batch_size=25)

model = model.to(device)


def fwd(inputs, mask):
    return model(inputs, attention_mask=mask).logits


grads = torch.tensor(np.load("data/fullsize_grads_dense.npy"))

test_point = test_ds[0]
all_scores = torch.zeros(len(test_ds), len(train_ds))
all_attributions = torch.zeros(len(test_ds), get_total_features(layers))
for test_point in tqdm(test_ds):
    index = test_point["id"].item()
    scores, attributions = get_scores(
        test_point, grads, layers, fwd, device, return_attributions=True
    )
    all_scores[index] = scores
    all_attributions[index] = attributions

ipdb.set_trace()
np.save("./data/all_scores.npy", all_scores)
np.save("./data/all_attributions.npy", all_attributions)
