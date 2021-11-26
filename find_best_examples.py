from models.distilbert_finetuned import get_distilbert_finetuned
from utils.compute_gradients import get_layer_integrated_gradients
from utils.compare_gradients import get_cos_similarities_batch
from utils.process_data import get_sst2
import numpy as np
import torch
from matplotlib import pyplot as plt
from time import perf_counter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

gradients = torch.tensor(np.load("./data/fullsize_grads_dense.npy"))
plt.imshow(gradients[9000:11000, 0:20000], cmap="hot", interpolation="nearest")
plt.colorbar()
plt.show()
# feature = 9806
model, tokenizer, layers = get_distilbert_finetuned()
model.to(device)
train_dataset, test_dataset = get_sst2(tokenizer)


def fwd(inputs, mask):
    return model(inputs, attention_mask=mask).logits


train_dataset.set_format(
    "torch",
    columns=["input_ids", "attention_mask", "label", "id"],
    output_all_columns=True,
)
test_dataset.set_format(
    "torch",
    columns=["input_ids", "attention_mask", "label", "id"],
    output_all_columns=True,
)
test_point = test_dataset[123]
print(test_point)
start = perf_counter()
test_attributions = get_layer_integrated_gradients(
    test_point["input_ids"],
    test_point["attention_mask"],
    layers=layers,
    fwd=fwd,
    device=device,
    target=test_point["label"],
)
attr_time = perf_counter()
scores = torch.zeros(len(gradients))
cos_batch_size = 500
i = 0
test_attributions = test_attributions.to(device)
while i < len(gradients):
    batch = gradients[i : i + cos_batch_size]
    simils = get_cos_similarities_batch(test_attributions, batch.to(device))
    scores[i : i + cos_batch_size] = simils
    i += cos_batch_size
test_attributions = test_attributions.cpu()
sorted_scores, indices = torch.sort(scores, descending=True)
scores_time = perf_counter()
print(f"Cos batch size: {cos_batch_size}")
print(f"Attributions done in {attr_time-start}s.")
print(f"Scores done in  done in {scores_time-attr_time}s.")

print("TEST SENTENCE\n" + test_point["sentence"])
print("BEST RESULTS")
for rank in range(10):
    print(
        f"""
[{rank}]\t{train_dataset[indices[rank].item()]["sentence"]}\t{sorted_scores[rank]}\t{indices[rank].item()}
"""
    )
print("WORST RESULTS")
for rank in range(-1, -11, -1):
    print(
        f"""
[{rank}]\t{train_dataset[indices[rank].item()]["sentence"]}\t{sorted_scores[rank]}\t{indices[rank].item()}
"""
    )


sizes = [
    768,
    768,
    768,
    768,
    3072,
    768,
    768,
    768,
    768,
    768,
    3072,
    768,
    768,
    768,
    768,
    768,
    3072,
    768,
    768,
    768,
    768,
    768,
    3072,
    768,
    768,
    768,
    768,
    768,
    3072,
    768,
    768,
    768,
    768,
    768,
    3072,
    768,
    768,
]
cutoffs = np.cumsum(sizes)
layers_of_interest = [
    "0.attn.q_lin",
    "0.attn.k_lin",
    "0.attn.v_lin",
    "0.attn.out_lin",
    "0.ffn.lin1",
    "0.ffn.lin2",
    "1.attn.q_lin",
    "1.attn.k_lin",
    "1.attn.v_lin",
    "1.attn.out_lin",
    "1.ffn.lin1",
    "1.ffn.lin2",
    "2.attn.q_lin",
    "2.attn.k_lin",
    "2.attn.v_lin",
    "2.attn.out_lin",
    "2.ffn.lin1",
    "2.ffn.lin2",
    "3.attn.q_lin",
    "3.attn.k_lin",
    "3.attn.v_lin",
    "3.attn.out_lin",
    "3.ffn.lin1",
    "3.ffn.lin2",
    "4.attn.q_lin",
    "4.attn.k_lin",
    "4.attn.v_lin",
    "4.attn.out_lin",
    "4.ffn.lin1",
    "4.ffn.lin2",
    "5.attn.q_lin",
    "5.attn.k_lin",
    "5.attn.v_lin",
    "5.attn.out_lin",
    "5.ffn.lin1",
    "5.ffn.lin2",
    "pre_classifier",
]


best_idx = indices[0].item()
worst_idx = indices[-1].item()
fig, ax = plt.subplots()
ax.plot(gradients[worst_idx] / torch.max(gradients[worst_idx]), label="worst match")
ax.plot(gradients[best_idx] / torch.max(gradients[best_idx]), label="best match")
ax.plot(test_attributions / torch.max(test_attributions), label="test attributions")
ax.legend()
for layer, cutoff in zip(layers_of_interest, cutoffs):
    ax.annotate(
        layer,
        xy=(cutoff, 0),
        xycoords=("data", "axes fraction"),
        xytext=(0, -32),
        textcoords="offset points",
        va="top",
        ha="center",
        rotation=45,
    )
plt.title("overplot")

best_diff = test_attributions / torch.max(test_attributions) - gradients[
    best_idx
] / torch.max(gradients[best_idx])
worst_diff = test_attributions / torch.max(test_attributions) - gradients[
    worst_idx
] / torch.max(gradients[worst_idx])
plt.figure()
plt.plot(best_diff, label="best diff")
plt.title("best diff")
plt.figure()
plt.title("worst diff")
plt.plot(worst_diff, label="worst diff")

plt.show()
