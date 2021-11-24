from models.distilbert_finetuned import get_distilbert_finetuned
from utils.compute_gradients import get_layer_integrated_gradients
from utils.compare_gradients import get_cos_similarities_batch
from utils.process_data import get_sst2
import numpy as np
import torch
from matplotlib import pyplot as plt


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

gradients = torch.tensor(np.load("./data/fullsize_dense_grads.npy"))
plt.imshow(gradients[:2000, :10000], cmap="hot", interpolation="nearest")
plt.colorbar()
plt.show()
# feature = 1729

# model, tokenizer, layers = get_distilbert_finetuned()
# model.to(device)
# train_dataset, test_dataset = get_sst2(tokenizer)


# def fwd(inputs, mask):
#     return model(inputs, attention_mask=mask).logits


# train_dataset.set_format(
#     "torch",
#     columns=["input_ids", "attention_mask", "label", "id"],
#     output_all_columns=True,
# )
# test_dataset.set_format(
#     "torch",
#     columns=["input_ids", "attention_mask", "label", "id"],
#     output_all_columns=True,
# )
# test_point = test_dataset[324]
# print(test_point)
# test_attributions = get_layer_integrated_gradients(
#     test_point["input_ids"],
#     test_point["attention_mask"],
#     layers=layers,
#     fwd=fwd,
#     device=device,
#     target=test_point["label"],
# )

# scores = torch.zeros(len(gradients))
# cos_batch_size = 1000
# i = 0
# while i < len(gradients):
#     batch = gradients[i : i + cos_batch_size]
#     simils = get_cos_similarities_batch(test_attributions, batch)
#     scores[i : i + cos_batch_size] = simils
#     i += cos_batch_size

# sorted_scores, indices = torch.sort(scores, descending=True)
# print("TEST SENTENCE\n" + test_point["sentence"])
# print("RESULTS")
# for rank in range(10):
#     print(
#         f"""
# [{rank}]\t{train_dataset[indices[rank].item()]["sentence"]}\t{sorted_scores[rank]}\t{indices[rank].item()}
# """
#     )
