import torch
from models.distilbert_finetuned import get_distilbert_finetuned
from utils.process_data import get_sst2
from utils.compute_gradients import get_layer_output_gradients
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model, tokenizer, layers = get_distilbert_finetuned()


def fwd_return_best(inputs, mask):
    model.zero_grad()
    results = model(inputs, attention_mask=mask).logits
    best = torch.argmax(results, dim=1)
    return torch.gather(results, 1, best.unsqueeze(1))


train_dataset, test_dataset = get_sst2(tokenizer)
loader = torch.utils.data.DataLoader(
    train_dataset,
    collate_fn=tokenizer.pad,
    batch_size=1,
)

grads = get_layer_output_gradients(loader, layers, fwd_return_best, device)
np.save("./data/fullsize_dense_grads", grads.numpy())
