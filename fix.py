# %%
import torch
from models.distilbert_finetuned import get_distilbert_finetuned
from utils.compute_gradients import get_layer_integrated_gradients
from utils.process_data import encode, get_sst2
from datasets import load_dataset


def fwd(inputs, mask):
    return model(inputs, attention_mask=mask).logits


# %%

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model, tokenizer, layers = get_distilbert_finetuned()
# %%
layers_of_interest = [
    "distilbert.transformer.layer.0.attention.q_lin",
    "distilbert.transformer.layer.0.attention.k_lin",
    "distilbert.transformer.layer.0.attention.v_lin",
    "distilbert.transformer.layer.0.attention.out_lin",
    "distilbert.transformer.layer.0.ffn.lin1",
    "distilbert.transformer.layer.0.ffn.lin2",
    "distilbert.transformer.layer.1.attention.q_lin",
    "distilbert.transformer.layer.1.attention.k_lin",
    "distilbert.transformer.layer.1.attention.v_lin",
    "distilbert.transformer.layer.1.attention.out_lin",
    "distilbert.transformer.layer.1.ffn.lin1",
    "distilbert.transformer.layer.1.ffn.lin2",
    "distilbert.transformer.layer.2.attention.q_lin",
    "distilbert.transformer.layer.2.attention.k_lin",
    "distilbert.transformer.layer.2.attention.v_lin",
    "distilbert.transformer.layer.2.attention.out_lin",
    "distilbert.transformer.layer.2.ffn.lin1",
    "distilbert.transformer.layer.2.ffn.lin2",
    "distilbert.transformer.layer.3.attention.q_lin",
    "distilbert.transformer.layer.3.attention.k_lin",
    "distilbert.transformer.layer.3.attention.v_lin",
    "distilbert.transformer.layer.3.attention.out_lin",
    "distilbert.transformer.layer.3.ffn.lin1",
    "distilbert.transformer.layer.3.ffn.lin2",
    "distilbert.transformer.layer.4.attention.q_lin",
    "distilbert.transformer.layer.4.attention.k_lin",
    "distilbert.transformer.layer.4.attention.v_lin",
    "distilbert.transformer.layer.4.attention.out_lin",
    "distilbert.transformer.layer.4.ffn.lin1",
    "distilbert.transformer.layer.4.ffn.lin2",
    "distilbert.transformer.layer.5.attention.q_lin",
    "distilbert.transformer.layer.5.attention.k_lin",
    "distilbert.transformer.layer.5.attention.v_lin",
    "distilbert.transformer.layer.5.attention.out_lin",
    "distilbert.transformer.layer.5.ffn.lin1",
    "distilbert.transformer.layer.5.ffn.lin2",
    "pre_classifier",
    "classifier",
]


def get_nested_property(parent, key):
    cur_parent = parent
    key_list = key.split(".")
    for key in key_list:
        if key.isdigit():
            cur_parent = cur_parent[int(key)]
        else:
            cur_parent = getattr(cur_parent, key)
    return cur_parent


# %%
test_examples = load_dataset("glue", "sst2", split="test[:10]")
test_examples = test_examples.map(encode, fn_kwargs={"tokenizer": tokenizer})
test_examples.set_format(
    "torch",
    columns=["input_ids", "attention_mask", "label"],
    output_all_columns=True,
)
test_example = test_examples[0]
layers_flat = [get_nested_property(model, x) for x in layers_of_interest]
# %%
model.to(device)
output_igs = get_layer_integrated_gradients(
    test_example["input_ids"].to(device),
    test_example["attention_mask"].to(device),
    target=test_example["label"].to(device),
    layers=layers_flat,
    fwd=fwd,
    device=device,
)
output_igs_processed = [
    x.squeeze().mean(dim=0) if len(x.shape) == 3 else x.squeeze() for x in output_igs
]
# %%
model(
    test_example["input_ids"].unsqueeze(0).to(device),
    attention_mask=test_example["attention_mask"].unsqueeze(0).to(device),
).logits.squeeze()[test_example["label"].to(device)].backward()
grads = []
for layer in layers_flat:
    for param in layer.parameters():
        grads.append(param.grad)
output_grads = [x for x in grads if len(x.shape) == 1]
# grads: torch.Size([66955010])
# flat ligs: tensor(769032)

# %%
train_dataset, test_dataset = get_sst2(tokenizer)
loader = torch.utils.data.DataLoader(
    train_dataset,
    collate_fn=tokenizer.pad,
    batch_size=1,
)

# %%
from tqdm.notebook import tqdm


def fwd_return_best(inputs, mask):
    results = model(inputs, attention_mask=mask).logits
    best = torch.argmax(results, dim=1)
    return torch.gather(results, 1, best.unsqueeze(1))


all_grads = torch.zeros(len(train_dataset), 42242)
for batch in tqdm(loader):
    batch.to(device)
    best = fwd_return_best(batch["input_ids"], batch["attention_mask"]).squeeze()
    best.backward()
    batch_grads = []
    for layer in layers:
        for param in layer.parameters():
            batch_grads.append(param.grad)
    all_grads[batch["id"].item()] = torch.cat(
        [x for x in batch_grads if len(x.shape) == 1]
    )
# 17:45 mins, 22Gb mem
# %%
grads_npy = all_grads.numpy()
# %%
import numpy as np

# %%
np.save("./data/fullsize_dense_grads", grads_npy)  # 11Gb
# %%
