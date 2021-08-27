import torch


def encode(examples, tokenizer):
    return tokenizer(examples["sentence"], truncation=True)


def make_input(text, tokenizer):
    return torch.tensor([tokenizer.encode(text)])


def make_input_batch(batch_text, tokenizer, padding=True):
    tokenized = tokenizer.batch_encode_plus(batch_text, padding=padding)
    return torch.tensor(tokenized["input_ids"]), torch.tensor(
        tokenized["attention_mask"]
    )


def get_embeddings(text, model, tokenizer, padding=True):
    return model.distilbert.embeddings(
        make_input_batch(text, tokenizer, padding=padding)
    )


def embeddings_forward_fn(embs, model, attention_mask=None):
    pred = model(inputs_embeds=embs, attention_mask=attention_mask)
    return pred.logits


def dense_to_topk_sparse(tensor, k):
    # Dims are examples x tokens x features
    # Sum over tokens
    summed = torch.sum(tensor, dim=1)
    # Find top k values for each example
    topk = torch.topk(summed, k=k, dim=1)
    # Transform indices for sparse representation
    indices = torch.tensor(
        [[i, x.item()] for i in range(len(topk.indices)) for x in topk.indices[i]]
    ).T
    values = topk.values.flatten()
    # Create sparse tensor
    sparse = torch.sparse_coo_tensor(indices, values, summed.shape)
    return sparse


def pad_to_equal_length(x, y):
    diff = len(x) - len(y)
    if diff > 0:
        y = torch.nn.functional.pad(y, (0, diff))
    if diff < 0:
        x = torch.nn.functional.pad(x, (0, -diff))
    return x, y
