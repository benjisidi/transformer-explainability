import torch


def make_input(text, tokenizer):
    return torch.tensor([tokenizer.encode(text)])


def make_input_batch(batch_text, tokenizer, padding=True):
    return torch.tensor(
        tokenizer.batch_encode_plus(batch_text, padding=padding)["input_ids"]
    )


def get_embeddings(text, model, tokenizer, padding=True):
    return model.distilbert.embeddings(
        make_input_batch(text, tokenizer, padding=padding)
    )


def embeddings_forward_fn(embs, model, attention_mask=None):
    pred = model(inputs_embeds=embs, attention_mask=attention_mask)
    return pred.logits


def dense_to_topk_sparse(tensor, k):
    # Note: In the case of many repeated values, will retain more than k elements.
    # Note: This could be fixed by using indices instead but it's proving tricky so
    # Note: I'm leaving that for another time.
    flat = tensor.flatten()
    topk = torch.topk(flat, k)
    sparse = torch.where(
        tensor < topk.values[-1], torch.tensor(0, dtype=torch.float), tensor
    ).to_sparse()
    return sparse
