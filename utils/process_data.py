import torch
from uuid import uuid4
from datasets import load_dataset


def encode(examples, tokenizer):
    return tokenizer(examples["sentence"], truncation=True)


def encode_with_hash(examples, tokenizer):
    tokenizer_features = tokenizer(examples["sentence"], truncation=True)
    # uuid = {"uuid": [str(uuid4()) for x in examples["idx"]]}
    id = {"id": [torch.tensor([x]) for x in examples["idx"]]}
    return {**tokenizer_features, **id}
    # return tokenizer_features


def make_input(batch_text, tokenizer, padding=True):
    tokenized = tokenizer.batch_encode_plus(batch_text, padding=padding)
    return torch.tensor(tokenized["input_ids"]), torch.tensor(
        tokenized["attention_mask"]
    )


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


def get_sst2(tokenizer):
    train_dataset = load_dataset("glue", "sst2", split="train")
    test_dataset = load_dataset("glue", "sst2", split="test")
    train_dataset_tokenized = train_dataset.map(
        encode_with_hash,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer},
        load_from_cache_file=True,
        cache_file_name="./data/sst2_train_dataset_cache",
    )
    train_dataset_tokenized.set_format(
        "torch", columns=["input_ids", "attention_mask", "label", "id"]
    )
    test_dataset_tokenized = test_dataset.map(
        encode_with_hash,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer},
        load_from_cache_file=True,
        cache_file_name="./data/sst2_test_dataset_cache",
    )
    test_dataset_tokenized.set_format(
        "torch", columns=["input_ids", "attention_mask", "label", "id"]
    )
    return train_dataset_tokenized, test_dataset_tokenized
