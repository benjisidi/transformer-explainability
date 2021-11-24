import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification


def get_distilbert_finetuned():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english"
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english"
    )
    model.eval()
    model.zero_grad()
    model.to(device)
    layers = model.distilbert.transformer.layer
    return model, tokenizer, layers
