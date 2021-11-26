import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .util import get_nested_property

layers_of_interest = [
    "distilbert.embeddings.word_embeddings",
    "distilbert.transformer.layer.0.attention.q_lin",
    "distilbert.transformer.layer.0.attention.v_lin",
    "distilbert.transformer.layer.0.attention.out_lin",
    "distilbert.transformer.layer.0.ffn.lin1",
    "distilbert.transformer.layer.0.ffn.lin2",
    "distilbert.transformer.layer.1.attention.q_lin",
    "distilbert.transformer.layer.1.attention.v_lin",
    "distilbert.transformer.layer.1.attention.out_lin",
    "distilbert.transformer.layer.1.ffn.lin1",
    "distilbert.transformer.layer.1.ffn.lin2",
    "distilbert.transformer.layer.2.attention.q_lin",
    "distilbert.transformer.layer.2.attention.v_lin",
    "distilbert.transformer.layer.2.attention.out_lin",
    "distilbert.transformer.layer.2.ffn.lin1",
    "distilbert.transformer.layer.2.ffn.lin2",
    "distilbert.transformer.layer.3.attention.q_lin",
    "distilbert.transformer.layer.3.attention.v_lin",
    "distilbert.transformer.layer.3.attention.out_lin",
    "distilbert.transformer.layer.3.ffn.lin1",
    "distilbert.transformer.layer.3.ffn.lin2",
    "distilbert.transformer.layer.4.attention.q_lin",
    "distilbert.transformer.layer.4.attention.v_lin",
    "distilbert.transformer.layer.4.attention.out_lin",
    "distilbert.transformer.layer.4.ffn.lin1",
    "distilbert.transformer.layer.4.ffn.lin2",
    "distilbert.transformer.layer.5.attention.q_lin",
    "distilbert.transformer.layer.5.attention.v_lin",
    "distilbert.transformer.layer.5.attention.out_lin",
    "distilbert.transformer.layer.5.ffn.lin1",
    "distilbert.transformer.layer.5.ffn.lin2",
    "pre_classifier",
]


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
    layers = [get_nested_property(model, x) for x in layers_of_interest]
    return model, tokenizer, layers
