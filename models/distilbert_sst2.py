import functools

import torch
from captum.attr import IntegratedGradients, LayerConductance, LayerIntegratedGradients
from captum._utils.gradient import compute_layer_gradients_and_eval


def make_input(text, tokenizer):
    return torch.tensor([tokenizer.encode(text)])


def get_embeddings(text, model, tokenizer):
    return model.distilbert.embeddings(make_input(text, tokenizer))


def embeddings_forward_fn(embs, model, target=0, attention_mask=None):
    pred = model(inputs_embeds=embs, attention_mask=attention_mask)
    return pred.logits[:, target]


def get_single_layer_conductance(inputs, target, model, tokenizer, layer_index):
    layers = model.distilbert.transformer.layer
    fwd_fn = functools.partial(embeddings_forward_fn, model=model, target=target)
    layer_conductance = LayerConductance(fwd_fn, layers[layer_index])
    input_embeddings = get_embeddings(inputs, model, tokenizer)
    baseline = torch.zeros_like(input_embeddings)
    attrs = layer_conductance.attribute(input_embeddings, baselines=baseline)
    return attrs


def get_all_layer_conductance(inputs, target, model, tokenizer):
    layers = model.distilbert.transformer.layer
    fwd_fn = functools.partial(embeddings_forward_fn, model=model, target=target)
    input_embeddings = get_embeddings(inputs, model, tokenizer)
    baseline = torch.zeros_like(input_embeddings)
    output = []
    for layer in layers:
        layer_conductance = LayerConductance(fwd_fn, layer)
        attrs = layer_conductance.attribute(input_embeddings, baselines=baseline)
        output.append(attrs)
    return output


def get_single_layer_gradients(inputs, target, model, tokenizer, layer_index):
    layers = model.distilbert.transformer.layer
    gradients, output = compute_layer_gradients_and_eval(
        embeddings_forward_fn,
        layers[layer_index],
        inputs=get_embeddings(inputs, model, tokenizer),
        additional_forward_args=(model, target),
    )
    return gradients


def get_all_layer_gradients(inputs, target, model, tokenizer):
    layers = model.distilbert.transformer.layer
    all_grads = []
    for layer in layers:
        gradients, output = compute_layer_gradients_and_eval(
            embeddings_forward_fn,
            layer,
            inputs=get_embeddings(inputs, model, tokenizer),
            additional_forward_args=(model, target),
        )
        all_grads.append(gradients[0])
    return all_grads


def get_all_layer_integrated_gradients(inputs, target, model, tokenizer):
    layers = model.distilbert.transformer.layer
    fwd_fn = functools.partial(embeddings_forward_fn, model=model, target=target)
    input_embeddings = get_embeddings(inputs, model, tokenizer)
    baseline = torch.zeros_like(input_embeddings)
    output = []
    for layer in layers:
        layer_integrated_grads = LayerIntegratedGradients(
            fwd_fn, layer, multiply_by_inputs=False
        )
        attrs = layer_integrated_grads.attribute(input_embeddings, baselines=baseline)
        output.append(attrs)
    return output


if __name__ == "__main__":
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    cg = Chronograph()

    cg.start("Load Tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english"
    )

    cg.split("Load Model")
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english"
    )
    model.eval()
    cg.split("Layer Conductance")
    test_input = "absolutely tremendous"
    attrs = get_all_layer_conductance(test_input, 1, model, tokenizer)
    cg.split("Layer Integrated Gradients")
    attrs_2 = get_all_layer_integrated_gradients(test_input, 1, model, tokenizer)
    cg.split("Gradients Only")
    grads = get_all_layer_gradients(test_input, 1, model, tokenizer)
    cg.stop()
    print(cg.report())

"""
TIMINGS (from Tachymeter):
    Load Tokenizer: 3.84 sec
    Load Model: 1.5 sec
    Layer Conductance: 794 msec
    Layer Integrated Gradients: 951 msec
    Gradients Only: 157 msec
"""
