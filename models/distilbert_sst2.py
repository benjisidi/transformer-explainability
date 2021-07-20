import functools

import torch
from captum.attr import IntegratedGradients, LayerConductance, LayerIntegratedGradients
from captum._utils.gradient import compute_layer_gradients_and_eval


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


# ToDo: remove target as it is now unneeded
def embeddings_forward_fn(embs, model, target=0, attention_mask=None):
    pred = model(inputs_embeds=embs, attention_mask=attention_mask)
    return pred.logits


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


def get_all_layer_gradients_batch(inputs, targets, model, tokenizer):
    layers = model.distilbert.transformer.layer
    all_grads = []
    inputs = get_embeddings(inputs, model, tokenizer)
    for layer in layers:
        gradients, _ = compute_layer_gradients_and_eval(
            embeddings_forward_fn,
            layer,
            inputs=inputs,
            additional_forward_args=(model, targets),
            target_ind=targets
        )
        all_grads.append(gradients)
    return all_grads


def get_all_layer_integrated_gradients_batch(inputs, target, model, tokenizer):
    layers = model.distilbert.transformer.layer
    fwd_fn = functools.partial(embeddings_forward_fn, model=model, target=target)
    input_embeddings = get_embeddings(inputs, model, tokenizer)
    baseline = torch.zeros_like(input_embeddings)
    output = []
    for layer in layers:
        layer_integrated_grads = LayerIntegratedGradients(
            fwd_fn, layer, multiply_by_inputs=False
        )
        attrs = layer_integrated_grads.attribute(input_embeddings, baselines=baseline, target=target)
        output.append(attrs)
    return output

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

    tokenizer = AutoTokenizer.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english"
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english"
    )
    model.eval()

    test_input = "absolutely tremendous"
    attrs = get_all_layer_conductance(test_input, 1, model, tokenizer)
    attrs_2 = get_all_layer_integrated_gradients(test_input, 1, model, tokenizer)
    grads = get_all_layer_gradients(test_input, 1, model, tokenizer)

"""
TIMINGS (from Tachymeter):
    Load Tokenizer: 3.84 sec
    Load Model: 1.5 sec
    Layer Conductance: 794 msec
    Layer Integrated Gradients: 951 msec
    Gradients Only: 157 msec
"""
