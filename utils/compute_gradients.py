from .process_data import get_embeddings, embeddings_forward_fn
from captum._utils.gradient import compute_layer_gradients_and_eval
from captum.attr import LayerIntegratedGradients
import functools
import torch


def get_all_layer_gradients(inputs, targets, model, tokenizer):
    layers = model.distilbert.transformer.layer
    all_grads = []
    inputs = get_embeddings(inputs, model, tokenizer)
    for layer in layers:
        gradients, _ = compute_layer_gradients_and_eval(
            embeddings_forward_fn,
            layer,
            inputs=inputs,
            additional_forward_args=(model),
            target_ind=targets,
        )
        all_grads.append(gradients[0])
    return all_grads


def get_all_layer_integrated_gradients(inputs, target, model, tokenizer):
    layers = model.distilbert.transformer.layer
    fwd_fn = functools.partial(embeddings_forward_fn, model=model)
    input_embeddings = get_embeddings(inputs, model, tokenizer)
    baseline = torch.zeros_like(input_embeddings)
    output = []
    for layer in layers:
        layer_integrated_grads = LayerIntegratedGradients(
            fwd_fn, layer, multiply_by_inputs=False
        )
        attrs = layer_integrated_grads.attribute(
            input_embeddings, baselines=baseline, target=target
        )
        output.append(attrs)
    return output
