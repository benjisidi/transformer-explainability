from .process_data import get_embeddings, embeddings_forward_fn
from captum._utils.gradient import compute_layer_gradients_and_eval
from captum.attr import LayerIntegratedGradients
import functools
import torch
from .process_data import dense_to_topk_sparse
from tqdm import tqdm


def get_all_layer_gradients_2(
    loader, layers, fwd, fwd_args=(), sparse=True, k=70, device="cpu"
):
    """
    Params:
        loader: dataloader that returns batches of (inputs, attn_masks, labels)
        layers: list of model layers
        fwd: forward function for captum
        fwd_args: any additional arguments for the fwd function
        sparse: whether to save grads in sparse format
        k: if sparse, how many gradients to save for each layer
    """
    all_grads = []
    for layer in layers:
        layer_grads = []
        for batch in tqdm(loader):
            batch.to(device)
            batch_grads, _ = compute_layer_gradients_and_eval(
                fwd,
                layer,
                inputs=batch["input_ids"],
                additional_forward_args=(batch["attention_mask"]),
                target_ind=batch["label"],
            )
            if sparse:
                layer_grads.append(dense_to_topk_sparse(batch_grads[0], k=k))
            else:
                batch_grads = batch_grads[0].sum(dim=1)
                layer_grads.append(batch_grads.cpu())
        all_grads.append(torch.cat(layer_grads))
    return torch.stack(all_grads)


def get_all_layer_gradients(
    inputs, targets, model, tokenizer, batch_size=30, sparse=False
):
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
        # outputs = embeddings_forward_fn(inputs, model) ---> Maybe don't use embeddings and use proper forward if possible?
        # outputs_selected = torch.gather(outputs, 0, torch.tensor(targets).unsqueeze(1))
        # grads = torch.grad.autograd(torch.unbind(outputs), inputs, only_inputs=False)
        # This only seems to return grads wrt the input. Might need to get all layer outputs (which is what captum does).
        # Not ideal - is this just many backwards passes through?
        # Should post an update asap.
        if sparse:
            all_grads.append(dense_to_topk_sparse(gradients[0], k=70))
        else:
            all_grads.append(gradients[0])
    return all_grads


def get_all_layer_integrated_gradients(
    inputs, mask, target, layers, fwd, fwd_args=(), device="cpu"
):
    if len(inputs.shape) == 1:
        inputs = inputs.unsqueeze(0)
    baseline = torch.zeros_like(inputs)
    output = []
    inputs.to(device)
    for layer in layers:
        layer_integrated_grads = LayerIntegratedGradients(
            fwd, layer, multiply_by_inputs=False
        )
        layer_attrs = layer_integrated_grads.attribute(
            inputs,
            baselines=baseline,
            target=target,
            additional_forward_args=[mask, *fwd_args] if fwd_args else mask,
        )
        output.append(layer_attrs.cpu())
    return torch.stack(output)
