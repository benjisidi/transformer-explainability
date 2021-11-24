import torch
from captum._utils.gradient import compute_layer_gradients_and_eval
from captum.attr import LayerIntegratedGradients
from tqdm import tqdm

from .process_data import dense_to_topk_sparse


def get_layer_gradients(
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


def get_layer_integrated_gradients(
    inputs, mask, layers, fwd, fwd_args=(), device="cpu", target=None
):
    if len(inputs.shape) == 1:
        inputs = inputs.unsqueeze(0)
    baseline = torch.zeros_like(inputs)
    baseline.to(device)
    output = []
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
