import torch
from captum._utils.gradient import compute_layer_gradients_and_eval
from captum.attr import LayerIntegratedGradients, LayerGradientXActivation
from tqdm import tqdm

from .process_data import dense_to_topk_sparse, get_total_features


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
    num_features = sum([x.out_features for x in layers])
    all_grads = torch.zeros(len(loader.dataset), num_features)
    # for layer in layers:
    #     layer_grads = []
    for batch in tqdm(loader):
        batch.to(device)
        batch_grads, _ = compute_layer_gradients_and_eval(
            fwd,
            layers,
            inputs=batch["input_ids"],
            additional_forward_args=(batch["attention_mask"]),
            target_ind=batch["label"],
        )
        # if sparse:
        #     layer_grads.append(dense_to_topk_sparse(batch_grads[0], k=k))
        # else:
        #     batch_grads = batch_grads[0].sum(dim=1)
        #     layer_grads.append(batch_grads.cpu())

        # batch_grads is [(batch_size x n_tokens x params),...]
        batch_grads = [x[0] for x in batch_grads]
        batch_grads = [
            torch.mean(x, dim=1) if len(x.shape) == 3 else x for x in batch_grads
        ]
        batch_grads = torch.cat(batch_grads, dim=1)

        all_grads[batch["id"].squeeze()] = batch_grads.cpu()

    return all_grads


def get_layer_gradients_independent(
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
    num_features = get_total_features(layers)
    all_grads = torch.zeros(len(loader.dataset), num_features)
    # for layer in layers:
    #     layer_grads = []
    for batch in tqdm(loader):
        batch.to(device)
        batch_output = []
        for layer in layers:
            batch_grads, _ = compute_layer_gradients_and_eval(
                fwd,
                layer,
                inputs=batch["input_ids"],
                additional_forward_args=(batch["attention_mask"]),
                target_ind=batch["label"],
            )
            batch_output.append(batch_grads)
        # if sparse:
        #     layer_grads.append(dense_to_topk_sparse(batch_grads[0], k=k))
        # else:
        #     batch_grads = batch_grads[0].sum(dim=1)
        #     layer_grads.append(batch_grads.cpu())

        # batch_output is [(batch_size x n_tokens x params),...]
        batch_output = [x[0] for x in batch_output]
        batch_output = [
            torch.mean(x, dim=1) if len(x.shape) == 3 else x for x in batch_output
        ]
        batch_output = torch.cat(batch_output, dim=1)

        all_grads[batch["id"].squeeze()] = batch_output.cpu()

    return all_grads


# def get_layer_output_gradients(loader, layers, fwd, device="cpu"):
#     """
#     Params:
#         loader: dataloader that returns batches of (inputs, attn_masks, labels)
#         layers: list of model layers
#         fwd: forward function for captum
#     """
#     all_grads = torch.zeros(len(loader), 42242)
#     for batch in tqdm(loader):
#         batch.to(device)
#         best = fwd(batch["input_ids"], batch["attention_mask"]).squeeze()
#         best.backward()
#         batch_grads = []
#         for layer in layers:
#             for param in layer.parameters():
#                 batch_grads.append(param.grad)
#         # layer outputs have 1 dim
#         all_grads[batch["id"].item()] = torch.cat(
#             [x for x in batch_grads if len(x.shape) == 1]
#         )
#     return all_grads


def get_layer_integrated_gradients(
    inputs,
    mask,
    layers,
    fwd,
    fwd_args=(),
    device="cpu",
    target=None,
    attr_to_inputs=False,
):
    if len(inputs.shape) == 1:
        inputs = inputs.unsqueeze(0)
    baseline = torch.zeros_like(inputs)
    inputs = inputs.to(device)
    baseline = baseline.to(device)
    mask = mask.to(device)
    target = target.to(device)
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
            attribute_to_layer_input=attr_to_inputs,
        )
        output.append(layer_attrs.cpu())
    output = [
        x.squeeze().mean(dim=0) if len(x.shape) == 3 else x.squeeze() for x in output
    ]
    return torch.cat(output)


def get_embeddings(input_loader, model, embed_dim):
    print("Calculating embeddings...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    all_embeddings = torch.zeros(len(input_loader.dataset), embed_dim)
    embeddings = model.get_input_embeddings()
    for batch in tqdm(input_loader):
        batch_embeddings = embeddings(batch["input_ids"].to(device)).detach()
        all_embeddings[batch["id"].squeeze().cpu()] = torch.mean(
            batch_embeddings, dim=1
        ).cpu()
    return all_embeddings
