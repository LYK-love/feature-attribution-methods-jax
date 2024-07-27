import copy
import jax
from jax import grad
from jax import numpy as jnp
from flax.core.frozen_dict import freeze

SIGN = -1


def get_layer_activations(state, layer_idx: int = 4, within_block=False):
    key = f"gradcam_sow_{layer_idx}"
    intermediates = state["intermediates"]

    if within_block:
        assert len(intermediates) == 1, "There should be only one intermediate"
        intermediates = next(iter(intermediates.values()))

    layer_activations = intermediates[key][0]
    return layer_activations


def get_layer_gradients(
    model,
    variables,
    layer_activations,
    input_tensor,
    target,
    loss_fn,
    layer_idx: int = 4,
    within_block=False,
):
    key = f"gradcam_perturb_{layer_idx}"
    perturbations = copy.deepcopy(variables["perturbations"])

    layer_perturbations = {
        key: layer_activations
    }  # This is the perturbation that we will use to extract the gradients.

    if within_block:
        assert len(perturbations) == 1, "There should be only one perturbation"
        key_of_block = next(iter(perturbations.keys()))

    # Manually reconstruct the dictionary, ensuring the specific key is frozen
    if within_block:
        block_perturbations = perturbations[key_of_block]
        block_perturbations.update(layer_perturbations)
        perturbations.update({key_of_block: block_perturbations})

    else:
        perturbations.update(layer_perturbations)

    perturbations = freeze(perturbations)

    layer_grads = grad(loss_fn, argnums=3)(
        model,
        variables["params"],
        variables["batch_stats"],
        perturbations,
        input_tensor,
        target,
    )

    # Here I use the negated gradients w.r.t. the classificaiton loss to get the positive contributions.
    if within_block:
        layer_grads = SIGN * layer_grads[key_of_block][key]
    else:
        layer_grads = SIGN * layer_grads[key]

    # variables["perturbations"] = {
    #     key: jnp.zeros_like(value) for key, value in perturbations.items()
    # }
    return layer_grads


def get_layer_weights(layer_grads):
    # Get weights using global average pooling
    layer_weights = jnp.mean(layer_grads, axis=(1, 2))  # (B, H', W', C') -> (B, C')
    return layer_weights


def get_layer_grad_cam(layer_weights, layer_activations):
    # Get the weighted sum of all the filters
    weighted_activations = (
        layer_weights[:, None, None, :] * layer_activations
    )  # (B, H', W', C')

    layer_grad_cam = jnp.sum(
        weighted_activations, axis=-1, keepdims=True
    )  # (B, H', W', C') --> (B, H', W', 1). Now it's gray scale image
    layer_grad_cam = jnp.maximum(
        layer_grad_cam, 0
    )  # They take only the positive contributions by applying a RELU function:
    return layer_grad_cam


def aggregate_scaled_grad_cams(scaled_layer_grad_cams):
    # Concatenate the list of arrays along the last axis
    concatenated = jnp.concatenate(scaled_layer_grad_cams, axis=-1)  # (B, H, W, K)

    concatenated = jnp.maximum(concatenated, 0)  # Apply ReLU

    # Average the concatenated array along the new channel axis (last axis)
    averaged = jnp.mean(
        concatenated, axis=-1, keepdims=True
    )  # (B, H, W, K) -> (B, H, W, 1)

    return averaged
