import numpy as np
import jax.numpy as jnp


def get_predictions(logits):
    return jnp.argmax(logits, axis=1)


def get_accuracy(predictions, Y):
    return jnp.mean(predictions == Y)
