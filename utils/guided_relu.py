import jax
from jax import custom_vjp
import jax.numpy as jnp


@custom_vjp
def guided_relu(x):
    return jax.nn.relu(x)


def guided_relu_fwd(x):
    residual = x
    primal = guided_relu(x)

    return primal, residual


def guided_relu_bwd(residual, grad):
    # Derivative of relu is 1 for values > 0 and 0 otherwise
    # Using relu devivative for both residuals and grads
    grad_gate = jnp.float32(grad > 0)
    residual_gate = jnp.float32(residual > 0)
    output = residual_gate * grad_gate * grad
    return (output,)


guided_relu.defvjp(guided_relu_fwd, guided_relu_bwd)
