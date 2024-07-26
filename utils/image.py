from functools import partial
from PIL import Image
import jax
import numpy as np
import jax.numpy as jnp


def read_and_normalize_image(image_path):
    # Open the image file
    img = Image.open(image_path).convert("RGB")  # Ensure it is in RGB format
    # Convert the image to a NumPy array
    img_array = np.array(img, dtype=np.float32)
    # Normalize the image to the range [0, 1]
    img_normalized = img_array / 255.0
    return jnp.array(img_normalized)


def preprocess_image(
    img: jnp.ndarray, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
) -> jnp.ndarray:
    """
    Preprocesses the image by normalizing it with the given mean and standard deviation.

    Args:
        img (jnp.ndarray): The image as a JAX array.
        mean (list): The mean for normalization.
        std (list): The standard deviation for normalization.

    Returns:
        jnp.ndarray: The preprocessed image.
    """
    mean = jnp.array(mean).reshape(1, 1, 3)
    std = jnp.array(std).reshape(1, 1, 3)
    img = (img - mean) / std
    img = jnp.clip(img, 0.0, 1.0)
    return img


def deprocess_image(img):
    """see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65"""
    img = img - jnp.mean(img)
    img = img / (jnp.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = jnp.clip(img, 0, 1)
    return img


@partial(jax.vmap, in_axes=(0, None), out_axes=0)
def scale_cam_image(img: jnp.ndarray, target_shape) -> jnp.ndarray:
    img = img - jnp.min(img)
    img = img / (1e-7 + jnp.max(img))
    img = jax.image.resize(img, target_shape, method="linear")
    return img


# def scale_cam_image(cam, target_size=None):
#     result = []
#     for img in cam:
#         img = img - np.min(img)
#         img = img / (1e-7 + np.max(img))
#         if target_size is not None:
#             if len(img.shape) > 3:
#                 img = zoom(
#                     np.float32(img),
#                     [(t_s / i_s) for i_s, t_s in zip(img.shape, target_size[::-1])],
#                 )
#             else:
#                 img = cv2.resize(np.float32(img), target_size)

#         result.append(img)
#     result = np.float32(result)

#     return result


# def show_cam_on_image(
#     img: np.ndarray,
#     mask: np.ndarray,
#     use_rgb: bool = False,
#     colormap: int = cv2.COLORMAP_JET,
#     image_weight: float = 0.5,
# ) -> np.ndarray:
#     """This function overlays the cam mask on the image as an heatmap.
#     By default the heatmap is in BGR format.

#     :param img: The base image in RGB or BGR format.
#     :param mask: The cam mask.
#     :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
#     :param colormap: The OpenCV colormap to be used.
#     :param image_weight: The final result is image_weight * img + (1-image_weight) * mask.
#     :returns: The default image with the cam overlay.
#     """
#     heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
#     if use_rgb:
#         heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
#     heatmap = np.float32(heatmap) / 255

#     if np.max(img) > 1:
#         raise Exception("The input image should np.float32 in the range [0, 1]")

#     if image_weight < 0 or image_weight > 1:
#         raise Exception(
#             f"image_weight should be in the range [0, 1].\
#                 Got: {image_weight}"
#         )

#     cam = (1 - image_weight) * heatmap + image_weight * img  # RGB headmap + RGB image
#     cam = cam / np.max(cam)
#     return np.uint8(255 * cam)


def show_cam_on_image(
    img: jnp.ndarray, mask: jnp.ndarray, image_weight: float
) -> jnp.ndarray:
    """This function overlays the cam mask on the image as a heatmap.
    The heatmap is in RGB format.

    :param img: The base image in RGB format.
    :param mask: The cam mask.
    :param image_weight: The final result is image_weight * img + (1-image_weight) * mask.
    :returns: The image with the cam overlay.
    """
    # Ensure the mask is in the range [0, 1]
    mask = jnp.clip(mask, 0, 1)  # (H, W, 1)

    # Apply smoother JET colormap to the mask
    heatmap = smooth_jet_colormap(mask)  # (H, W, 3)

    # Blend images
    cam = (1 - image_weight) * heatmap + image_weight * img  # RGB heatmap + RGB image
    cam = cam / jnp.max(cam)
    return cam


def smooth_jet_colormap(value):
    """Applies a smoother JET colormap to a grayscale value."""
    value = jnp.clip(value, 0, 1)  # Ensure values are within [0, 1]
    value = value[..., 0]  # Remove the singleton dimension
    c = jnp.zeros((*value.shape, 3))

    # Blue to cyan
    c = c.at[..., 2].set(jnp.clip(1.5 - jnp.abs(4.0 * value - 1.5), 0.0, 1.0))
    # Cyan to yellow
    c = c.at[..., 1].set(jnp.clip(1.5 - jnp.abs(4.0 * value - 2.5), 0.0, 1.0))
    # Yellow to red
    c = c.at[..., 0].set(jnp.clip(1.5 - jnp.abs(4.0 * value - 3.5), 0.0, 1.0))

    return c
