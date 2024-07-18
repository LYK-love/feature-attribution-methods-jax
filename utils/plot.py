import numpy as np
import matplotlib.pyplot as plt


def plot_image(img: np.ndarray, title: str = "Image", ax=None):
    """
    Plots a np image array.

    Args:
        img (np.ndarray): The image array to plot.
        title (str): Title of the plot.
        ax (matplotlib.axes._subplots.AxesSubplot, optional): Axes object to plot on. If None, create a new figure.
    """
    img = np.array(img)  # Convert JAX array to NumPy array for plotting
    if ax is None:
        plt.imshow(img)
        plt.title(title)
        plt.axis("off")
        plt.show()
    else:
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")


def plot_all_images(cam_image, attributions, attributions_guided, guided_cam_image):
    """
    Plots all four images in a 2x2 grid.

    Args:
        cam_image (np.ndarray): The CAM image.
        attributions (np.ndarray): The Attribution Map.
        attributions_guided (np.ndarray): The Guided Attribution Map.
        guided_cam_image (np.ndarray): The Guided Grad-CAM image.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    plot_image(cam_image, title="Grad-CAM", ax=axes[0, 0])
    plot_image(attributions, title="Attribution Map", ax=axes[0, 1])
    plot_image(attributions_guided, title="Guided Attribution Map", ax=axes[1, 0])
    plot_image(guided_cam_image, title="Guided Grad-CAM", ax=axes[1, 1])

    plt.tight_layout()
    plt.show()
