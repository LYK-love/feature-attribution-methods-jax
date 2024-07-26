import numpy as np
import matplotlib.pyplot as plt


def plot_image(image, title=None, figsize=(5, 5)):
    plt.figure(figsize=figsize)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.axis("off")
    plt.show()
