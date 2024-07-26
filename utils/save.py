import os
import matplotlib.pyplot as plt


def save_image(image, title=None, save_dir=None, save_format="png"):
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        filename = f"{title}.{save_format}" if title else f"image.{save_format}"
        save_path = os.path.join(save_dir, filename)
        plt.imsave(save_path, image)
