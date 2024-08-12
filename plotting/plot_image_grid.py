from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def plot_images_grid(images: list, cols: int, save_path: str, figsize=(10, 10), dpi=500):
    """
    Plot a grid of images in a single plot, adjusting to the number of images.

    Parameters:
    - images: List of file paths to the images or PIL Image objects
    - cols: Number of columns in the grid
    - figsize: Size of the figure (width, height)
    """
    if not images:
        raise ValueError("Number of images must > 0")

    num_images = len(images)
    rows = num_images // cols

    if num_images % cols > 0:
        rows += 1
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()  
    
    for i in range(num_images, len(axes)):
        axes[i].axis('off')

    for ax, image_path in zip(axes, images):
        if isinstance(image_path, str):
            image = Image.open(image_path)
        elif isinstance(image_path, Image.Image):
            image = image_path
        else:
            raise TypeError("Images should be file paths or PIL Image objects")

        ax.imshow(np.array(image))
        ax.axis('off') 

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
