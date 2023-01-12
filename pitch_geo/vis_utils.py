"""
Utils for visualizing the keypoints.
"""

from pathlib import Path
from typing import Union

from matplotlib import image
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from pitch_geo.constants import DATA_FOLDER, GRAPHS_FOLDER


def visualize_keypoints(
    image_path: Union[Path, str],
    df: pd.DataFrame,
    dot_radius: float = 20.0,
    images_base_path=DATA_FOLDER,
):
    """
    Plot the image from the given filepath and its corresponding keypoints.
    Args:
        image_path: A path to the image. It might be a full path to the image or the path relative to DATA_FOLDER.
        df: A pandas DataFrame with keypoints annotations. It must contain columns: `x`, `y`, 'kid'
        dot_radius: A size of a dot plotted for every keypoint.

    Returns:

    """
    images_base_path = Path(images_base_path)
    image_full_path = images_base_path / image_path

    img = image.imread(image_full_path)
    fig, ax = plt.subplots(1, figsize=(12, 6.75))  # 16:9 aspect ratio

    # Show the image
    ax.imshow(img)

    # Draw the keypoints

    keypoints = df[df["image_path"] == image_path]

    x_offset = 0.5 * dot_radius  # offset between keypoint and text
    y_offset = -0.5 * dot_radius
    for _, row in keypoints.iterrows():
        x = row["x"]
        y = row["y"]
        vis = row["vis"]
        kid = row["kid"]

        # If a keypoint is not visible, then its coordinates are zeroed out. Don't plot them
        if vis != 0:
            circ = Circle(xy=(x, y), radius=dot_radius)
            ax.add_patch(circ)
            ax.text(x=x + x_offset, y=y + y_offset, s=kid)

    plt.axis("off")


def show_image_with_annotations(
    img: np.ndarray,
    keypoints: np.ndarray,
    labels,
    dot_radius: float = 2.0,
    normalized: bool = True,
    vis: bool = False,
):

    fig, ax = plt.subplots(1)
    plot_image_with_annotations_on_ax(
        ax, img, keypoints, labels, dot_radius, normalized, vis
    )
    plt.axis("off")
    plt.show()


def plot_image_with_annotations_on_ax(
    ax,
    img: np.ndarray,
    keypoints: np.ndarray,
    labels,
    dot_radius: float = 2.0,
    normalized: bool = True,
    vis: bool = False,
):
    # Show the image
    ax.imshow(img)

    # Draw keypoints

    x_offset = 0.5 * dot_radius  # offset between keypoint and text
    y_offset = -0.5 * dot_radius

    if vis:
        keypoints = keypoints.reshape(34, 3).copy()
    else:
        keypoints = keypoints.reshape(34, 2).copy()
    if normalized:
        keypoints[:, 0] *= img.shape[1]
        keypoints[:, 1] *= img.shape[0]

    for kid, point in enumerate(keypoints):
        if vis:
            x, y, v = point
            if v > 0.1:
                color = "orange"
                circ = Circle(xy=(x, y), radius=dot_radius, color=color)
                ax.add_patch(circ)
                ax.text(x=x + x_offset, y=y + y_offset, s=f"{labels[kid]}")
            # else:
            #     color = 'teal'

        else:
            x, y = point
            circ = Circle(xy=(x, y), radius=dot_radius)
            ax.add_patch(circ)
            ax.text(x=x + x_offset, y=y + y_offset, s=f"{labels[kid]}")


def show_field_with_keypoint_frequency(df, max_dot_size=1000):
    keypoints = {
        1: (35, 23),
        2: (355, 23),
        22: (675, 23),
        5: (35, 106),
        11: (35, 175),
        14: (35, 285),
        10: (35, 350),
        3: (35, 435),
        12: (65, 175),
        13: (65, 285),
        15: (100, 230),
        6: (135, 106),
        7: (135, 182),
        8: (135, 268),
        9: (135, 350),
        16: (355, 175),
        17: (355, 230),
        18: (355, 285),
        19: (155, 230),
        4: (355, 435),
        20: (300, 230),
        21: (410, 230),
        25: (675, 106),
        31: (675, 175),
        34: (675, 285),
        30: (675, 350),
        23: (675, 435),
        26: (575, 106),
        27: (575, 182),
        28: (575, 285),
        29: (575, 350),
        39: (553, 230),
        32: (643, 175),
        33: (643, 285),
        35: (610, 230),
    }

    fig, ax = plt.subplots(1, 2, figsize=(16, 16))

    pitch_img = image.imread(GRAPHS_FOLDER / "pitch.png")
    ax[0].imshow(pitch_img)
    ax[1].imshow(pitch_img)

    keypoint_counts = df[df["vis"] != 0]["kid"].value_counts()
    n_of_images = df["image_path"].nunique()

    for kid, count in keypoint_counts.items():
        freq = count / n_of_images
        size = freq * max_dot_size
        # print(kid, type(kid), kid in keypoints)
        try:
            x, y = keypoints[kid]
        except KeyError:
            pass
        else:
            ax[1].scatter(x, y, s=size, c="orange", alpha=0.8)
            fontsize = "medium" if freq > 0.3 else "small"
            ax[1].text(x, y, s=f"{freq:0.0%}", alpha=0.8, fontsize=fontsize)

    ax[0].axis("off")
    ax[1].axis("off")
    ax[1].set_title("Location of each keypoint")
    ax[1].set_title("# times a given keypoint is visible / # frames")
    plt.show()
