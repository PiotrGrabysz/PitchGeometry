from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import typer

from pitch_geo import vis_utils


def show_keypoints(keypoints_filepath: str, images_dir: str, output_dir: str) -> None:
    """
    Plot keypoints on images.

    Args:

        keypoints_filepath: Filepath to the file containing a data frame with keypoints.

        images_dir: Folder containing images that keypoints file refers to.

        output_dir: Folder where the processed images will be saved.
    """

    keypoints_df = pd.read_csv(keypoints_filepath)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    relative_image_paths = keypoints_df["image_path"].unique()
    for img_path in relative_image_paths:
        print(img_path)
        vis_utils.visualize_keypoints(
            image_path=img_path, df=keypoints_df, images_base_path=images_dir
        )
        plt.savefig(fname=(output_dir / Path(img_path).name))
        plt.close()


if __name__ == "__main__":
    typer.run(show_keypoints)
