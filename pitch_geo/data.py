""" Functionalities to read and process the dataset. """


from functools import partial
from pathlib import Path
from typing import Dict, Sequence, Tuple, Union

from matplotlib import image
import numpy as np
import pandas as pd
import tensorflow as tf

from pitch_geo.constants import DATA_FOLDER


def image_loader(img_paths: Sequence, image_size: Union[int, Tuple[int, int]], autotune):

    dataset_img_paths = tf.data.Dataset.from_tensor_slices(img_paths)

    # Prepare image loader

    if isinstance(image_size, int):
        image_size = (image_size, image_size)

    images = dataset_img_paths.map(
        partial(
            load_image,
            image_size=image_size,
            num_channels=3,
            interpolation='bilinear'
        ),
        num_parallel_calls=autotune
    )
    return images


def get_data_loader(
        paths: Sequence,
        df: pd.DataFrame,
        image_size: Union[int, Tuple[int, int]],
        autotune=None,
):

    # paths are expected to be relative to the DATA_FOLDER. Get the absolute paths
    absolute_paths = [str(DATA_FOLDER / p) for p in paths]
    dataset_paths = tf.data.Dataset.from_tensor_slices(paths)

    images = image_loader(absolute_paths, image_size, autotune)

    # Prepare keypoints loader

    annotations_dict = df_to_keypoints_dict(df)

    def _map_func(image_path):
        # image_path is supposed to be bytes object
        return annotations_dict[image_path.numpy().decode('utf-8')]

    keypoints = (
        dataset_paths
        .map(
            lambda file_name: tf.py_function(_map_func, [file_name], tf.float16),
            num_parallel_calls=autotune
        )
    )

    dataset = tf.data.Dataset.zip((images, keypoints))
    return dataset


def df_to_keypoints_dict(df: pd.DataFrame, dtype=np.float32) -> Dict[str, np.ndarray]:
    """
    Turn the keypoints data from the data frame into a dictionary `{image_path: annotations}`. Annotations are a
    numpy.ndarray with three columns standing for x, y, and vis.
    """
    def _get_kps_func(image_annotations):
        """ A helper function getting keypoint annotations from a part of the data frame describing one image. """
        return image_annotations[['x', 'y', 'vis']].values.copy().astype(dtype)

    return df.groupby('image_path').apply(_get_kps_func).to_dict()


def read_image_and_keypoints(img_path: str, df: pd.DataFrame, root_data_path: Path) -> (np.ndarray, np.ndarray):
    img = image.imread(root_data_path / img_path)
    keypoints = df[df['image_path'] == img_path]

    # Ensure that the keypoints as always in the exact same order
    keypoints.sort_values(by='kid', ascending=True)
    keypoints = keypoints[['x', 'y', 'vis']].values
    return img, keypoints


