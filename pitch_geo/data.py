""" Functionalities to read and process the data. """


from functools import partial
import json
import os
from pathlib import Path
from PIL import Image
import traceback
from typing import Dict, Literal, Sequence, Tuple, Union

from matplotlib import image
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf

from pitch_geo.constants import DATA_FOLDER, KEYPOINTS_PATH

# IMG_SIZE = 224
# BATCH_SIZE = 64
# EPOCHS = 5
# NUM_KEYPOINTS = 38 * 2  # 38 pairs each having x and y coordinates


def get_keypoints(dataset: Literal['train', 'test']) -> pd.DataFrame:
    """
    Read the data frame with annotated keypoints and preprocess it.

    Preprocessing consists of:
     * normalizing the keypoint coordinates,
     * dropping frames with no visible keypoint annotated,
     * dropping keypoints which are never visible,
     * switching visibility 2 to 1 (I found it easier to work with {0, 1} visibility),
     * correcting a probably mislabelled example of '38' keypoint (See the notebook with Data Exploration),
     * sorting the data frame by image_path and kid.

    Preprocessing is applied only if `dataset == 'train'`.

    Args:
        dataset: Which part of the dataset to read, either train or test. If train, then the data is preprocessed.

    Returns:
        A pandas Data Frame.
    """

    df = pd.read_csv(KEYPOINTS_PATH)
    if dataset == 'train':
        df = df[df['dataset'] == 'train'].copy()

        # Change visibility 2 to visibility 1
        df['vis'].replace(to_replace=2, value=1, inplace=True)

        # There is one frame where keypoint '38' was probably mislabelled with (see the notebook with data exploration)

        key_38_mislabelled = df.query('kid == 38 and vis != 0')['image_path'].item()
        idx_39 = df.query(f'kid == 39 and image_path == "{key_38_mislabelled}"').index
        x, y = df.query('kid == 38 and vis != 0')[['x', 'y']].values.flatten()
        df.loc[idx_39, 'x'] = x
        df.loc[idx_39, 'y'] = y
        df.loc[idx_39, 'vis'] = 1

        # Now there are no visible keypoints '38'.
        # Drop all the rows with 'kid' == 38 as they don't contain any information.
        df = df[df['kid'] != 38].copy()

        # Normalize the keypoints coordinates to be between 0 and 1 (knowing that every picture is 1920 by 1080)
        df['x'] = df['x'] / 1920
        df['y'] = df['y'] / 1080

        assert 0.0 <= df[['x', 'y']].min().min() and df[['x', 'y']].max().max() < 1.0, \
            'Normalized x, y coordinates should be between 0 and 1.'

        # Drop the rows with no keypoints annotated

        imgs_with_no_keypoints = df.groupby('image_path')['vis'].sum() == 0
        imgs_with_no_keypoints = list(filter(lambda item: item[1], imgs_with_no_keypoints.items()))
        df = df[~df['image_path'].isin(imgs_with_no_keypoints)]

        # Drop the rows with keypoints which are never visible
        never_visible = df['kid'].unique()[df.groupby('kid')['vis'].sum() == 0]
        df = df[~df['kid'].isin(never_visible)]

        df.sort_values(by=['image_path', 'kid'], inplace=True)

        return df

    elif dataset == 'test':
        df = df[df['dataset'] == 'test'].copy()
        return df


def load_image(
    path: str,
    image_size: Tuple[int, int] = None,
    num_channels: int = 3,
    interpolation: tf.image.ResizeMethod = tf.image.ResizeMethod.BILINEAR,
):
    """
    Load an image from a path and resize it. The code is taken from
    https://github.com/keras-team/keras/blob/v2.10.0/keras/utils/image_dataset.py#L30.

    Args:
        path: A path to the image.
        image_size: Resize the image to a new shape: (new_height, new_width).
        num_channels: Number of color channels for the decoded image.
        interpolation: Interpolation method used for resizing the image.

    Returns:
        A 3D tensor of shape [height, width, channels].
    """

    img = tf.io.read_file(path)
    img = tf.image.decode_image(
        img, channels=num_channels, expand_animations=False
    )
    if image_size is not None:
        img = tf.image.resize(img, image_size, method=interpolation)
    img.set_shape((image_size[0], image_size[1], num_channels))
    return img


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


