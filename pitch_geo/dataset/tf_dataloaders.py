from functools import partial
from typing import Union, Tuple, Sequence, Dict

import numpy as np
import pandas as pd
import tensorflow as tf

from pitch_geo.constants import DATA_FOLDER
from pitch_geo.dataset.image_utils import load_image
from pitch_geo.models.augmentation import Augmentation

AUTOTUNE = tf.data.AUTOTUNE


class DatasetBuilder:
    """Class for building a Tensorflow dataset with images and keypoints. """

    shuffle_buffer_size = 100

    def __init__(
            self,
            data_frame: pd.DataFrame,
            image_size: int,
            batch_size: int,
            shuffle: bool = False,
            augmentation: Augmentation = None
    ):
        self.data_frame = data_frame
        self.image_size = image_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augmentation = augmentation

    def build(self):
        dataset = get_data_loader(df=self.data_frame, image_size=self.image_size)
        if self.shuffle:
            dataset = dataset.shuffle(self.shuffle_buffer_size)

        if self.augmentation:
            dataset = dataset.map(
                lambda image, keypoints: tf.numpy_function(
                    func=self.augmentation.aug_fn, inp=[image, keypoints], Tout=[tf.float32, tf.float16]
                ),
                num_parallel_calls=AUTOTUNE
            )

        dataset = (
            dataset
            .map(
                partial(self._set_img_shape, img_shape=(self.image_size, self.image_size, 3)),
                num_parallel_calls=AUTOTUNE
            )
            .batch(self.batch_size)
            .prefetch(buffer_size=AUTOTUNE)
        )
        return dataset

    @staticmethod
    def _set_img_shape(img, keypoints, img_shape):
        img.set_shape(img_shape)
        return img, keypoints


def get_data_loader(
        df: pd.DataFrame,
        image_size: Union[int, Tuple[int, int]],
):

    image_paths = df['image_path'].unique().tolist()
    # image_paths are expected to be relative to the DATA_FOLDER. Get the absolute paths
    absolute_paths = [str(DATA_FOLDER / p) for p in image_paths]
    dataset_paths = tf.data.Dataset.from_tensor_slices(image_paths)

    images_dataset = get_image_loader(absolute_paths, image_size)

    # Prepare keypoints loader

    annotations_dict = df_to_keypoints_dict(df)

    def _map_img_path_to_annotations(image_path):
        # image_path is supposed to be bytes object
        return annotations_dict[image_path.numpy().decode('utf-8')]

    dataset_paths = tf.data.Dataset.from_tensor_slices(image_paths)
    keypoints = (
        dataset_paths
        .map(
            lambda file_name: tf.py_function(_map_img_path_to_annotations, [file_name], tf.float16),
            num_parallel_calls=AUTOTUNE
        )
    )

    dataset = tf.data.Dataset.zip((images_dataset, keypoints))
    return dataset


def get_image_loader(img_paths: Sequence, image_size: Union[int, Tuple[int, int]]):

    img_paths_dataset = tf.data.Dataset.from_tensor_slices(img_paths)

    # Prepare image loader

    if isinstance(image_size, int):
        image_size = (image_size, image_size)

    images_dataset = img_paths_dataset.map(
        partial(
            load_image,
            image_size=image_size,
            num_channels=3,
            interpolation='bilinear'
        ),
        num_parallel_calls=AUTOTUNE
    )
    return images_dataset


def df_to_keypoints_dict(df: pd.DataFrame, dtype=np.float32) -> Dict[str, np.ndarray]:
    """
    Turn the keypoints dataset from the data frame into a dictionary `{image_path: annotations}`. Annotations are a
    numpy.ndarray with three columns standing for x, y, and vis.
    """
    def _get_kps_func(image_annotations):
        """ A helper function getting keypoint annotations from a part of the dataset frame describing one image. """
        return image_annotations[['x', 'y', 'vis']].values.copy().astype(dtype)

    return df.groupby('image_path').apply(_get_kps_func).to_dict()
