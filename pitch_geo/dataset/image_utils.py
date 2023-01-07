from typing import Tuple

import tensorflow as tf


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