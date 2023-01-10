from pathlib import Path
from typing import Optional, Union, Tuple

import tensorflow as tf
from tensorflow.keras import layers


def get_model(img_size: Union[int, Tuple], num_keypoints: int, dropout: Optional[float] = None):
    """
    Construct the model with keras functional API. The model backbone is EfficientNetB1.

    Args:
        img_size: image input size. Can be a tuple of (height, width) or an integer.
        num_keypoints: number of keypoints. The model's output is going to be (batch_size, 1, 1, 3 * num_keypoints).
        dropout: Dropout rate to use before the final layer. If None, then no dropout is used.

    Returns:
        A tensorflow model.
    """
    backbone = tf.keras.applications.EfficientNetB1(
        weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3)
    )
    backbone.trainable = False

    inputs = layers.Input((img_size, img_size, 3))

    x = tf.keras.applications.efficientnet.preprocess_input(inputs)
    x = backbone(x)
    x = layers.Conv2D(512, 3, padding='same', activation='relu')(x)

    x = layers.Conv2D(256, 3, 2, padding='same', activation='relu')(x)
    x = layers.Conv2D(256, 2, 2, activation='relu')(x)
    if dropout is not None:
        x = layers.Dropout(dropout)(x)
    outputs = layers.Conv2D(3 * num_keypoints, 2, 2, activation='sigmoid')(x)
    outputs = layers.Reshape((num_keypoints, 3))(outputs)

    return tf.keras.Model(inputs, outputs, name="keypoint_detector")


def load_saved_model(model_path: Union[str, Path]):
    model = tf.keras.models.load_model(model_path)
    print(f'Loaded the model {model_path}.')
    return model
