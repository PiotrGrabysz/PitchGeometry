""" Implementation of custom metrics. """

import numpy as np
import tensorflow as tf


def not_visible(x: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    """
    Check which keypoints should be treated as not visible.

    In the training data, keypoints which are not visible have their xy coordinates zeroed out. If the model predicts a
    point to be very close to zero then it is assumed it is not visible.

    Returns:

    """

    x_shape = tf.shape(x)
    n_keypoints = x_shape[-1] // 2
    batch_size = x_shape[0]
    keypoints = tf.reshape(tensor=x, shape=[batch_size, n_keypoints, 2])
    keypoints = tf.cast(keypoints, dtype='float32')

    # Calculate the distance between keypoints and the point (0, 0) - the vector norm

    dist = tf.math.sqrt(tf.math.reduce_sum(tf.pow(keypoints, 2), axis=-1))
    vis = (dist < tf.constant(threshold))

    # Flatten the result
    vis = tf.reshape(vis, [-1])
    return vis


def get_visible_from_array(y, threshold):
    return y[:, :, :, 2::3] > threshold


class VisiblePrecision(tf.keras.metrics.Precision):
    """
    Class implementing a custom metric for measuring precision of detecting visible keypoints versus not visible
    keypoints.
    """
    def __init__(self, name: str = 'visible_precision', threshold: float = 0.5):
        """
        Constructor method for VisiblePrecision class.

        Args:
            name: name of the metric logged during the training loop.
            threshold: visibility values bigger than the threshold are assumed to belong to the class 'visible' and
            'not visible' otherwise.
        """
        super().__init__(name=name)
        self.threshold = tf.constant(threshold)

    def update_state(self, y_true, y_pred, *args, **kwargs):
        """Accumulates true positive and false positive statistics of the prediction that a given keypoint is invisible.

        Args:
            y_true: An array of ground true labels.
            y_pred: An array of models predictions.
        """
        visible_true = get_visible_from_array(y_true, self.threshold)
        visible_pred = get_visible_from_array(y_pred, self.threshold)

        return super().update_state(y_true=visible_true, y_pred=visible_pred, *args, **kwargs)


class VisibleRecall(tf.keras.metrics.Recall):
    """
    Class implementing a custom metric for measuring recall of detecting visible keypoints versus not visible
    keypoints.
    """
    def __init__(self, name: str = 'visible_recall', threshold: float = 0.5):
        """
        Constructor method for VisibleRecall class.

        Args:
            name: name of the metric logged during the training loop.
            threshold: visibility values bigger than the threshold are assumed to belong to the class 'visible' and
            'not visible' otherwise.
        """
        super().__init__(name=name)
        self.threshold = tf.constant(threshold)

    def update_state(self, y_true, y_pred, *args, **kwargs):
        """Accumulates true positive and false positive statistics of the prediction that a given keypoint is invisible.

        Args:
            y_true: An array of ground true labels.
            y_pred: An array of models predictions.
        """
        visible_true = get_visible_from_array(y_true, self.threshold)
        visible_pred = get_visible_from_array(y_pred, self.threshold)

        return super().update_state(y_true=visible_true, y_pred=visible_pred, *args, **kwargs)


class XYMeanSquaredError(tf.keras.losses.MeanSquaredError):
    """ Calculates mean squared error loss only for x, y coordinates of the keypoint. """
    def __init__(self, n_keypoints: int, name: str = 'xy_loss'):
        super().__init__(name=name)
        self.n_keypoints = n_keypoints

    def update_state(self, y_true, y_pred, *args, **kwargs):
        xy_true = tf.reshape(tensor=y_true, shape=[-1, self.n_keypoints, 3])[:, :, :2]
        xy_pred = tf.reshape(tensor=y_pred, shape=[-1, self.n_keypoints, 3])[:, :, :2]

        return super().update_state(xy_true, xy_pred, *args, **kwargs)
