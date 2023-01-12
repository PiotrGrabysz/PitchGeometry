from collections import OrderedDict

import numpy as np
import pandas as pd

from pitch_geo.dataset import tf_dataloaders
from tests.dataset import sample_dataframes


def test_keypoints_to_dict():
    """Tests the function keypoints_to_dict, which turns keypoint annotations of one image into a dictionary."""

    image_path = "images/0001.jpg"
    df = pd.DataFrame(
        data={
            "x": [0, 10, 20],
            "y": [0, 30, 40],
            "vis": [0, 2, 2],
            "kid": [10, 11, 12],
            "image_path": [image_path] * 3,
        }
    )

    expected_output = {
        image_path: np.array([[0, 0, 0], [10, 30, 2], [20, 40, 2]]),
    }

    output_dict = tf_dataloaders.df_to_keypoints_dict(df)

    compare_annotations(annotations_true=expected_output, annotations_pred=output_dict)


def compare_annotations(annotations_true, annotations_pred):
    for (k_true, v_true), (k_pred, v_pred) in zip(
        annotations_true.items(), annotations_pred.items()
    ):
        assert k_pred == k_true
        assert np.array_equal(v_pred, v_true)


def test_df_to_keypoints_dict():
    expected_result = {
        "images/train/45e6306c2c13b547eb5e3f016dc7c5.jpg": np.array(
            [[0, 0, 0], [918, 315, 2], [0, 0, 0], [0, 0, 0]]
        ),
        "images/train/c814ce8e31e077df861c2848fe14fb.jpg": np.array(
            [[532, 158, 2], [1732, 227, 2], [0, 0, 0], [0, 0, 0]]
        ),
        "images/train/f028a66261b63d215b6cbddf7fcc34.jpg": np.array(
            [[745, 276, 2], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
        ),
    }

    output = tf_dataloaders.df_to_keypoints_dict(sample_dataframes.df1)
    for (img_path1, annotations1), (img_path2, annotations2) in zip(
        expected_result.items(), output.items()
    ):
        assert img_path1 == img_path2
        assert np.array_equal(annotations1, annotations2)
