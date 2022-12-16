from collections import OrderedDict

import numpy as np
import pandas as pd
import pytest

from pitch_geo import data
from tests import sample_dataframes


def test_keypoints_to_dict():
    """ Tests the function keypoints_to_dict, which turns keypoint annotations of one image into a dictionary. """

    df = pd.DataFrame(data={
        'x': [0, 1, 2],
        'y': [0, 3, 4],
        'vis': [0, 2, 2],
        'kid': [10, 11, 12]
    })

    expected_output = {
        'keypoints': np.array([0, 0, 1, 3, 2, 4]),
        'vis': np.array([0, 2, 2]),
        'kid': np.array([10, 11, 12])
    }

    output_dict = data.keypoints_to_dict(df)

    compare_annotations(annotations_true=expected_output, annotations_pred=output_dict)


def compare_annotations(annotations_true, annotations_pred):
    for (k_true, v_true), (k_pred, v_pred) in zip(annotations_true.items(), annotations_pred.items()):
        assert k_pred == k_true
        assert np.array_equal(v_pred, v_true)


def test_df_to_dict():
    expected_result = {
        'images/train/c814ce8e31e077df861c2848fe14fb.jpg': {
            'keypoints': np.array([532, 158, 1732, 227, 0, 0, 0, 0]),
            'vis': np.array([2, 2, 0, 0]),
            'kid': np.array([1, 2, 3, 4])
        },
        'images/train/f028a66261b63d215b6cbddf7fcc34.jpg': {
            'keypoints': np.array([745, 276, 0, 0, 0, 0, 0, 0]),
            'vis': np.array([2, 0, 0, 0]),
            'kid': np.array([1, 2, 3, 4])
        },
        'images/train/45e6306c2c13b547eb5e3f016dc7c5.jpg': {
            'keypoints': np.array([0, 0, 918, 315, 0, 0, 0, 0]),
            'vis': np.array([0, 2, 0, 0]),
            'kid': np.array([1, 2, 3, 4])
        }
    }
    expected_result = OrderedDict(sorted(expected_result.items()))

    output = data.df_to_dict(sample_dataframes.df1)
    output = OrderedDict(sorted(output.items()))

    for (img_path1, annotations1), (img_path2, annotations2) in zip(expected_result.items(), output.items()):
        assert img_path1 == img_path2
        compare_annotations(annotations_true=annotations1, annotations_pred=annotations2)
