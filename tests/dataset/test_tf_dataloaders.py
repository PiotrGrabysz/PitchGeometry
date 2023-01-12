import numpy as np

from pitch_geo.dataset import tf_dataloaders
from tests.dataset import sample_dataframes


def test_df_to_keypoints_dict():
    """
    Tests the function df_to_keypoints_dict, which turns data frame with keypoint annotations into a dictionary:
    {image_path: np.array([x, y, vis])}.
    """

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
