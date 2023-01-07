from pathlib import Path

import pandas as pd

from pitch_geo.dataset import _keypoints_preprocessing_utils
from tests.dataset import sample_data_frames


def test_correct_mislabeled_examples():
    init_df = pd.read_csv(Path('tests/dataset/sample_keypoints.csv'))
    processed_df = _keypoints_preprocessing_utils.correct_mislabeled_examples(init_df)
    expected_data_frame = pd.read_csv(Path('tests/dataset/sample_keypoints_38_corrected.csv'))
    pd.testing.assert_frame_equal(processed_df, expected_data_frame)


def test_drop_empty_images():
    init_df = pd.DataFrame.from_dict(sample_data_frames.drop_empty_images_init)
    processed_df = _keypoints_preprocessing_utils.drop_empty_images(init_df)
    expected_data_frame = pd.DataFrame.from_dict(sample_data_frames.drop_empty_images_after)
    pd.testing.assert_frame_equal(processed_df, expected_data_frame)

