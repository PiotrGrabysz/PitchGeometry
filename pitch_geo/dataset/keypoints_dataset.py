from typing import Dict, Literal, Sequence, Tuple, Union
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from pitch_geo import constants
from pitch_geo.dataset import _keypoints_preprocessing_utils


class KeypointDataset:
    """
    Class representing a data frame with keypoints.

    It reads the data frame and preprocess it.
    Preprocessing consists of:
     * normalizing the keypoint coordinates,
     * dropping frames with no visible keypoint annotated,
     * dropping keypoints which are never visible,
     * switching visibility 2 to 1 (I found it easier to work with {0, 1} visibility),
     * correcting a probably mislabelled example of '38' keypoint (See the notebook with Data Exploration),
     * sorting the dataset frame by image_path and kid.

    Preprocessing is applied only if `dataset == 'train'`.
    """

    def __init__(
        self,
        dataset: Literal["train", "test"],
        keypoints_data_path: Path,
        images_base_dir: Path,
    ):
        self.dataset = dataset
        self._keypoints_data_path = keypoints_data_path
        self._images_base_dir = images_base_dir

        raw_df = pd.read_csv(self._keypoints_data_path)
        subset_df = raw_df.query(f'dataset == "{dataset}"')

        self._df = self._preprocess_data(subset_df.copy())

    @property
    def df(self):
        return self._df.copy()

    @property
    def num_keypoints(self):
        return self._df["kid"].nunique()

    @property
    def label_map(self):
        keypoint_ids = sorted(self._df["kid"].unique())
        return {
            key: value for key, value in zip(range(len(keypoint_ids)), keypoint_ids)
        }

    def split(self, test_size: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the keypoints data frame into train and validation sets.

        Args:
            test_size: A fraction of all images that goes into the validation set.

        """
        unique_images_paths = self._df["image_path"].unique()
        train_paths, val_paths = train_test_split(
            unique_images_paths, test_size=test_size, random_state=42, shuffle=True
        )
        train_df = self._df[self._df["image_path"].isin(train_paths)]
        val_df = self._df[self._df["image_path"].isin(val_paths)]
        return train_df, val_df

    def __repr__(self):
        return (
            f"KeypointDataset(dataset={self.dataset}, keypoints_data_path={self._keypoints_data_path}, "
            f"images_base_dir={self._images_base_dir})"
        )

    def _preprocess_data(self, df: pd.DataFrame):
        if self.dataset == "test":
            return df

        # Change visibility 2 to visibility 1
        df["vis"].replace(to_replace=2, value=1, inplace=True)

        df = _keypoints_preprocessing_utils.correct_mislabeled_examples(df)
        df = _keypoints_preprocessing_utils.drop_empty_images(df)
        df = _keypoints_preprocessing_utils.drop_ghost_keypoints(df)
        df = _keypoints_preprocessing_utils.normalize_coordinates(df)

        df.sort_values(by=["image_path", "kid"], inplace=True)

        return df


def get_data(dataset: Literal["train", "test"]) -> KeypointDataset:
    return KeypointDataset(
        dataset=dataset,
        keypoints_data_path=constants.KEYPOINTS_PATH,
        images_base_dir=constants.DATA_FOLDER,
    )
