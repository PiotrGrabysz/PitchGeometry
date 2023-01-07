import pandas as pd


def correct_mislabeled_examples(df: pd.DataFrame) -> pd.DataFrame:
    """
    There is one frame where keypoint '38' is probably a mistake, because it is put in the place where '39' is in
    every other frame (see the notebook with dataset exploration).
    Find that one sample where keypoint '38' has non-zero visibility and set it to '39'.

    Returns:
        df: A processed dataset frame.
    """

    df = df.copy()

    idx_38 = df.query('kid == 38 and vis != 0').index
    image_path_38, x_38, y_38, vis_38 = df.loc[idx_38, ['image_path', 'x', 'y', 'vis']].values.flatten()
    idx_39 = df.query(f'kid == 39 and image_path == "{image_path_38}"').index

    df.loc[idx_39, 'x'] = x_38
    df.loc[idx_39, 'y'] = y_38
    df.loc[idx_39, 'vis'] = vis_38

    df.loc[idx_38, 'x'] = 0
    df.loc[idx_38, 'y'] = 0
    df.loc[idx_38, 'vis'] = 0

    return df


def drop_empty_images(df: pd.DataFrame) -> pd.DataFrame:
    """
    Some images in the data frame has no annotated keypoints: every keypoint is marked as not visible.
    Drop image_path corresponding to such images.
    """

    df = df.copy()

    images_with_no_keypoints = df.groupby('image_path')['vis'].sum() == 0
    images_with_no_keypoints = [item[0] for item in images_with_no_keypoints.items() if item[1]]
    df = df[~df['image_path'].isin(images_with_no_keypoints)]
    return df


def drop_ghost_keypoints(df: pd.DataFrame) -> pd.DataFrame:
    """
    Some keypoints in the data frame never has a non-zero value. I call them 'ghost' keypoints and remove from the data.
    """
    df = df.copy()

    never_visible = df['kid'].unique()[df.groupby('kid')['vis'].sum() == 0]
    df = df[~df['kid'].isin(never_visible)]
    return df


def normalize_coordinates(df: pd.DataFrame, img_width: int = 1920, img_height: int = 1080) -> pd.DataFrame:
    """
    Coordinates in the data frame are describing a 1920 by 1080 images. Normalize them to lie inside [0, 1] interval.
    """

    df = df.copy()
    df['x'] = df['x'] / img_width
    df['y'] = df['y'] / img_height

    assert 0.0 <= df[['x', 'y']].min().min() and df[['x', 'y']].max().max() < 1.0, \
        'Normalized x, y coordinates should lie between 0 and 1.'
    return df
