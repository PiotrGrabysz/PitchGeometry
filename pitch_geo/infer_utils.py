import pandas as pd

from pitch_geo.constants import DATA_FOLDER, LABELS, NUM_KEYPOINTS


def keypoints_to_df(keypoints, images_paths, should_add_ghost_keypoints: bool = False) -> pd.DataFrame:
    """
    Convert numpy array with keypoints into pandas data frame with format defined in the task specification.

    Args:
        keypoints: numpy ndarray of shape [batch size, num_keypoints, 3]
        images_paths:
        should_add_ghost_keypoints: I noticed that there are some keypoints in the data which are never visible.
            The model does not predict them, but for the sake of consistency with the original it can add them to the
            final frame, with x, y and vis equal to 0.

    Returns:
        df: a pandas DataFrame with a format defined in the task specification.

    """
    keypoints = keypoints.reshape(len(images_paths), NUM_KEYPOINTS, 3).reshape(-1, 3)

    kid = pd.Series([label for _, label in LABELS.items()])
    kid = pd.concat([kid for _ in range(len(images_paths))], ignore_index=True)

    paths = pd.concat(
        [
            pd.Series([strip_datafolder_name(p) for _ in range(NUM_KEYPOINTS)])
            for p in images_paths
        ],
        ignore_index=True
    )

    df = pd.DataFrame(data={
        'x': keypoints[:, 0],
        'y': keypoints[:, 1],
        'vis': keypoints[:, 2],
        'kid': kid,
        'dataset': 'test',
        'image_path': paths
    })

    df = rescale_xy(df)
    df = rename_visibility(df)

    if should_add_ghost_keypoints:
        df = add_ghost_keypoints(df)

    df = zero_out_invisible_keypoints(df)

    df = df.sort_values(by=['image_path', 'kid'])

    return df


def rescale_xy(df, width: int = 1920, height: int = 1080):
    """" Rescale xy back to original image dimension. """
    df['x'] = round(df['x'] * width).astype('int64')
    df['y'] = round(df['y'] * height).astype('int64')
    return df


def rename_visibility(df, threshold=0.5):
    """ Remap visibility back to original 0 and 2. """
    df['vis'] = df['vis'].apply(lambda x: 2 if x >= threshold else 0)
    return df


def zero_out_invisible_keypoints(df: pd.DataFrame) -> pd.DataFrame:
    """ There is a convention that invisible keypoints (visibility 0) has their xy coordinates zeroed out. """
    df.loc[df['vis'] == 0, 'x'] = 0.0
    df.loc[df['vis'] == 0, 'y'] = 0.0
    return df


def add_ghost_keypoints(df: pd.DataFrame) -> pd.DataFrame:
    """
    I noticed that there are some keypoints in the data which are never visible.
    The model does not predict them, but for the sake of consistency with the original data I can add those 'ghost'
    keypoints to the final data frame frame, with their corresponding x, y and vis equal to 0.
    """
    ghost_keypoints = pd.concat(
        (get_remaining_keypoints(img_path) for img_path in df['image_path'].unique()),
        ignore_index=True
    )
    df = pd.concat([df, ghost_keypoints], ignore_index=True)
    return df


def strip_datafolder_name(path: str):
    if str(DATA_FOLDER) in path:
        return path[len(str(DATA_FOLDER)) + 1:]
    return path


def get_remaining_keypoints(img_path):
    df_tmp = pd.DataFrame({
        'x': 0,
        'y': 0,
        'vis': 0,
        'kid': [24, 30, 36, 37, 38],
        'dataset': 'test',
        'image_path': img_path
    })
    return df_tmp
