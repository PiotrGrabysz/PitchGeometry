import pandas as pd

from pitch_geo.constants import DATA_FOLDER, LABELS, NUM_KEYPOINTS


def scale_xy(df):
    df['x'] = round(df['x'] * 1920).astype('int64')
    df['y'] = round(df['y'] * 1080).astype('int64')
    return df


def visibility(df, threshold=0.5):
    df['vis'] = df['vis'].apply(lambda x: 2 if x >= threshold else 0)
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


def keypoints_to_df(keypoints, images_paths):
    keypoints = keypoints.reshape(290, NUM_KEYPOINTS, 3).reshape(-1, 3)

    kid = pd.Series([label for _, label in LABELS.items()])
    kid = pd.concat([kid for _ in range(290)], ignore_index=True)

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
        'dataset': 'test',
        'kid': kid,
        'image_path': paths
    })

    df = scale_xy(df)
    df = visibility(df)

    remaining_keypoints = pd.concat(
        (get_remaining_keypoints(img_path) for img_path in df['image_path'].unique()),
        ignore_index=True
    )
    df = pd.concat([df, remaining_keypoints], ignore_index=True)
    df = df.sort_values(by=['image_path', 'kid'])

    df[df['vis'] == 0]['x'] = 0.0
    df[df['vis'] == 0]['y'] = 0.0

    return df
