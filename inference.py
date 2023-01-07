import argparse
from pathlib import Path

import tensorflow as tf

from pitch_geo.constants import DATA_FOLDER
from pitch_geo.data import image_loader
from pitch_geo import infer_utils

AUTOTUNE = tf.data.AUTOTUNE


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        '-m',
        default='best_model',
        help='Path to the saved model. Default to ./best_model/.'
    )
    parser.add_argument(
        '--input_dir', '-i',
        default='images/test',
        help='Directory containing images for inference. This directory must be inside ./data/ folder. '
             'The default value is images/test.'
    )
    parser.add_argument('--batch', '-b', type=int, default=32, help='Batch size. Default to 32.')
    parser.add_argument('--output_csv', '-o', type=Path, help='Filename of a csv file with keypoint annotations.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Read the model

    model = tf.keras.models.load_model(args.model)
    model_input_shape = model.input_shape[1:3]
    print(f'Loaded the model {args.model}.')

    # Get the data loader

    data_path = DATA_FOLDER / args.input_dir
    images_paths = sorted(str(x) for x in data_path.glob('*.jpg'))
    dataset = image_loader(images_paths, model_input_shape, AUTOTUNE)
    dataset = dataset.batch(args.batch).prefetch(buffer_size=AUTOTUNE)

    # Inference
    keypoints = model.predict(dataset)

    # Data Frame with keypoints annotations
    df = infer_utils.keypoints_to_df(keypoints, images_paths, add_empty_keypoints=True)

    df.to_csv(args.output_csv, index=False)
    print(f'Keypoints saved in {args.output_csv}')


if __name__ == '__main__':
    main()

