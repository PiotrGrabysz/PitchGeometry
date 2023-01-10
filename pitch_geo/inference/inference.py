import argparse
from pathlib import Path

from pitch_geo.inference import utils
from pitch_geo.dataset.tf_dataloaders import ImageDatasetBuilder
from pitch_geo.models.models import load_saved_model


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

    model = load_saved_model(args.model)
    model_input_shape = model.input_shape[1:3]

    # Get the data loader

    dataset_builder = ImageDatasetBuilder(args.input_dir, image_size=model_input_shape, batch_size=args.batch)
    dataset = dataset_builder.build()

    # Inference
    keypoints = model.predict(dataset)

    # Data Frame with keypoints annotations
    df = utils.keypoints_to_df(keypoints, dataset_builder.images_paths, should_add_ghost_keypoints=True)

    df.to_csv(args.output_csv, index=False)
    print(f'Keypoints saved in {args.output_csv}')


if __name__ == '__main__':
    main()

