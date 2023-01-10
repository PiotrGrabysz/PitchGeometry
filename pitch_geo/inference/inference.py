import argparse
from pathlib import Path

import typer

from pitch_geo.inference import utils
from pitch_geo.dataset.tf_dataloaders import ImageDatasetBuilder
from pitch_geo.models.models import load_saved_model


def infer(
        model_name: Path = './best_model',
        input_dir: Path = './data/images/test',
        output_csv: Path = './outputs/output_keypoints.csv',
        batch: int = 32
):
    """
    Run inference on images from a given folder.

    Args:

        model_name: Path to a folder containing a saved_model.

        input_dir: Directory containing images for inference.

        output_csv: Path to a csv file the keypoints will be saved in.

        batch: Size a batch which the images are processed in.
    """

    # Read the model

    model = load_saved_model(model_name)
    model_input_shape = model.input_shape[1:3]

    # Get the data loader

    dataset_builder = ImageDatasetBuilder(input_dir, image_size=model_input_shape, batch_size=batch)
    dataset = dataset_builder.build()

    # Inference
    keypoints = model.predict(dataset)

    # Data Frame with keypoints annotations
    df = utils.keypoints_to_df(keypoints, dataset_builder.images_paths, should_add_ghost_keypoints=True)

    df.to_csv(output_csv, index=False)
    print(f'Keypoints saved in {output_csv}')


if __name__ == '__main__':
    typer.run(infer)
