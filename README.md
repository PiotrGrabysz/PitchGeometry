# ReSpo.Vision Recruitment Task - Pitch Geometry

Currently working at a teacher forcing strategy to the loss function.

## Installation

My solution should be run in a docker container. To build the Docker image, run in the terminal:
```shell
$ docker/build.sh
```

It pulls the image with `tensorflow 2.9.1` and installs additional dependencies.

## Running the solution

The run the container, run:
```shell
$ docker/run.sh
```

The training pipeline is described in `Training.ipynb` and can be run in the notebook. To start a jupyter lab session, 
run
```shell
$ jupyter-lab
```

The inference pipeline can be run through a command
```shell
python3 inference.py --model MODEL --input_dir INPUT_DIR [--batch BATCH --output_csv OUTPUT_CSV

Arguments:
--model MODEL, -m MODEL
                        Path to the saved model.
--input_dir INPUT_DIR, -i INPUT_DIR
                        Directory containing images for inference. This directory must be inside ./data/ folder.
--batch BATCH, -b BATCH
                        Batch size.
--output_csv OUTPUT_CSV, -o OUTPUT_CSV                        
                        Filename of a csv file with keypoint annotations.
```

## Project description

I provide two notebook: `DataExploration.ipynb` with some initial thoughts on the data and `Training.ipynb` describing 
the training procedure.

The final dataframe with predictions is the file `./df_keypoints.csv`. A model which I used for this prediction is 
`./checkpoints/20221124_094146/saved_model`.
