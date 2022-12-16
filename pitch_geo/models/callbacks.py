import io
import itertools

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf

from pitch_geo.models.metrics import get_visible_from_array
from pitch_geo.vis_utils import plot_image_with_annotations_on_ax


class LogConfusionMatrixCallback:
    def __init__(self, model, logdir, dataset, threshold=0.5):
        self.model = model
        self.logdir = logdir
        self.file_writer_cm = tf.summary.create_file_writer(str(logdir / 'cm'))
        self.dataset = dataset
        self.class_names = ['not_visible', 'visible']
        self.threshold = threshold
        self.visibility = np.concatenate([
            tf.reshape(tensor=get_visible_from_array(kps, self.threshold), shape=[-1])
            for _, kps in iter(self.dataset)
        ])

    def __call__(self, epoch, logs):
        """ Log the confusion matrix. """

        cm_image = self.calculate_cm()

        # Log the confusion matrix as an image summary.
        with self.file_writer_cm.as_default():
            tf.summary.image("Confusion Matrix", cm_image, step=epoch)

    def calculate_cm(self):
        # Use the model to predict the values from the test_images.
        test_pred_raw = self.model.predict(self.dataset)
        test_pred = get_visible_from_array(test_pred_raw, threshold=self.threshold).reshape(-1)

        # Calculate the confusion matrix using sklearn.metrics
        cm = confusion_matrix(self.visibility, test_pred)

        figure = self.plot_confusion_matrix(cm, class_names=self.class_names)
        cm_image = plot_to_image(figure)
        plt.close(figure)
        return cm_image

    @staticmethod
    def plot_confusion_matrix(cm, class_names):
        """
        Returns a matplotlib figure containing the plotted confusion matrix.

        Args:
           cm (array, shape = [n, n]): a confusion matrix of integer classes
           class_names (array, shape = [n]): String names of the integer classes
        """

        # plt.ioff()
        figure = plt.figure(figsize=(8, 8))
        disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
        _ = disp.plot()
        plt.tight_layout()
        return figure


class LogPredictedImages:
    def __init__(self, model, logdir, dataset, labels, threshold=0.5, grid_size=4):
        self.model = model
        self.logdir = logdir
        self.file_writer = tf.summary.create_file_writer(str(logdir / 'predictions'))
        self.dataset = dataset
        self.threshold = threshold
        self.grid_size = grid_size
        self.n_images = self.grid_size ** 2
        self.labels = labels

        # Prepare the data
        # Take only the first batch of data. We don't want to plot all the images
        self.sample_images, _ = next(iter(self.dataset))
        self.batch_size = self.sample_images.shape[0]

        # Take the first 25 items from the batch
        if self.batch_size >= self.n_images:
            self.sample_images = self.sample_images[:self.n_images, :, :, :]

    def __call__(self, epoch, logs):
        keypoints = self.predict()
        fig = self.image_grid(self.sample_images, keypoints)
        fig_to_save = plot_to_image(fig)
        with self.file_writer.as_default():
            tf.summary.image("Sample images", fig_to_save, step=epoch)
        plt.close(fig)

    def image_grid(self, images, keypoints):
        """Return a grid of images as a matplotlib figure."""
        # Create a figure to contain the plot.
        figure, axes = plt.subplots(self.grid_size, self.grid_size, figsize=(10, 10))

        for i, (ax, img, kps) in enumerate(zip(axes.flatten(), images, keypoints)):
            plot_image_with_annotations_on_ax(ax, img/255., kps, labels=self.labels, dot_radius=3, vis=True)
            ax.grid(False)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        plt.tight_layout()
        # plt.plot()
        return figure

    def predict(self):
        keypoints_hat = self.model.predict(self.sample_images)
        if self.batch_size <= self.n_images:
            keypoints_hat = keypoints_hat[:self.n_images, :, :, :]
        return keypoints_hat


def plot_to_image(figure):
    """
    Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call.
    """

    buf = io.BytesIO()

    # Use plt.savefig to save the plot to a PNG in memory.
    plt.savefig(buf, format='png')

    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)

    # Use tf.image.decode_png to convert the PNG buffer
    # to a TF image. Make sure you use 4 channels.
    image = tf.image.decode_png(buf.getvalue(), channels=4)

    # Use tf.expand_dims to add the batch dimension
    image = tf.expand_dims(image, 0)

    return image
