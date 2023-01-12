""" Test the functions from metrics module. """

import numpy as np
import pytest

from pitch_geo.models import metrics

x = [
    532,
    158,
    1732,
    227,
    0,
    0,
    0,
    0,
    359,
    203,
    723,
    234,
    564,
    300,
    290,
    410,
    0,
    0,
    0,
    0,
]
y = [
    174,
    250,
    300,
    264,
    0,
    0,
    0,
    0,
    276,
    336,
    1718,
    383,
    1712,
    474,
    1702,
    596,
    561,
    358,
    1332,
    443,
]
vis = [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
keypoints1 = np.expand_dims(np.array([x, y, vis]).T, axis=0)

tp_ground_truth = np.array([[[532, 1732, 1], [158, 227, 1], [0, 0, 0]]])

tp_pred1 = np.array([[[532, 1732, 1], [158, 227, 1], [10, 10, 1]]])
tp_pred2 = np.array([[[532, 0, 1], [158, 0, 1], [0, 0, 0]]])
tp_pred3 = np.array([[[0.05, 0.05, 1], [0, 0, 0], [10, 10, 1]]])
tp_pred4 = np.array([[[0.05, 0.05, 1], [0, 0, 0], [0, 0, 0]]])


@pytest.mark.parametrize(
    "kps_true, kps_pred, precision",
    [
        (keypoints1, keypoints1, 1.0),  # identical keypoint should have precision = 1.0
        (tp_ground_truth, tp_pred1, 0.66666666),  # two TP, one FP
        (tp_ground_truth, tp_pred2, 1.0),  # two TP, zero FP
        (tp_ground_truth, tp_pred3, 0.5),  # one TP one FP
        (tp_ground_truth, tp_pred4, 1.0),  # one TP, zero FP
    ],
)
def test_precision_metric(kps_true, kps_pred, precision):
    m = metrics.VisiblePrecision(name="test_not_visible_precision_metric")
    m.update_state(kps_true, kps_pred)
    assert np.isclose(m.result().numpy(), precision)


@pytest.mark.parametrize(
    "kps_true, kps_pred, precision",
    [
        (tp_ground_truth, tp_pred1, 0.66666666),  # two TP, one FP
        (tp_ground_truth, tp_pred2, 1.0),  # two TP, zero FP
        (tp_ground_truth, tp_pred3, 0.5),  # one TP one FP
        (tp_ground_truth, tp_pred4, 1.0),  # one TP, zero FP
    ],
)
def test_precision_metric2(kps_true, kps_pred, precision):
    m = metrics.VisiblePrecision()
    m.update_state(np.expand_dims(kps_true[:, 0, :], axis=0), np.expand_dims(kps_pred[:, 0, :], axis=0))
    m.update_state(np.expand_dims(kps_true[:, 1, :], axis=0), np.expand_dims(kps_pred[:, 1, :], axis=0))
    m.update_state(np.expand_dims(kps_true[:, 2, :], axis=0), np.expand_dims(kps_pred[:, 2, :], axis=0))
    assert np.isclose(m.result().numpy(), precision)


@pytest.mark.parametrize(
    "kps_true, kps_pred, precision",
    [
        (keypoints1, keypoints1, 1.0),  # identical keypoint should have recall = 1.0
        (tp_ground_truth, tp_pred1, 1.0),
        (tp_ground_truth, tp_pred2, 1.0),
        (tp_ground_truth, tp_pred3, 0.5),
        (tp_ground_truth, tp_pred4, 0.5),
    ],
)
def test_recall_metric(kps_true, kps_pred, precision):
    m = metrics.VisibleRecall()
    m.update_state(kps_true, kps_pred)
    assert np.isclose(m.result().numpy(), precision)
