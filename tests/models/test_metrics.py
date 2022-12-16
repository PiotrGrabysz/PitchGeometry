""" Test the functions from metrics module. """

import numpy as np
import pytest

from pitch_geo.models import metrics

keypoints1 = np.array(
    [[[[
        532, 158, 1732, 227, 0, 0, 0, 0, 359, 203,
        723, 234, 564, 300, 290, 410, 0, 0, 0, 0,
        174, 250, 300, 264, 0, 0, 0, 0, 276, 336,
        1718, 383, 1712, 474, 1702, 596, 561, 358, 1332, 443,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0
    ]]]]
)
visibility1 = np.array(
    [
        False, False, True, True, False, False, False, False, True,
        True, False, False, True, True, False, False, False, False,
        False, False, True, True, True, True, True, True, True,
        True, True, True, True, True, True, True, True, True,
        True, True
    ],
    dtype=bool
)


@pytest.mark.parametrize('x, vis', [
    (keypoints1, visibility1)
])
def test_not_visible(x, vis):
    assert np.array_equal(metrics.not_visible(keypoints1), vis)


tp_ground_truth = np.array(
    [[[[
        532, 158, 1732, 227, 0, 0, 0, 0
    ]]]]
)
tp_pred1 = np.array(
    [[[[
        532, 158, 1732, 227, 10, 10, 10, 10
    ]]]]
)
tp_pred2 = np.array(
    [[[[
        532, 158, 0, 0, 0, 0, 0, 0
    ]]]]
)
tp_pred3 = np.array(
    [[[[
        0.05, 0.05, 0, 0, 0, 0, 10, 10
    ]]]]
)
tp_pred4 = np.array(
    [[[[
        0.05, 0.05, 0, 0, 10, 10, 10, 10
    ]]]]
)


@pytest.mark.parametrize('kps_true, kps_pred, precision', [
    (keypoints1, keypoints1, 1.0),  # identical keypoint should have precision = 1.0
    (tp_ground_truth, tp_pred1, 0.0),   # no TP
    (tp_ground_truth, tp_pred2, 0.66666666),  # two TP and one FP
    (tp_ground_truth, tp_pred3, 0.33333333),  # one TP and two FP
    (tp_ground_truth, tp_pred4, 0)  # zero TP
])
def test_precision_metric(kps_true, kps_pred, precision):
    m = metrics.NotVisiblePrecision(name='test_not_visible_precision_metric')
    m.update_state(kps_true, kps_pred)
    assert np.isclose(m.result().numpy(), precision)


@pytest.mark.parametrize('kps_true, kps_pred, precision', [
    (tp_ground_truth, tp_pred1, 0.0),   # no TP
    (tp_ground_truth, tp_pred2, 0.66666666),  # two TP and one FP
    (tp_ground_truth, tp_pred3, 0.33333333),  # one TP and two FP
    (tp_ground_truth, tp_pred4, 0)  # zero TP
])
def test_precision_metric2(kps_true, kps_pred, precision):
    m = metrics.NotVisiblePrecision()
    m.update_state(kps_true[:, :, :, :2], kps_pred[:, :, :, :2])
    m.update_state(kps_true[:, :, :, 2:4], kps_pred[:, :, :, 2:4])
    m.update_state(kps_true[:, :, :, 4:6], kps_pred[:, :, :, 4:6])
    m.update_state(kps_true[:, :, :, 6:], kps_pred[:, :, :, 6:])
    assert np.isclose(m.result().numpy(), precision)


@pytest.mark.parametrize('kps_true, kps_pred, precision', [
    (keypoints1, keypoints1, 1.0),  # identical keypoint should have recall = 1.0
    (tp_ground_truth, tp_pred1, 0.0),
    (tp_ground_truth, tp_pred2, 1.0),  # two TP
    (tp_ground_truth, tp_pred3, 0.5),   # one TP and one FN
    (tp_ground_truth, tp_pred4, 0.0)
])
def test_recall_metric(kps_true, kps_pred, precision):
    m = metrics.NotVisibleRecall()
    m.update_state(kps_true, kps_pred)
    assert np.isclose(m.result().numpy(), precision)
