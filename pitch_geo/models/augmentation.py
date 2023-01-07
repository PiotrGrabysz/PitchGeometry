from abc import ABC, abstractmethod
from typing import Sequence

import cv2
import numpy as np


class Augmentation(ABC):
    @abstractmethod
    def aug_fn(self, image, keypoints):
        pass


class RandomTranslation(Augmentation):
    def __init__(self, limit=0.5, p=0.5):
        self.limit = limit
        self.p = p

    def __repr__(self):
        return f'RandomTranslation(limit={self.limit}, p={self.p})'

    def aug_fn(self, image, keypoints):
        return self._random_translate(image, keypoints)

    def _random_translate(self, image, keypoints):
        if self.p > np.random.uniform(low=0, high=1):
            x_percent, y_percent = np.random.uniform(low=-self.limit, high=self.limit, size=2)
            return translate(image, keypoints, x_percent, y_percent)
        return image, keypoints


class RandomRotation(Augmentation):
    def __init__(self, angle, scale, p=0.5):
        self.angle = angle
        self.scale = scale
        self.p = p

    def __repr__(self):
        return f'RandomRotation(angle={self.angle}, scale={self.scale}, p={self.p})'

    def aug_fn(self, image, keypoints):
        return self._random_rotate(image, keypoints)

    def _random_rotate(self, image, keypoints):
        if self.p > np.random.uniform(low=0, high=1):
            rand_angle = np.random.uniform(low=-self.angle, high=self.angle)
            rand_scale = np.random.uniform(low=self.scale[0], high=self.scale[1])
            return rotate(image, keypoints, rand_angle, rand_scale)
        return image, keypoints


class Sequential(Augmentation):
    def __init__(self, augmentations: Sequence[Augmentation]):
        self.augmentations = augmentations

    def aug_fn(self, image, keypoints):
        for aug in self.augmentations:
            image, keypoints = aug.aug_fn(image, keypoints)
        return image, keypoints


def translate(image, keypoints, x_percent, y_percent):
    image_h, image_w = image.shape[:2]

    x = round(x_percent * image_w)
    y = round(y_percent * image_h)

    T = np.float32([[1, 0, x], [0, 1, y]])
    img_translation = cv2.warpAffine(image, T, (image_w, image_h))

    mask = keypoints[:, 2] != 0

    v_percent = np.array([x_percent, y_percent])
    keypoints[mask, :2] += v_percent

    # Check if after translation some kyepoint are out of the frame
    mask = np.logical_or((keypoints[:, :2] > np.ones(2)).any(axis=1), (keypoints[:, :2] < np.zeros(2)).any(axis=1))
    keypoints[mask] = np.array([0., 0., 0.])
    return img_translation, keypoints


def rotate(image, keypoints, angle, scale):
    image_h, image_w = image.shape[:2]

    mtx = cv2.getRotationMatrix2D((image_w // 2, image_h // 2), angle, scale=scale)

    img_translation = cv2.warpAffine(image, mtx, (image_w, image_h))

    mask = keypoints[:, 2] != 0

    mtx2 = cv2.getRotationMatrix2D((0.5, 0.5), angle, scale=scale)

    visible_kps = keypoints[mask, :2]

    keypoints[mask, :2] = mtx2.dot(np.hstack([visible_kps, np.ones((len(visible_kps), 1))]).T).T

    # Check if after translation some kyepoint are out of the frame
    mask = np.logical_or((keypoints[:, :2] > np.ones(2)).any(axis=1), (keypoints[:, :2] < np.zeros(2)).any(axis=1))
    keypoints[mask] = np.array([0., 0., 0.])
    return img_translation, keypoints
