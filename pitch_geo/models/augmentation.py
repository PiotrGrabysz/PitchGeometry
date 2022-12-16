import cv2
import numpy as np


def random_translate(image, keypoints, limit=0.5, p=0.5):
    if p > np.random.uniform(low=0, high=1):
        # print(f'DEBUG: randomly translating')
        x_percent, y_percent = np.random.uniform(low=-limit, high=limit, size=2)
        return translate(image, keypoints, x_percent, y_percent)
    return image, keypoints


def translate(image, keypoints, x_percent, y_percent):
    image_h, image_w = image.shape[:2]

    # print(f'{image.shape=}')
    # print(f'{keypoints.shape=}')
    # print(f'{image=}')

    x = round(x_percent * image_w)
    y = round(y_percent * image_h)

    # print(f'DEBUG: randomly translating by {x=} {y=}')

    T = np.float32([[1, 0, x], [0, 1, y]])
    img_translation = cv2.warpAffine(image, T, (image_w, image_h))

    # mask = keypoints.sum(axis=-1) != 0
    mask = keypoints[:, 2] != 0
    # print(f'DEBUG {mask=}')

    v_percent = np.array([x_percent, y_percent])
    keypoints[mask, :2] += v_percent
    # print(f'DEBUG: {keypoints[mask, :2]=}')
    # keypoints += v_percent

    # Check if after translation some kyepoint are out of the frame
    mask = np.logical_or((keypoints[:, :2] > np.ones(2)).any(axis=1), (keypoints[:, :2] < np.zeros(2)).any(axis=1))
    keypoints[mask] = np.array([0., 0., 0.])
    # keypoints[mask] = v_percent'
    return img_translation, keypoints


def rotate(image, keypoints, angle, scale):
    image_h, image_w = image.shape[:2]

    mtx = cv2.getRotationMatrix2D((image_w // 2, image_h // 2), angle, scale=scale)

    # T = np.float32([[1, 0, x], [0, 1, y]])
    img_translation = cv2.warpAffine(image, mtx, (image_w, image_h))

    mask = keypoints[:, 2] != 0

    mtx2 = cv2.getRotationMatrix2D((0.5, 0.5), angle, scale=scale)

    visible_kps = keypoints[mask, :2]

    keypoints[mask, :2] = mtx2.dot(np.hstack([visible_kps, np.ones((len(visible_kps), 1))]).T).T

    # keypoints += v_percent

    # Check if after translation some kyepoint are out of the frame
    mask = np.logical_or((keypoints[:, :2] > np.ones(2)).any(axis=1), (keypoints[:, :2] < np.zeros(2)).any(axis=1))
    keypoints[mask] = np.array([0., 0., 0.])
    # keypoints[mask] = v_percent
    # print(f'DEBUG: rotate done.')
    # print(f'DEBUG: {img_translation.shape=}')
    return img_translation, keypoints


def random_rotate(image, keypoints, angle, scale, p=0.5):
    # print(f'DEBUG getting {image.shape=}')
    if p > np.random.uniform(low=0, high=1):
        rand_angle = np.random.uniform(low=-angle, high=angle)
        # if scale == 1.0:
        #     rand_scale = 1.0
        # elif len(scale) == 2:
        #     rand_scale = np.random.uniform(low=scale[0], high=scale[1])
        # else:
        #     raise ValueError('Argument scale has to be 1.0 or tuple of two floats.')
        rand_scale = np.random.uniform(low=scale[0], high=scale[1])
        return rotate(image, keypoints, rand_scale, rand_scale)
    return image, keypoints


def set_shapes(img, labels, img_shape, labels_shape):
    img.set_shape(img_shape)
    labels.set_shape(labels_shape)
    return img, labels
