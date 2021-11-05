import cv2
import numpy as np
from scipy.stats import truncnorm


def fill(img, h, w):
    img = cv2.resize(img, (w, h), cv2.INTER_CUBIC)
    return img

def zoom(x, h_start, h_end, w_start, w_end):
    h, w = x.shape[:2]
    x = x[h_start:h_end, w_start:w_end]
    out = fill(x, h, w)
    return out

def scale_brightness(x, scale):
    return x * scale

def translate_brightness(x, translation):
    return (x - translation)

def horizontal_flip(x):
    return cv2.flip(x, 1)

def vertical_flip(x):
    return cv2.flip(x, 0)


class Augment(object):
    def __init__(self,
                 segment_label_type='discrete',
                 zoom_prob=0.3,
                 zoom_range=[0.8, 1.],
                 scale_prob=0.5,
                 scale_range=[0.9, 1.11],
                 translation_prob=0.5,
                 translation_std=0.02,
                 flip_prob=0.5):

        self.segment_label_type = segment_label_type
        self.zoom_prob = zoom_prob
        self.zoom_range = zoom_range
        self.scale_prob = scale_prob
        self.scale_range = scale_range
        self.translation_prob = translation_prob
        self.translation_std = translation_std
        self.flip_prob = flip_prob
        self.rv = truncnorm(-3, 3)



    def __call__(self, x, y=None, weight=None):
        x_shape = x.shape
        x_dtype = x.dtype
        if not y is None and not weight is None:
          y_shape = y.shape
          y_dtype = y.dtype
          weight_shape = weight.shape
          weight_dtype = weight.dtype

        h, w = x_shape[:2]
        if np.random.rand() < self.zoom_prob:
            h_zoom_ratio = np.random.uniform(*self.zoom_range)
            h_start = int(np.random.uniform(0, h * (1 - h_zoom_ratio)))
            h_end = h_start + int(h * h_zoom_ratio)

            w_zoom_ratio = np.random.uniform(*self.zoom_range)
            w_start = int(np.random.uniform(0, w * (1 - w_zoom_ratio)))
            w_end = w_start + int(w * w_zoom_ratio)
            x = zoom(x, h_start, h_end, w_start, w_end)
            if not y is None and not weight is None:
                weight = zoom(weight, h_start, h_end, w_start, w_end)
                y = zoom(y.astype(float), h_start, h_end, w_start, w_end)
                if self.segment_label_type == 'discrete':
                    weight[np.where(y != y.astype(int))] = 0
                    y[np.where(y != y.astype(int))] = 0
        if np.random.rand() < self.scale_prob:
            scale_ratio = np.random.uniform(*self.scale_range)
            x = scale_brightness(x, scale_ratio)
        if np.random.rand() < self.translation_prob:
            translation = self.rv.rvs() * self.translation_std
            x = translate_brightness(x, translation)
        if np.random.rand() < self.flip_prob:
            x = horizontal_flip(x)
            if not y is None and not weight is None:
                y = horizontal_flip(y)
                weight = horizontal_flip(weight)
        if np.random.rand() < self.flip_prob:
            x = vertical_flip(x)
            if not y is None and not weight is None:
                y = vertical_flip(y)
                weight = vertical_flip(weight)

        if not y is None and not weight is None:
            x = x.reshape(x_shape).astype(x_dtype)
            y = y.reshape(y_shape).astype(y_dtype)
            weight = weight.reshape(weight_shape).astype(weight_dtype)
            return x, y, weight
        else:
            x = x.reshape(x_shape).astype(x_dtype)
            return x
