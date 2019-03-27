from .BasePredictor import BasePredictor
import numpy as np


class ClassPredictor(BasePredictor):
    def __init__(self, conf):
        """
        Initialize class

        :param conf: configuration (json string or file path)
        """
        super().__init__(conf=conf)

    def reverse_aug(self, aug_patch):
        """
        Reverse augmentations applied and calculate their combined mean
        :param aug_patch: set of prediction of the model to different augmentations

        :returns: single combined patch
        """
        if self.mean == "ARITH":
            return np.mean(aug_patch, axis=0)
        elif self.mean == "GEO":
            product = np.prod(aug_patch, axis=0)
            return processed ** (1. / len(self.augs))

    def _predict_single(self, img):
        """
        predict single image
        :param img: image to predict

        :return: prediction on the image
        """
        aug_patches = self.apply_aug(img)
        pred = self.predict_patches(aug_patches)
        return self.reverse_aug(pred)