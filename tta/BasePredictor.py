import tifffile as tif
import os
import cv2
from .utils import *
from .exceptions import *
import numpy as np
from abc import ABC, abstractmethod
import json
import warnings


class BasePredictor(ABC):
    """
    An abstract class (wrapper for your model) to apply test time augmentation (TTA)
    """

    def __init__(self, conf):
        """
        Class constructor

        :param conf: configuration (json string or file path)
        """

        self._parse_conf(conf)

    @abstractmethod
    def predict_patches(self, patches):
        """
        Virtual method uses your model to predict patches
        :param patches: input patches to model for prediction
        :type patches: numpy.ndarray

        :return: prediction on these patches
        :rtype: numpy.ndarray
        """
        pass

    # @abstractmethod
    # def preprocess(self,img):
    # 	"""
    # 	Virtual method to preprocess image before passing to model (normalize, contrast enhancement, ...)
    # 	:param img: input image just after reading it

    # 	:returns: processed image
    # 	"""
    # 	pass

    # @abstractmethod
    # def postprocess(self,pred):
    # 	"""
    # 	Virtual method to postprocess image after model prediction (reverse normalization, clipping, ...)
    # 	:param pred: image predicted using model

    # 	:returns: processed image
    # 	"""
    # 	pass

    def _parse_conf(self, conf):
        """
        Parse the configuration file
        :param conf: configuration (json string or file path)
        """
        try:
            loaded = json.loads(conf)
        except:
            with open(conf) as f:
                loaded = json.load(f)

        if "augs" in loaded:
            self.augs = loaded["augs"]
            for aug in self.augs:
                if aug not in AUGS:
                    raise AugmentationUnrecognized('Unrecognized augmentation: %s in configuration.' % aug)
        else:
            warnings.warn('No "augs" found in configuration file. No augmentations will be used.', SyntaxWarning)
            self.augs = ["NO"]

        if "mean" in loaded:
            self.mean = loaded["mean"]
            if self.mean not in MEANS:
                raise MeanUnrecognized('Unrecognized mean: %s in configuration.' % self.mean)
        else:
            warnings.warn('No "mean" found in configuration file. "ARITH" mean will be used.', SyntaxWarning)
            self.mean = "ARITH"

        if "bits" in loaded:
            self.bits = loaded["bits"]
        else:
            warnings.warn('No "bits" found in configuration file. 8-bits will be used.', SyntaxWarning)
            self.bits = 8

    def apply_aug(self, img):
        """
        Apply augmentations to the supplied image
        :param img: original image before augmentation

        :returns: a set of augmentations of original image
        """
        aug_patch = np.zeros((len(self.augs), *img.shape), dtype=img.dtype)
        for i, aug in enumerate(self.augs):
            aug_patch[i] = apply(aug, img, self.bits)
        return aug_patch

    @abstractmethod
    def reverse_aug(self, aug_patch):
        """
        Reverse augmentations applied and calculate their combined mean
        :param aug_patch: set of prediction of the model to different augmentations

        :returns: single combined patch
        """
        pass

    @abstractmethod
    def _predict_single(self, img, overlap=0):
        """
        predict single image
        :param img: image to predict
        :param overlap: overlap size between patches in prediction of large image (default = 0)

        :return: prediction on the image
        """
        pass

    def predict_images(self, imgs, overlap=0):
        """
        predict a set of images
        :param imgs: a list of images to predict
        :param overlap: overlap size between patches in prediction of large image (default = 0)

        :return: predictions of all images
        """
        preds = []
        for img in imgs:
            if len(img.shape) == 4 and img.shape[0] == 1:
                img = img[0, :, :, :]
            pred = self._predict_single(img)
            preds.append(pred)
        return np.array(preds)

# def predict_dir(self,in_path,out_path,overlap=0,extension='.png'):
# 	"""
# 	Predict all images in directory

# 	:param in_path: directory where original images exist
# 	:param out_path: directory where predictions should be saved
# 	:param overlap: overlap size between patches in prediction (default = 0)
# 	:param extension: extension of saved images (default = '.png')

# 	"""
# 	for f in os.listdir(in_path):
# 		if f.split('.')[-1].lower() not in EXTENSIONS:
# 			continue

# 		if f.endswith('.tif') or f.endswith('.tiff'):
# 			img = tif.imread(os.path.join(in_path,f))
# 		else:
# 			img = cv2.imread(os.path.join(in_path,f))

# 		preprocessed = self.preprocess(img)
# 		if len(preprocessed.shape) == 4:
# 			preprocessed = preprocessed[0,:,:,:]

# 		pred = self.predict_single(preprocessed,overlap=overlap)

# 		out_filename = f.split('.')[0] + extension
# 		processed = self.postprocess(pred)
# 		cv2.imwrite(os.path.join(out_path,out_filename),processed)
