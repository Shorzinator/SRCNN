import sys
import keras
import cv2
import numpy as np
import matplotlib
import skimage
import math
import os

print('Python {}'.format(sys.version))
print('Keras {}'.format(keras.__version__))
print('OpenCV {}'.format(cv2.__version__))
print('NumPy {}'.format(np.__version__))
print('MatPlotLib {}'.format(matplotlib.__version__))
print('Skimage {}'.format(skimage.__version__))

from keras.models import Sequential
from keras.layers import Conv2D, Input
from keras.optimizers import SGD, Adam
from skimage.measure import compare_ssim as ssim
from matplotlib import pyplot as plt
