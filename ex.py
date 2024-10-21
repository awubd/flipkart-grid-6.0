import cv2
import numpy as np
import tensorflow as tf  # Use TensorFlow directly instead of standalone Keras
from tensorflow.keras.models import load_model
import time
import os
import re
import glob
import pytesseract
from ocr import ImageOCR
from postprocessing import TextPostprocessing
from fuzzywuzzy import process
from tensorflow.keras.preprocessing.image import load_img, img_to_array  # Update this too

# Set the path for Tesseract OCR executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load pre-trained CNN models using tensorflow.keras
cnn_model1 = tf.keras.models.load_model('brand_pred_4.keras')
cnn_model2 = tf.keras.models.load_model('fruit_veg_cnn_model.keras')

