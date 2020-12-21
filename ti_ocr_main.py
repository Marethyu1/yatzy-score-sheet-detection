""" This program detects and classifies numbers in a yatzy sheet  """
import argparse
import tensorflow as tf
import numpy as np

import cv2
import ocr_sheet
import cv_utils

# print('Reading tensorflow model...')
# print(tf.__version__)
# predict_model = tf.keras.models.load_model('./models/model_tensorflow')
# predict_model.summary()

img_path = './assets/sample_sheets/tiwhole.png'
num_rows = 19  # specify num rows for the yatzy sheet, original yatzy has 19

print("Reading image from path", img_path)
input_img = cv2.imread(img_path)

cv_utils.show_window('raw_image', input_img, debug=True)

img_yatzy_sheet, img_binary_yatzy_sheet, img_binary_only_numbers, yatzy_cells_bounding_rects = ocr_sheet.generate_ti_sheet(input_img, num_rows_in_grid=num_rows)

# Debugging step
img_yatzy_cells = img_yatzy_sheet.copy()
cv_utils.draw_bounding_rects(img_yatzy_cells, yatzy_cells_bounding_rects)
cv_utils.show_window('img_yatzy_cells', img_yatzy_cells)



