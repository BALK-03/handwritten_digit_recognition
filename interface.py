import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


model = tf.keras.models.load_model('my_model')


def make_prediction(image_path):
  img = cv.imread(image_path)[:, :, 0]
  img = np.invert(np.array([img]))
  prediction = model.predict(img)
  return np.argmax(prediction)





# choose the path of your picture - In my case I use the folder test_digits
number = 7
image_path = f"test_digits/test-{number}.png"


img = plt.imread(image_path)
plt.imshow(img)

cv.waitKey(0)


print(f"The number is probably: {make_prediction(image_path)}")