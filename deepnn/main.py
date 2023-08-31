import numpy as np
from PIL import Image

from deepnn.deep_nn import predict
from deepnn.dnn_utils import load_data
from deepnn.l_layer_model import L_layer_model

train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

n_x = 12288     # num_px * num_px * 3
n_h = 7
n_y = 1
layers_dims = (n_x, n_h, n_y)

parameters, costs = L_layer_model(train_x, train_y, layers_dims, 0.0065,  3000, True)

pred_train = predict(train_x, train_y, parameters)

pred_test = predict(test_x, test_y, parameters)

my_image = "../logistic_regression/images/eu.jpeg" # change this to the name of your image file
my_label_y = [1] # the true class of your image (1 -> cat, 0 -> non-cat)


m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

fname = my_image
image = np.array(Image.open(fname).resize((num_px, num_px)))
# plt.imshow(image)
image = image / 255.
image = image.reshape((1, num_px * num_px * 3)).T

my_predicted_image = predict(image, my_label_y, parameters)