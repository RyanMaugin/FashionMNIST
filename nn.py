# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# Reference & Download the Dataset
fashion_mnist = keras.datasets.fashion_mnist

# Load the training images and label along with test set
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Classifications of data
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print("Training Set :", train_images.shape) # Size of training set (0) along with resolution of images (1, 2)
print("Test Set     :", train_images.shape) # Size of test set (0) along with resolution of images (1, 2)

# Normalising/Feature scaling the value of each pixel intensity
train_images = train_images / 255.0
test_images = test_images / 255.0

