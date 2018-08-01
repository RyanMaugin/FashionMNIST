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
test_images  = test_images / 255.0

# Create the neural network with sequential flow and 3 layers
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),       # Input layer with 784 input nodes
    keras.layers.Dense(128, activation=tf.nn.relu),   # Hidden layer with 128 densley connected neurons using relu for activation
    keras.layers.Dense(10, activation=tf.nn.softmax)  # Output layer with 10 possible outputs using softmax (sigmoid) activation
])

# Model configuration of optimizer, loss function and metric of the neural net
model.compile(
    optimizer = tf.train.AdamOptimizer(),             # Modified version SGD which implements adaptive learning rates for each weight instead of one for all
    loss      = 'sparse_categorical_crossentropy',    # Our targets are represented as integers so we use 'sparse_categorical_crossentropy'
    metrics   = ['accuracy']                          # Measure network accuracy by fraction of correctly classified images
)

# Start training the model and run training for 5 epochs
model.fit(train_images, train_labels, epochs=5)

# Evaluating accurancy of model by testing it on the test data
test_loss, test_accurancy = model.evaluate(test_images, test_labels)
print("Test Loss: {} \nTest Accurancy: {}".format(test_loss, test_accurancy))

# Now that the model is trained and evaluated for accurancy we can now use it to make predictions if performance is good
predictions = model.predict(test_images)

# Check prediction and see if classified first image correctly
print(predictions[0])
print("Prediction:", np.argmax(predictions[0]))
print("Answer:", test_labels[0])