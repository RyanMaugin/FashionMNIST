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

# Start training the model and run training for 5 epochs (seems to give good results with little overfitting)
model.fit(train_images, train_labels, epochs=5)

# Evaluating accurancy of model by testing it on the test data
test_loss, test_accurancy = model.evaluate(test_images, test_labels)
print("Test Loss: {} \nTest Accurancy: {}".format(test_loss, test_accurancy))

# Now that the model is trained and evaluated for accuracy we can now use it to make predictions if performance is good
predictions = model.predict(test_images)

# Plot the first 25 predicted images with their predicted labels 
# along with colour code green for correct and red for incorrect prediction
plt.figure(figsize=(10, 10))
for x in range(25):
    plt.subplot(5, 5, x + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[x], cmap=plt.cm.binary) # Plot image
    predicted_label = np.argmax(predictions[x])    # Get classification with highest confidence
    true_label = test_labels[x]                    # Get correct label

    # Check if prediction was correct and if so make label green
    if predicted_label == true_label: colour = 'green'
    else: colour = 'red'
    
    # Plot the label under image with colour corresponding to correct prediction or not
    plt.xlabel(
        "{} ({})".format(class_names[predicted_label], class_names[true_label]),
        color=colour
    )

plt.show() # Reveal the plotting with the predictions