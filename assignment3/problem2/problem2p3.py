import tensorflow as tf #Import the tensorflow module
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


mnist = tf.keras.datasets.mnist #Loads the MNIST dataset

(x_train, y_train),(x_test, y_test) = mnist.load_data() #Loads the training and test data for the MNIST dataset
x_train, x_test = x_train / 255.0, x_test / 255.0 # Pre-processes training and testing data

model = tf.keras.models.Sequential([    #Instantiates a sequential model
  tf.keras.layers.Flatten(input_shape=(28, 28)), # Flattens data into a vector
  tf.keras.layers.Dense(10, activation=tf.nn.softmax) # Adds a layer with 10 units using a softmax activation function
])
model.compile(optimizer='adam', # Optimizes the model using the Adam algorithm
              loss='sparse_categorical_crossentropy', # Selects the loss function
              metrics=['accuracy']) # Calculates the accuracy of the model

model.fit(x_train, y_train, epochs=5) # Trains the model with 5 iterations
model.evaluate(x_test, y_test) # Returns loss and accuracy values for the model\

weights = model.get_weights()[0]

for i in range (0, 10):
    number = np.empty(784)
    for k in range(0, 784):
        number[k] = weights[k][i]

    #Normailze data
    number = (number[:]-min(number[:]))/(max(number[:])-min(number[:]))

    #Put data into image
    img = np.reshape(number, (28, 28))

    imgplot = plt.imshow(img, cmap="gray")
    plt.show()