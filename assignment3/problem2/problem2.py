import tensorflow as tf #Import the tensorflow module
mnist = tf.keras.datasets.mnist #Loads the MNIST dataset

(x_train, y_train),(x_test, y_test) = mnist.load_data() #Loads the training and test data for the MNIST dataset
x_train, x_test = x_train / 255.0, x_test / 255.0 # Pre-processes training and testing data

model = tf.keras.models.Sequential([    #Instantiates a sequential model
  tf.keras.layers.Flatten(input_shape=(28, 28)), # Flattens data into a vector
  tf.keras.layers.Dense(512, activation=tf.nn.relu), # Adds a layer with 512 units using a reLU activation function
  tf.keras.layers.Dropout(0.2), # Applies dropout to the input at a rate of 20%
  tf.keras.layers.Dense(10, activation=tf.nn.softmax) # Adds a layer with 10 units using a softmax activation function
])
model.compile(optimizer='adam', # Optimizes the model using the Adam algorithm
              loss='sparse_categorical_crossentropy', # Selects the loss function
              metrics=['accuracy']) # Calculates the accuracy of the model

model.fit(x_train, y_train, epochs=5) # Trains the model with 5 iterations
model.evaluate(x_test, y_test) # Returns loss and accuracy values for the model