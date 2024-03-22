1 st program 

import numpy as np 

  

def sigmoid(z): 

    return 1 / (1 + np.exp(-z)) 

  

def initialize_parameters(input_size, hidden_size, output_size): 

    np.random.seed(42) 

    W1 = np.random.randn(hidden_size, input_size) * 0.01 

    b1 = np.zeros((hidden_size, 1)) 

    W2 = np.random.randn(output_size, hidden_size) * 0.01 

    b2 = np.zeros((output_size, 1)) 

  

    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2} 

    return parameters 

  

def forward_propagation(X, parameters): 

    W1, b1, W2, b2 = parameters['W1'], parameters['b1'], parameters['W2'], parameters['b2'] 

  

    Z1 = np.dot(W1, X) + b1 

    A1 = np.tanh(Z1) 

    Z2 = np.dot(W2, A1) + b2 

    A2 = sigmoid(Z2) 

  

    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2} 

    return A2, cache 

  

def compute_cost(A2, Y): 

    m = Y.shape[1] 

    cost = (-1/m) * np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2)) 

    return cost 

  

def backward_propagation(parameters, cache, X, Y): 

    m = X.shape[1] 

  

    W1, W2 = parameters['W1'], parameters['W2'] 

    A1, A2 = cache['A1'], cache['A2'] 

  

    dZ2 = A2 - Y 

    dW2 = (1/m) * np.dot(dZ2, A1.T) 

    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True) 

    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2)) 

    dW1 = (1/m) * np.dot(dZ1, X.T) 

    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True) 

  

    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2} 

    return grads 

  

def update_parameters(parameters, grads, learning_rate=0.01): 

    W1, b1, W2, b2 = parameters['W1'], parameters['b1'], parameters['W2'], parameters['b2'] 

    dW1, db1, dW2, db2 = grads['dW1'], grads['db1'], grads['dW2'], grads['db2'] 

  

    W1 -= learning_rate * dW1 

    b1 -= learning_rate * db1 

    W2 -= learning_rate * dW2 

    b2 -= learning_rate * db2 

  

    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2} 

    return parameters 

  

def model(X, Y, input_size, hidden_size, output_size, num_iterations=10000, learning_rate=0.01): 

    parameters = initialize_parameters(input_size, hidden_size, output_size) 

  

    for i in range(num_iterations): 

         

        A2, cache = forward_propagation(X, parameters) 

  

         

        cost = compute_cost(A2, Y) 

  

         

        grads = backward_propagation(parameters, cache, X, Y) 

  

         

        parameters = update_parameters(parameters, grads, learning_rate) 

  

         

        if i % 1000 == 0: 

            print(f"Cost after iteration {i}: {cost}") 

  

    return parameters 

  

  

input_size = 2 

hidden_size = 4 

output_size = 1 

  

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T  

Y = np.array([[0, 1, 1, 0]])   

  

trained_parameters = model(X, Y, input_size, hidden_size, output_size, num_iterations=10000, learning_rate=0.01) 

 

2nd program 

#program =2 

import numpy as np 

import matplotlib.pyplot as plt 

from sklearn.datasets import make_moons 

  

class NeuralNetwork: 

    def __init__(self, input_size, hidden_size, output_size): 

        self.input_size = input_size 

        self.hidden_size = hidden_size 

        self.output_size = output_size 

  

        # Initialize parameters 

        self.W1 = np.random.randn(hidden_size, input_size) * 0.01 

        self.b1 = np.zeros((hidden_size, 1)) 

        self.W2 = np.random.randn(output_size, hidden_size) * 0.01 

        self.b2 = np.zeros((output_size, 1)) 

  

    def forward_propagation(self, X): 

        # Forward pass 

        self.Z1 = np.dot(self.W1, X) + self.b1 

        self.A1 = np.tanh(self.Z1) 

        self.Z2 = np.dot(self.W2, self.A1) + self.b2 

        self.A2 = self.sigmoid(self.Z2) 

        return self.A2 

  

    def backward_propagation(self, X, Y): 

        m = X.shape[1]  # Number of samples 

  

        # Backward pass 

        dZ2 = self.A2 - Y 

        dW2 = (1 / m) * np.dot(dZ2, self.A1.T) 

        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True) 

        dZ1 = np.dot(self.W2.T, dZ2) * (1 - np.power(self.A1, 2))  # derivative of tanh 

        dW1 = (1 / m) * np.dot(dZ1, X.T) 

        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True) 

  

        # Update parameters 

        self.W2 -= dW2 

        self.b2 -= db2 

        self.W1 -= dW1 

        self.b1 -= db1 

  

    def train(self, X, Y, num_epochs, learning_rate): 

        for epoch in range(num_epochs): 

            # Forward propagation 

            predictions = self.forward_propagation(X) 

  

            # Compute cross-entropy loss 

            loss = self.cross_entropy_loss(Y, predictions) 

  

            # Backward propagation 

            self.backward_propagation(X, Y) 

  

            # Print the loss every 100 epochs 

            if epoch % 100 == 0: 

                print(f'Epoch {epoch}, Loss: {loss}') 

  

    def sigmoid(self, Z): 

        return 1 / (1 + np.exp(-Z)) 

  

    def cross_entropy_loss(self, Y, A): 

        m = Y.shape[1]  # Number of samples 

        return -(1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)) 

  

# Load and preprocess the data 

X, Y = make_moons(n_samples=1000, noise=0.2, random_state=42) 

X = X.T 

Y = Y.reshape(1, -1) 

  

# Plot the data 

plt.scatter(X[0, :], X[1, :], c=Y.ravel(), cmap=plt.cm.Spectral) 

plt.xlabel('Feature 1') 

plt.ylabel('Feature 2') 

plt.title('Planar Data Classification Dataset') 

plt.show() 

  

# Example usage 

input_size = 2 

hidden_size = 4 

output_size = 1 

  

model = NeuralNetwork(input_size, hidden_size, output_size) 

model.train(X, Y, num_epochs=1000, learning_rate=0.01) 

 

3rd program 

#Program to Build your neural network: ·  

#•	Use non-linear units like ReLU to improve your model ·  

#•	Build a deeper neural network (with more than 1 hidden layer) ·  

#•	Implement an easy-to-use neural network class 

  

import tensorflow as tf 

from tensorflow.keras import layers, models 

from tensorflow.keras.datasets import mnist 

from tensorflow.keras.utils import to_categorical 

import matplotlib.pyplot as plt 

# Load and preprocess the MNIST dataset 

(train_images, train_labels), (test_images, test_labels) = mnist.load_data() 

train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255 

test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255 

train_labels = to_categorical(train_labels) 

test_labels = to_categorical(test_labels) 

# Define the NeuralNetwork class 

class NeuralNetwork: 

    def __init__(self, input_shape, num_classes): 

        self.model = self.build_model(input_shape, num_classes) 

    def build_model(self, input_shape, num_classes): 

        model = models.Sequential() 

        model.add(layers.Flatten(input_shape=input_shape)) 

        model.add(layers.Dense(128, activation='relu')) 

        model.add(layers.Dense(64, activation='relu')) 

        model.add(layers.Dense(num_classes, activation='softmax')) 

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) 

        return model 

  

    def train(self, train_images, train_labels, epochs=5, batch_size=64, validation_split=0.1): 

        history = self.model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size, validation_split=validation_split) 

        return history 

    def evaluate(self, test_images, test_labels): 

        return self.model.evaluate(test_images, test_labels) 

    def predict(self, images): 

        return self.model.predict(images) 

# Create an instance of the NeuralNetwork class 

input_shape = (28, 28, 1) 

num_classes = 10 

nn = NeuralNetwork(input_shape, num_classes) 

# Train the neural network 

history = nn.train(train_images, train_labels, epochs=5) 

  

# Evaluate the model on the test set 

test_loss, test_acc = nn.evaluate(test_images, test_labels) 

print(f'Test accuracy: {test_acc}') 

  

# Make predictions on a few test images 

predictions = nn.predict(test_images[:5]) 

  

# Plot the first few test images and their predicted labels 

plt.figure(figsize=(10, 4)) 

for i in range(5): 

    plt.subplot(1, 5, i + 1) 

    plt.imshow(test_images[i, :, :, 0], cmap='gray') 

    plt.title(f'Predicted: {tf.argmax(predictions[i])}') 

    plt.axis('off') 

plt.show() 

 

 

4th program 

#4th program=•	Program for image classification using deep neural network 

import tensorflow as tf 

from tensorflow.keras import layers, models 

from tensorflow.keras.datasets import mnist 

from tensorflow.keras.utils import to_categorical 

import matplotlib.pyplot as plt 

# Load and preprocess the MNIST dataset 

(train_images, train_labels), (test_images, test_labels) = mnist.load_data() 

train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255 

test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255 

train_labels = to_categorical(train_labels) 

test_labels = to_categorical(test_labels) 

# Define the deep neural network architecture 

model = models.Sequential() 

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1))) 

model.add(layers.MaxPooling2D((2, 2))) 

model.add(layers.Conv2D(64, (3, 3), activation='relu')) 

model.add(layers.MaxPooling2D((2, 2))) 

model.add(layers.Conv2D(64, (3, 3), activation='relu')) 

model.add(layers.Flatten()) 

model.add(layers.Dense(64, activation='relu')) 

model.add(layers.Dense(10, activation='softmax')) 

  

# Compile the model 

model.compile(optimizer='adam', 

              loss='categorical_crossentropy', 

              metrics=['accuracy']) 

# Train the model 

model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.2) 

# Evaluate the model on the test set 

test_loss, test_acc = model.evaluate(test_images, test_labels) 

print(f'Test accuracy: {test_acc}') 

# Make predictions on a few test images 

predictions = model.predict(test_images[:5]) 

# Display the test images and their predicted labels 

for i in range(5): 

    plt.imshow(test_images[i].reshape(28, 28), cmap='gray') 

    plt.title(f'Predicted: {tf.argmax(predictions[i])}, Actual: {tf.argmax(test_labels[i])}') 

    plt.show()1 st program 

import numpy as np 

  

def sigmoid(z): 

    return 1 / (1 + np.exp(-z)) 

  

def initialize_parameters(input_size, hidden_size, output_size): 

    np.random.seed(42) 

    W1 = np.random.randn(hidden_size, input_size) * 0.01 

    b1 = np.zeros((hidden_size, 1)) 

    W2 = np.random.randn(output_size, hidden_size) * 0.01 

    b2 = np.zeros((output_size, 1)) 

  

    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2} 

    return parameters 

  

def forward_propagation(X, parameters): 

    W1, b1, W2, b2 = parameters['W1'], parameters['b1'], parameters['W2'], parameters['b2'] 

  

    Z1 = np.dot(W1, X) + b1 

    A1 = np.tanh(Z1) 

    Z2 = np.dot(W2, A1) + b2 

    A2 = sigmoid(Z2) 

  

    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2} 

    return A2, cache 

  

def compute_cost(A2, Y): 

    m = Y.shape[1] 

    cost = (-1/m) * np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2)) 

    return cost 

  

def backward_propagation(parameters, cache, X, Y): 

    m = X.shape[1] 

  

    W1, W2 = parameters['W1'], parameters['W2'] 

    A1, A2 = cache['A1'], cache['A2'] 

  

    dZ2 = A2 - Y 

    dW2 = (1/m) * np.dot(dZ2, A1.T) 

    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True) 

    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2)) 

    dW1 = (1/m) * np.dot(dZ1, X.T) 

    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True) 

  

    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2} 

    return grads 

  

def update_parameters(parameters, grads, learning_rate=0.01): 

    W1, b1, W2, b2 = parameters['W1'], parameters['b1'], parameters['W2'], parameters['b2'] 

    dW1, db1, dW2, db2 = grads['dW1'], grads['db1'], grads['dW2'], grads['db2'] 

  

    W1 -= learning_rate * dW1 

    b1 -= learning_rate * db1 

    W2 -= learning_rate * dW2 

    b2 -= learning_rate * db2 

  

    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2} 

    return parameters 

  

def model(X, Y, input_size, hidden_size, output_size, num_iterations=10000, learning_rate=0.01): 

    parameters = initialize_parameters(input_size, hidden_size, output_size) 

  

    for i in range(num_iterations): 

         

        A2, cache = forward_propagation(X, parameters) 

  

         

        cost = compute_cost(A2, Y) 

  

         

        grads = backward_propagation(parameters, cache, X, Y) 

  

         

        parameters = update_parameters(parameters, grads, learning_rate) 

  

         

        if i % 1000 == 0: 

            print(f"Cost after iteration {i}: {cost}") 

  

    return parameters 

  

  

input_size = 2 

hidden_size = 4 

output_size = 1 

  

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T  

Y = np.array([[0, 1, 1, 0]])   

  

trained_parameters = model(X, Y, input_size, hidden_size, output_size, num_iterations=10000, learning_rate=0.01) 

 

2nd program 

#program =2 

import numpy as np 

import matplotlib.pyplot as plt 

from sklearn.datasets import make_moons 

  

class NeuralNetwork: 

    def __init__(self, input_size, hidden_size, output_size): 

        self.input_size = input_size 

        self.hidden_size = hidden_size 

        self.output_size = output_size 

  

        # Initialize parameters 

        self.W1 = np.random.randn(hidden_size, input_size) * 0.01 

        self.b1 = np.zeros((hidden_size, 1)) 

        self.W2 = np.random.randn(output_size, hidden_size) * 0.01 

        self.b2 = np.zeros((output_size, 1)) 

  

    def forward_propagation(self, X): 

        # Forward pass 

        self.Z1 = np.dot(self.W1, X) + self.b1 

        self.A1 = np.tanh(self.Z1) 

        self.Z2 = np.dot(self.W2, self.A1) + self.b2 

        self.A2 = self.sigmoid(self.Z2) 

        return self.A2 

  

    def backward_propagation(self, X, Y): 

        m = X.shape[1]  # Number of samples 

  

        # Backward pass 

        dZ2 = self.A2 - Y 

        dW2 = (1 / m) * np.dot(dZ2, self.A1.T) 

        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True) 

        dZ1 = np.dot(self.W2.T, dZ2) * (1 - np.power(self.A1, 2))  # derivative of tanh 

        dW1 = (1 / m) * np.dot(dZ1, X.T) 

        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True) 

  

        # Update parameters 

        self.W2 -= dW2 

        self.b2 -= db2 

        self.W1 -= dW1 

        self.b1 -= db1 

  

    def train(self, X, Y, num_epochs, learning_rate): 

        for epoch in range(num_epochs): 

            # Forward propagation 

            predictions = self.forward_propagation(X) 

  

            # Compute cross-entropy loss 

            loss = self.cross_entropy_loss(Y, predictions) 

  

            # Backward propagation 

            self.backward_propagation(X, Y) 

  

            # Print the loss every 100 epochs 

            if epoch % 100 == 0: 

                print(f'Epoch {epoch}, Loss: {loss}') 

  

    def sigmoid(self, Z): 

        return 1 / (1 + np.exp(-Z)) 

  

    def cross_entropy_loss(self, Y, A): 

        m = Y.shape[1]  # Number of samples 

        return -(1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)) 

  

# Load and preprocess the data 

X, Y = make_moons(n_samples=1000, noise=0.2, random_state=42) 

X = X.T 

Y = Y.reshape(1, -1) 

  

# Plot the data 

plt.scatter(X[0, :], X[1, :], c=Y.ravel(), cmap=plt.cm.Spectral) 

plt.xlabel('Feature 1') 

plt.ylabel('Feature 2') 

plt.title('Planar Data Classification Dataset') 

plt.show() 

  

# Example usage 

input_size = 2 

hidden_size = 4 

output_size = 1 

  

model = NeuralNetwork(input_size, hidden_size, output_size) 

model.train(X, Y, num_epochs=1000, learning_rate=0.01) 

 

3rd program 

#Program to Build your neural network: ·  

#•	Use non-linear units like ReLU to improve your model ·  

#•	Build a deeper neural network (with more than 1 hidden layer) ·  

#•	Implement an easy-to-use neural network class 

  

import tensorflow as tf 

from tensorflow.keras import layers, models 

from tensorflow.keras.datasets import mnist 

from tensorflow.keras.utils import to_categorical 

import matplotlib.pyplot as plt 

# Load and preprocess the MNIST dataset 

(train_images, train_labels), (test_images, test_labels) = mnist.load_data() 

train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255 

test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255 

train_labels = to_categorical(train_labels) 

test_labels = to_categorical(test_labels) 

# Define the NeuralNetwork class 

class NeuralNetwork: 

    def __init__(self, input_shape, num_classes): 

        self.model = self.build_model(input_shape, num_classes) 

    def build_model(self, input_shape, num_classes): 

        model = models.Sequential() 

        model.add(layers.Flatten(input_shape=input_shape)) 

        model.add(layers.Dense(128, activation='relu')) 

        model.add(layers.Dense(64, activation='relu')) 

        model.add(layers.Dense(num_classes, activation='softmax')) 

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) 

        return model 

  

    def train(self, train_images, train_labels, epochs=5, batch_size=64, validation_split=0.1): 

        history = self.model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size, validation_split=validation_split) 

        return history 

    def evaluate(self, test_images, test_labels): 

        return self.model.evaluate(test_images, test_labels) 

    def predict(self, images): 

        return self.model.predict(images) 

# Create an instance of the NeuralNetwork class 

input_shape = (28, 28, 1) 

num_classes = 10 

nn = NeuralNetwork(input_shape, num_classes) 

# Train the neural network 

history = nn.train(train_images, train_labels, epochs=5) 

  

# Evaluate the model on the test set 

test_loss, test_acc = nn.evaluate(test_images, test_labels) 

print(f'Test accuracy: {test_acc}') 

  

# Make predictions on a few test images 

predictions = nn.predict(test_images[:5]) 

  

# Plot the first few test images and their predicted labels 

plt.figure(figsize=(10, 4)) 

for i in range(5): 

    plt.subplot(1, 5, i + 1) 

    plt.imshow(test_images[i, :, :, 0], cmap='gray') 

    plt.title(f'Predicted: {tf.argmax(predictions[i])}') 

    plt.axis('off') 

plt.show() 

 

 

4th program 

#4th program=•	Program for image classification using deep neural network 

import tensorflow as tf 

from tensorflow.keras import layers, models 

from tensorflow.keras.datasets import mnist 

from tensorflow.keras.utils import to_categorical 

import matplotlib.pyplot as plt 

# Load and preprocess the MNIST dataset 

(train_images, train_labels), (test_images, test_labels) = mnist.load_data() 

train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255 

test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255 

train_labels = to_categorical(train_labels) 

test_labels = to_categorical(test_labels) 

# Define the deep neural network architecture 

model = models.Sequential() 

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1))) 

model.add(layers.MaxPooling2D((2, 2))) 

model.add(layers.Conv2D(64, (3, 3), activation='relu')) 

model.add(layers.MaxPooling2D((2, 2))) 

model.add(layers.Conv2D(64, (3, 3), activation='relu')) 

model.add(layers.Flatten()) 

model.add(layers.Dense(64, activation='relu')) 

model.add(layers.Dense(10, activation='softmax')) 

  

# Compile the model 

model.compile(optimizer='adam', 

              loss='categorical_crossentropy', 

              metrics=['accuracy']) 

# Train the model 

model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.2) 

# Evaluate the model on the test set 

test_loss, test_acc = model.evaluate(test_images, test_labels) 

print(f'Test accuracy: {test_acc}') 

# Make predictions on a few test images 

predictions = model.predict(test_images[:5]) 

# Display the test images and their predicted labels 

for i in range(5): 

    plt.imshow(test_images[i].reshape(28, 28), cmap='gray') 

    plt.title(f'Predicted: {tf.argmax(predictions[i])}, Actual: {tf.argmax(test_labels[i])}') 

    plt.show()
