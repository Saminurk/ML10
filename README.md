# ML10
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
 
# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
 
# One-hot encode the labels
encoder = OneHotEncoder(sparse_output=False)
y_onehot = encoder.fit_transform(y.reshape(-1, 1))
 
# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
 
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_onehot, test_size=0.3, random_state=42)
 
# Function to create an MLP model with custom optimizer and activation function
def create_mlp_model(optimizer='adam', activation='relu'):
   model = Sequential([
       Dense(16, input_dim=X_train.shape[1], activation=activation),
       Dense(8, activation=activation),
       Dense(y_train.shape[1], activation='softmax')  # Output layer with softmax for classification
   ])
   model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
   return model
 
# Train and evaluate the model with Adam optimizer and ReLU activation
model_adam_relu = create_mlp_model(optimizer='adam', activation='relu')
history_adam_relu = model_adam_relu.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)
y_pred_adam_relu = np.argmax(model_adam_relu.predict(X_test), axis=1)
accuracy_adam_relu = accuracy_score(np.argmax(y_test, axis=1), y_pred_adam_relu)
 
# Train and evaluate the model with SGD optimizer and tanh activation
model_sgd_tanh = create_mlp_model(optimizer='sgd', activation='tanh')
history_sgd_tanh = model_sgd_tanh.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)
y_pred_sgd_tanh = np.argmax(model_sgd_tanh.predict(X_test), axis=1)
accuracy_sgd_tanh = accuracy_score(np.argmax(y_test, axis=1), y_pred_sgd_tanh)
 
# Print the results
print("Accuracy with Adam optimizer and ReLU activation: {:.2f}%".format(accuracy_adam_relu * 100))
print("Accuracy with SGD optimizer and Tanh activation: {:.2f}%".format(accuracy_sgd_tanh * 100))
 
# Plot training histories for comparison
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.plot(history_adam_relu.history['accuracy'], label='Adam + ReLU')
plt.plot(history_sgd_tanh.history['accuracy'], label='SGD + Tanh')
plt.title('Training Accuracy Comparison')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()
