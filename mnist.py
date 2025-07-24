import tensorflow as tf
import matplotlib.pyplot as plt

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize values to 0-1
x_train = x_train / 255.0
x_test = x_test / 255.0

# Build a simple Neural Network
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),  # Convert 2D to 1D
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),  # Prevent overfitting
    tf.keras.layers.Dense(10, activation='softmax')  # Output layer for 10 digits
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, validation_split=0.1)
model.save("saved_model.keras")


# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print('\nTest accuracy:', test_acc)

# Predict on one image
plt.imshow(x_test[0], cmap='gray')
plt.title(f"Actual: {y_test[0]} | Predicted: {model.predict(x_test[:1]).argmax()}")
plt.show()
