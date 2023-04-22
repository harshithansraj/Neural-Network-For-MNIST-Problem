import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Load the MNIST dataset
(x_train_full, y_train_full), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize the pixel values to be between 0 and 1
x_train_full = x_train_full.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# Convert the labels to one-hot encoded vectors
y_train_full = keras.utils.to_categorical(y_train_full)
y_test = keras.utils.to_categorical(y_test)

# Split the training set into training and validation sets
train_size = int(0.8 * len(x_train_full))
x_train, y_train = x_train_full[:train_size], y_train_full[:train_size]
x_valid, y_valid = x_train_full[train_size:], y_train_full[train_size:]

# Define the model architecture
model = keras.Sequential(
    [
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ]
)

# Compile the model with categorical crossentropy loss
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# Train the model on the training set with validation set
history = model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=1, validation_data=(x_valid, y_valid))

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", test_loss)
print("Test accuracy:", test_acc)

# Plot the training and validation loss
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot the training and validation accuracy
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Get 20 random images from the test set and predict their labels
n_images = 20
random_indices = np.random.choice(len(x_test), n_images)
x_sample = x_test[random_indices]
y_true = y_test[random_indices]
y_pred = model.predict(x_sample)

# Plot the images and color-code the predicted labels as green (correct) and red (incorrect)
fig, axs = plt.subplots(4, 5, figsize=(10, 8))
axs = axs.flatten()
for i in range(n_images):
    axs[i].imshow(x_sample[i], cmap='gray')
    true_label = np.argmax(y_true[i])
    pred_label = np.argmax(y_pred[i])
    if true_label == pred_label:
        axs[i].set_title(str(true_label), color='g')
    else:
        axs[i].set_title(str(pred_label), color='r')
    axs[i].axis('off')
plt.show()

# Generate a confusion matrix for the test set
y_pred = model.predict(x_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_test_labels = np.argmax(y_test, axis=1)
conf_mat = confusion_matrix(y_test_labels, y_pred_labels)
plt.figure
