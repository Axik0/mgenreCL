import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

D_PATH = 'extr_data.json'

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")

tf.config.list_physical_devices('GPU')

genres = []


def load_data(dataset_path):
    """loads json and extracts two lists: MFCCs and labels(genre indices)"""
    with open(dataset_path, 'r') as dp:
        data = json.load(dp)
    # see, it's quite convenient to have two separate lists naturally linked by index
    inputs = np.array(data['mfcc'])
    targets = np.array(data['labels'])
    global genres
    genres = data['genres']
    return inputs, targets


def holdout(test_size, validation_size):
    """splits the dataset onto 3 parts by given size """
    X, y = load_data(D_PATH)
    # detach test section, sklearn performs shuffling by default
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    # detach validation section
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)
    # add new dimensions to the end of existing arrays of all inputs as CNN expects 3d array, i.e. (1000..., 13, 1)
    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    return X_train, X_validation, X_test, y_train, y_validation, y_test


def build_model(input_shape):
    """CNN with 3x convolution + maxpooling layers, then 1 dense layer"""
    model = tf.keras.Sequential()
    # feature generation layers
    # Batch normalization for faster convergence
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv2D(32, (2, 2), activation='relu'))
    model.add(tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())

    # flatten 2-d array to 1d array and apply dense layer
    # dropout layer to avoid overfitting
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))

    # output layer, softmax for more than 2 classes to get probas
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    return model


def predict(model, X, y):
    """visualise and compare model predictions on random data sample"""
    # input shape is batch size + (130, 13, 1), so that we should add a dimension from the beginning
    X = X[np.newaxis, ...]
    predictions = model.predict(X)
    # 2d array of 10 probas, need an index of max value
    predicted_index = np.argmax(predictions, axis=1)
    print(f"Expected {genres[predicted_index[0]]}, got {genres[y]}")


if __name__ == "__main__":
    X_train, X_validation, X_test, y_train, y_validation, y_test = holdout(0.2, 0.1)
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    # input shape is batch size + (130, 13, 1)
    model = build_model(input_shape)
    model.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs=30)

    loss, acc = model.evaluate(X_test, y_test, verbose=1)
    print(f"Accuracy on test set: {acc}")

    # model.save('models/CCNNTF')
    # checkpoint_model = tf.keras.models.load_model('models/CCNNTF')
    # checkpoint_model.fit(X_validation, y_validation, validation_data=(X_test, y_test), batch_size=32, epochs=30)

    # X, y = X_test[33], y_test[33]
    # predict(model, X, y)
