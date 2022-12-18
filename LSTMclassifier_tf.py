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


def holdout_R(test_size, validation_size):
    """splits the dataset onto 3 parts by given size """
    X, y = load_data(D_PATH)
    # detach test section, sklearn performs shuffling by default
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    # detach validation section
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)
    # RNN doesn't require 3d arrays on input
    return X_train, X_validation, X_test, y_train, y_validation, y_test


def build_model(input_shape):
    """RNN LSTM model with 2x, Seq2Sea and Seq2Vec RNN layers"""
    model = tf.keras.Sequential()
    # sequence to sequence layer, returns sequence which is passed to the 2nd LSTM layer
    model.add(tf.keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True))
    # sequence to vector layer
    model.add(tf.keras.layers.LSTM(64))

    # dropout layer to avoid overfitting
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
    X_train, X_validation, X_test, y_train, y_validation, y_test = holdout_R(0.2, 0.1)
    input_shape = (X_train.shape[1], X_train.shape[2])
    # input shape is batch size + (130, 13, 1)
    model = build_model(input_shape)
    model.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs=30)

    loss, acc = model.evaluate(X_test, y_test, verbose=1)
    print(f"Accuracy on test set: {acc}")

    model.save('models/LSTMTF')
    checkpoint_model = tf.keras.models.load_model('models/LSTMTF')
    checkpoint_model.fit(X_validation, y_validation, validation_data=(X_test, y_test), batch_size=32, epochs=10)

    # X, y = X_test[33], y_test[33]
    # predict(model, X, y)
