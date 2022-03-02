# Build a classifier.
# TODO: word embeddings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.feature_selection import SelectKBest, f_classif
from loadData import ht, train, validate, vectorize
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.layers import Dense, Dropout


# We're setting the seed first in loadData.py.
# random.seed(12345)


# The number of "best" features to use, as measured by f_classif.
NUM_TOP_FEATURES = 1000

# The number of layers in our neural net (including the last layer/activation).
NUM_LAYERS = 2
NUM_UNITS = 64
DROPOUT_RATE = .2
LEARNING_RATE = 1e-3
NUM_EPOCHS = 500
BATCH_SIZE = 128


train_vecs, validate_vecs, feature_names, _ = \
    vectorize(train.text, validate.text)

selector = SelectKBest(f_classif, k=NUM_TOP_FEATURES)
selector.fit(train_vecs, train.polarized)
x_train = selector.transform(train_vecs).astype('float32')
x_validate = selector.transform(validate_vecs).astype('float32')
fns = selector.get_feature_names_out(feature_names)


model = models.Sequential()
model.add(Dropout(rate=DROPOUT_RATE, input_shape=(NUM_TOP_FEATURES,)))
for _ in range(NUM_LAYERS-1):
    model.add(Dense(units=NUM_UNITS, activation='relu'))
    model.add(Dropout(rate=DROPOUT_RATE))
model.add(Dense(units=1, activation="sigmoid"))

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer, loss="binary_crossentropy",
    metrics=['acc'])

callbacks = []

history = model.fit(
    x_train,
    train.polarized,
    epochs=NUM_EPOCHS,
    callbacks=callbacks,
    validation_data=(x_validate, validate.polarized),
    verbose=2,  # Logs once per epoch.
    batch_size=BATCH_SIZE)

history = history.history
print('Validation accuracy: {acc}, loss: {loss}'.format(
    acc=history['val_acc'][-1], loss=history['val_loss'][-1]))
model.save('PolarOps.h5')


