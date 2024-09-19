import os
os.chdir("/Users/jacksonwalters/Documents/GitHub/enefit-kaggle/predict-energy-behavior-of-prosumers/")

import IPython
import IPython.display
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from load_data import merged_df

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)

#load the dataset, slice a smaller amount for now
REDUCED_DATESET_SIZE = 100_000
df = merged_df()
df = df[:REDUCED_DATESET_SIZE]

#test/train split
train_dataset = df.sample(frac=0.8, random_state=0)
test_dataset = df.drop(train_dataset.index)

#drop the target column 
train_labels = train_dataset['target']
test_labels = test_dataset['target']

train_features = train_dataset.drop(columns=['target'])
test_features = test_dataset.drop(columns=['target'])

#get number of features
num_features = len(test_features.columns.values)

#create a normalization layer explicitly specifying input_shape to avoid keras load errors
normalizer = tf.keras.layers.Normalization(input_shape=[num_features,],axis=-1)

#normalize the train_features by subtracting the mean and dividing by the std deviation
normalizer.adapt(np.array(train_features))

#build the three layer model with initial normalization layer
def build_and_compile_model(norm):
  model = keras.Sequential([
      norm,
      layers.Dense(64, activation='relu'),
      layers.Dense(64, activation='relu'),
      layers.Dense(1)
  ])

  model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001))
  return model

dnn_model = build_and_compile_model(normalizer)
dnn_model.summary()

#fit the model
history = dnn_model.fit(
    train_features,
    train_labels,
    validation_split=0.2,
    verbose=1, epochs=2)

# save the model to disk
import pickle
dnn_model_filename = '../models/dnn_model.sav'
pickle.dump(dnn_model, open(dnn_model_filename, 'wb'))

#save model using keras
dnn_model.save('../models/dnn_model.keras')