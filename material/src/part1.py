import numpy as np
import matplotlib.pyplot as plt
from keras import Input
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import pandas as pd

# Load datasets from Kaggle
# Make sure you have your Kaggle API key set up properly to use the Kaggle datasets
# Here, we assume you've already downloaded the datasets and placed them in your working directory
# cereal_data = pd.read_csv("cereal.csv")
# concrete_data = pd.read_csv("concrete.csv")
cereal_data = pd.read_csv("https://github.com/ML-Course-2026/session5/raw/refs/heads/main/material/datasets/cereal.csv")
concrete_data = pd.read_csv("https://github.com/ML-Course-2026/session5/raw/refs/heads/main/material/datasets/concrete.csv")
# Preprocess datasets
# For simplicity, we'll use only a subset of features for each dataset and treat it as a regression problem

# Cereal dataset preprocessing
cereal_features = cereal_data[['calories', 'protein', 'fat', 'sodium', 'fiber', 'carbo', 'sugars', 'potass', 'vitamins']]
cereal_target = cereal_data['rating']

# Concrete dataset preprocessing
concrete_features = concrete_data.drop(columns=['CompressiveStrength'])
concrete_target = concrete_data['CompressiveStrength']

# Split datasets into training and testing sets
cereal_X_train, cereal_X_test, cereal_y_train, cereal_y_test = train_test_split(cereal_features, cereal_target, test_size=0.2, random_state=42)
concrete_X_train, concrete_X_test, concrete_y_train, concrete_y_test = train_test_split(concrete_features, concrete_target, test_size=0.2, random_state=42)

# Define a Sequential model
model = Sequential()
model.add(Input(shape=(cereal_X_train.shape[1],)))
# Add dense layers
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))# Output layer for regression task

# Compile the model
model.compile(optimizer='adam',
              loss='mean_squared_error',  # Use MSE loss for regression
              metrics=['mae'])

# Print model summary
model.summary()

# Train the model on cereal dataset
history_cereal = model.fit(cereal_X_train, cereal_y_train, epochs=10, batch_size=64, validation_data=(cereal_X_test, cereal_y_test), verbose=1)

# Plot training history for cereal dataset
plt.plot(history_cereal.history['loss'], label='Cereal Train Loss')
plt.plot(history_cereal.history['val_loss'], label='Cereal Validation Loss')

plt.title('Training and Validation Loss for Cereal Dataset')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()