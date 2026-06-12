import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
import pickle


print("Loading data...")
df = pd.read_csv("gesture_data.csv")

X = df.drop("label", axis=1).values
y = df["label"].values

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_categorical = keras.utils.to_categorical(y_encoded)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)


X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")
print(f"Gestures: {list(encoder.classes_)}")
print(f"Input size: {X.shape[1]} features")

# Input is now 132 (63+63 hands + 6 body)
input_size = X.shape[1]

model = keras.Sequential([
    keras.layers.Dense(256, activation='relu', input_shape=(input_size,)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(len(encoder.classes_), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()


print("\nTraining...")
history = model.fit(
    X_train, y_train,
    epochs=60,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {accuracy*100:.2f}%")

model.save("gesture_model.keras")
print("Model saved as gesture_model.keras")