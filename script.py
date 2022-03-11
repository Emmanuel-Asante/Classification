# Import modules
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical
import numpy as np

# Load data into a DataFrame
data = pd.read_csv("heart_failure.csv")

# Print a concise summary of the DataFrame
print(data.info())

# Print out the distribution of death_event column
print(Counter(data["death_event"]))

# Extract label column from data
y = data["death_event"] 

# Extract features columns from data
x = data[['age','anaemia','creatinine_phosphokinase','diabetes','ejection_fraction','high_blood_pressure','platelets','serum_creatinine','serum_sodium','sex','smoking','time']]

# Convert categorical features in data to one-hot encoding
x = pd.get_dummies(x)

# Split data into training features, test features, training labels, and test labels
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# Scale numeric features
ct = ColumnTransformer([(
  "numeric",
  StandardScaler(),
  ['age','creatinine_phosphokinase','ejection_fraction','platelets','serum_creatinine','serum_sodium','time']
)])

# Train the scaler instance on the training data
X_train = ct.fit_transform(X_train)

# Scale the test data
X_test = ct.transform(X_test)

# Initialize a LabelEncoder object
le = LabelEncoder()

# Fit the le to the training labels Y_train
Y_train = le.fit_transform(Y_train.astype(str))

# Encode the test labels Y_test
Y_test = le.transform(Y_test.astype(str))

# Transform Y_train into a binary vector
Y_train = to_categorical(Y_train)

# Transform Y_test into a binary vector
Y_test = to_categorical(Y_test)

# Create a Sequential model
model = Sequential()

# Set and add the input layer to the model
model.add(InputLayer(input_shape=(X_train.shape[1],)))

# Set and add hidden layer to the model
model.add(Dense(12, activation="relu"))

# Set and add output layer to the model
model.add(Dense(2, activation="softmax"))

# Set the optimizer
model.compile(
  loss="categorical_crossentropy",
  optimizer="adam",
  metrics=["accuracy"]
)

# Train the model
model.fit(X_train, Y_train, epochs=100, batch_size=16, verbose=1)

# Evaluate the trained model
loss, acc = model.evaluate(X_test, Y_test, verbose=0)

# Print out the final loss and the accuracy metrics
print("\nFinal loss: {}\nAccuracy metrics: {}".format(loss, acc))

# Make prediction for the test data
y_estimate = model.predict(X_test)

# Select indicies of true classes for each label encoding in y_estimate
y_estimate = np.argmax(model.predict(X_test), axis=-1)

# Select indicies of true classes for each label encoding in Y_test
y_true = np.argmax(Y_test, axis=1)

# Print F1-score metrics
print(classification_report(y_true, y_estimate))