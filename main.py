# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_absolute_error

# Load the dataset
file_path = "your_dataset.csv"  # Replace with your actual file path
df = pd.read_csv(file_path)

# Preprocessing the data
X = pd.get_dummies(df.drop(["amount"], axis=1))  # Assuming "amount" is the column to predict
y = df["amount"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and compile the model
model = Sequential()
model.add(Dense(units=32, activation="relu", input_dim=len(X_train.columns)))
model.add(Dense(units=64, activation="relu"))
model.add(Dense(units=1, activation="linear"))  # Linear activation for regression problem

model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mae"])

# Fit the model
model.fit(X_train, y_train, epochs=200, batch_size=32, verbose=1)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error on Test Set: {mae}")

# Now, you can use the trained model to predict the amount of medicines for the coming month
# For example, if you have new data for the coming month, you can use model.predict(new_data) to get predictions.
# Make sure to preprocess the new data in the same way as you did for the training data.

# Save the trained model for future use
model.save("medicine_amount_prediction_model.h5")
print("Model saved successfully.")
