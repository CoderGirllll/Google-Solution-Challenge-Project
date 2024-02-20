#Importing data
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("Churn.csv")   #Dataset file name

X = pd.get_dummies(df.drop(["Churn", "Customer ID"], axis=1))   #Column names
y = df["Churn"].apply(lambda x: 1 if x=="Yes" else 0)   #Column name

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

y_train.head()


#Importing dependencies
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score


#Building and Compiling model
model = Sequential()
model.add(Dense(units=32, activation="relu", input_dim=len(X_train.columns)))
model.add(Dense(units=64, activation="relu"))
model.add(Dense(units=1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="sgd", metrics="accuracy")


#Fiting, predicting and evaluating
model.fit(X_train, y_train, epochs=200, batch_size=32)

y_hat = model.predict(X_test)
y_hat = [0 if val < 0.5 else 1 for val in y_hat]

accuracy_score(y_test, y_hat)


#Saving and Reloading
model.save("tfmodel")   #model name

#del model
#model = load_model("tfmodel")