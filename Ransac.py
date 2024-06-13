# Step 1 - Import packages
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import RANSACRegressor

# Step 2 - Importing Datasets
# Train Data
Train_Model = pd.read_csv('Supervised ML/train.csv')
Train = Train_Model.dropna()

# Variables of Training data set
X_Train = np.array(Train.iloc[:, :-1].values)  # Features (X)
Y_Train = np.array(Train.iloc[:, 1].values)    # Labels (Y)

# Test Data
Test_Model = pd.read_csv('Supervised ML/test.csv')
Test = Test_Model.dropna()

# Variables of Test data set
X_Test = np.array(Test.iloc[:, :-1].values)    # Features (X)
Y_Test = np.array(Test.iloc[:, 1].values)      # Labels (Y)

# Step 3 - Creating MODEL
Model = RANSACRegressor()
Model.fit(X_Train, Y_Train)
Prediction = Model.predict(X_Test)

# Step 4 - Checking Accuracy
Accuracy = Model.score(X_Test, Y_Test)

# Step 5 - Making Graph
plt.plot(X_Train, Model.predict(X_Train), color='green')  # Plotting the regression line
plt.show()

# Step 6 - Output Accuracy
print("Accuracy:", Accuracy)
