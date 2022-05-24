import pandas as pd
import numpy as np
import pickle
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from category_encoders.m_estimate import MEstimateEncoder
from sklearn.preprocessing import PolynomialFeatures

# Testing Steps:
#         1- Run Preprocessing script on the 3 given datasets.
#         2- Load Trained Models and test them using Test script.

# Load the model.
features = ['release_date', 'genre', 'director', 'character', 'voice-actor']
with open('MEstimate Encoding Model', 'rb') as file:
    model = pickle.load(file)

# Reading the test data set.
data = pd.read_csv('Preproccessed_Dataset/preproccessed_data_test.csv')
data.release_date = pd.to_datetime(data.release_date).dt.year


# Encoding data.

# M-Estimate Encoding.
MEstimateEncoder = MEstimateEncoder()
encoded_data = MEstimateEncoder.fit_transform(data[features], data['revenue'])

data[features] = encoded_data


X = encoded_data[features]
Y = data['revenue']


poly_features = PolynomialFeatures(degree=3)
x_test_poly = poly_features.fit_transform(X)

test_predictions = model.predict(x_test_poly)


print('_'*25)
print('Polynomial Regression With MEstimate Encoder')
print('_'*15)
print('Test MSE = ', metrics.mean_squared_error(np.array(Y), test_predictions))
print('Accuracy =', "{:.4f}".format(r2_score(Y,test_predictions)*100), "%")
print('_'*25)

