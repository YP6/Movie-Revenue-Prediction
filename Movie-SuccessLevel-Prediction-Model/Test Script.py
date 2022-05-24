import pandas as pd
import numpy as np
import pickle
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from category_encoders.james_stein import JamesSteinEncoder
from sklearn.preprocessing import PolynomialFeatures

# Testing Steps:
#         1- Run Preprocessing script on the 3 given datasets.
#         2- Load Trained Models and test them using Test script.

# Load the model.
features = ['release_date', 'genre', 'MPAA_rating', 'director', 'character', 'voice-actor']
with open('1vs1 Classifier', 'rb') as file:
    model = pickle.load(file)

# Reading the test data set.
data = pd.read_csv('test_preprocessed_data.csv')
data.release_date = pd.to_datetime(data.release_date).dt.year


# Encoding data.
encodedLabel = []
# 0 1 2 ...
for i in data['MovieSuccessLevel']:
    if i == 'S':
        encodedLabel.append(0)
    elif i == 'A':
        encodedLabel.append(1)
    elif i == 'B':
        encodedLabel.append(2)
    elif i == 'C':
        encodedLabel.append(3)
    elif i == 'D':
        encodedLabel.append(4)

encodedLabel = pd.DataFrame(encodedLabel)

# JamesStein Encoding.
JSE_encoder = JamesSteinEncoder()
encodedData = JSE_encoder.fit_transform(data[features], encodedLabel)

x = encodedData[features]
y = np.squeeze(pd.DataFrame(encodedLabel))

accuracy = model.score(x, y)

print('Accuracy: ' + str(accuracy*100) + "%")
