import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from category_encoders.m_estimate import MEstimateEncoder

successLevel = pd.read_csv('../Datasets/Classification Datasets/movies-revenue-classification.csv')
voice_actors = pd.read_csv('../Datasets/Regression Datasets/movie-voice-actors.csv')
directors = pd.read_csv('../Datasets/Regression Datasets/movie-director.csv')
features = ['release_date', 'genre', 'MPAA_rating', 'director', 'character', 'voice-actor']
"""Converts release-date data type to datetime instead of string."""
print("\nParsing Date: ")
print("-" * 25)

# Checking date format consistency.
date_lengths = successLevel.release_date.str.len()

print("Date Lengths :")
print(date_lengths.value_counts())
print("-" * 25)

print("Release-Date datatype before Parsing: ", successLevel.release_date.dtype)

# Fixing Parsing Wrong Dates
for i in range(successLevel.shape[0]):
    date = successLevel.loc[i, 'release_date']
    if 2 < int(date[-2]) < 7:
        new_date = date[:-2] + "19" + date[-2:]
        successLevel.loc[i, 'release_date'] = new_date
    elif int(date[-2]) == 2 and int(date[-1]) > 2:
        new_date = date[:-2] + "19" + date[-2:]
        successLevel.loc[i, 'release_date'] = new_date
successLevel.release_date = pd.to_datetime(successLevel.release_date)
print("Release-Date datatype after Parsing: ", successLevel.release_date.dtype)
print("-" * 50)
directors.rename(columns={'name': 'movie_title'}, inplace=True)
voice_actors.rename(columns={'movie': 'movie_title'}, inplace=True)
MovieSuccessLevels_actors = pd.merge(successLevel, voice_actors, on="movie_title", how="outer")
data = pd.merge(MovieSuccessLevels_actors, directors, on="movie_title", how="outer")
data = data.dropna(axis=0, subset=['MovieSuccessLevel'])
print("Shape after Joining :", data.shape)
# labelEncoder = LabelEncoder()
# encodedLabel = labelEncoder.fit_transform(data["MovieSuccessLevel"])
# encodedLabel = pd.DataFrame(encodedLabel)
encodedLabel = []
# 0 1 2 ...
for i in data['MovieSuccessLevel']:
    if (i == 'S'):
        encodedLabel.append(0)
    elif (i == 'A'):
        encodedLabel.append(1)
    elif (i == 'B'):
        encodedLabel.append(2)
    elif (i == 'C'):
        encodedLabel.append(3)
    elif (i == 'D'):
        encodedLabel.append(4)
encodedLabel = pd.DataFrame(encodedLabel)
encodedLabel.head(10)
y = encodedLabel
y = np.squeeze(y)
y.shape

MEE_encoder = MEstimateEncoder()
encodedData = MEE_encoder.fit_transform(data[features], encodedLabel)
encodedData['release_date'] = encodedData['release_date'].dt.year

x = encodedData[features]
x.head()
X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=10)




def_svc = svm.SVC(kernel='linear',degree=4, C=0.4, tol=1.5, gamma=0.8).fit(X_train, y_train)

trainPredictions = def_svc.predict(X_train)
trainAccuracy = np.mean(trainPredictions == y_train)
print("\nPolynomial SVC with degree 4 Train Accuracy:", "{:.2f}".format(trainAccuracy * 100), "\b%")

testPredictions = def_svc.predict(X_test)
testAccuracy = np.mean(testPredictions == y_test)

print("\nPolynomial SVC with degree 4 Accuracy:", "{:.2f}".format(testAccuracy * 100), "\b%\n")




print("Difference Between them:", "{:.2f}".format(abs((trainAccuracy * 100) - (testAccuracy * 100))), "\b%\n")
