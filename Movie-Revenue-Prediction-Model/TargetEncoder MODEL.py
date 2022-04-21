import pandas as pd
from category_encoders.target_encoder import TargetEncoder
from sklearn.preprocessing import PolynomialFeatures
from Preprocessing import features
import numpy as np
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


data = pd.read_csv('data_finalizing_test.csv')

data.release_date = pd.to_datetime(data.release_date).dt.year

te_encoder = TargetEncoder()
te_data = te_encoder.fit_transform(data[features], data['revenue'])

Y = data['revenue']

data[features] = te_data
corr = data.corr()
top_feature = corr.index[abs(corr['revenue']) > 0.45]
Features = top_feature.to_list()
Features.remove('revenue')
X = data[Features]

print("", '-'*50)
print(' Features :', Features)
print("", '-'*50)

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, train_size=0.80, shuffle=True, random_state=100)


poly_features = PolynomialFeatures(degree=2)
X_Train_poly = poly_features.fit_transform(X_Train)

poly_model = linear_model.LinearRegression()
poly_model.fit(X_Train_poly, Y_Train)

poly_Train_predections = poly_model.predict(X_Train_poly)

X_Test_poly = poly_features.fit_transform(X_Test)
poly_Test_predictions = poly_model.predict(X_Test_poly)

print('_'*25)
print('Polynomial Regression With Target Encoder')
print('_'*15)
print('Train MSE =', metrics.mean_squared_error(np.array(Y_Train), poly_Train_predections))
print('Accuracy =', "{:.4f}".format(r2_score(Y_Train,poly_Train_predections)*100), "%")
print('_'*15)
print('Test MSE = ', metrics.mean_squared_error(np.array(Y_Test), poly_Test_predictions))
print('Accuracy =', "{:.4f}".format(r2_score(Y_Test,poly_Test_predictions)*100), "%")
print('_'*25)
