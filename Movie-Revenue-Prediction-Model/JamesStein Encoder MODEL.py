import pandas as pd
from category_encoders.james_stein import JamesSteinEncoder
from Preprocessing import features
import numpy as np
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


data = pd.read_csv('data_finalizing_test.csv')

data.release_date = pd.to_datetime(data.release_date).dt.year

JSE_encoder = JamesSteinEncoder()
jse_data = JSE_encoder.fit_transform(data[features], data['revenue'])

Y = data['revenue']

data[features] = jse_data
corr = data.corr()
top_feature = corr.index[abs(corr['revenue'])>0.4]
Features = top_feature.to_list()
Features.remove('revenue')
X = data[Features]

print("", '-'*50)
print(' Features :', Features)
print("", '-'*50)

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, train_size=0.80)


model = linear_model.LinearRegression()
model.fit(X_Train, Y_Train)

Train_predictions = model.predict(X_Train)
Test_predictions = model.predict(X_Test)

print('_'*25)
print('Multi Linear Regression With James Stein Encoder')
print('Train MSE =', metrics.mean_squared_error(np.array(Y_Train), Train_predictions))
print('Accuracy =', "{:.4f}".format(r2_score(Y_Train, Train_predictions) * 100), "%")
print('Test MSE =', metrics.mean_squared_error(np.array(Y_Test), Test_predictions))
print('Accuracy =', "{:.4f}".format(r2_score(Y_Test, Test_predictions)*100), "%")
print('_'*25)
