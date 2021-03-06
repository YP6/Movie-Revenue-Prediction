import pandas as pd
from category_encoders.james_stein import JamesSteinEncoder
from Preprocessing import features
import numpy as np
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import timeit
import pickle


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

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, train_size=0.80, shuffle=True, random_state=100)

start = timeit.default_timer()

model = linear_model.LinearRegression()
model.fit(X_Train, Y_Train)

stop = timeit.default_timer()
print('Training Time :', "{:.2f}".format((stop-start)*1000), "ms")

Train_predictions = model.predict(X_Train)
Test_predictions = model.predict(X_Test)

with open('JamesStein Encoding Model', 'wb') as file:
    pickle.dump(model, file)

print('_'*25)
print('Multi Linear Regression With James Stein Encoder')
print('_'*15)
print('Train MSE =', metrics.mean_squared_error(np.array(Y_Train), Train_predictions))
print('Accuracy =', "{:.4f}".format(r2_score(Y_Train, Train_predictions) * 100), "%")
print('_'*15)
print('Test MSE =', metrics.mean_squared_error(np.array(Y_Test), Test_predictions))
print('Accuracy =', "{:.4f}".format(r2_score(Y_Test, Test_predictions)*100), "%")
print('_'*25)
