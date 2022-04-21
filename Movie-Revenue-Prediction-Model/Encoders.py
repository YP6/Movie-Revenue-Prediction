from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import pandas as pd
from category_encoders.target_encoder import TargetEncoder
import Preprocessing as pp

data = pd.read_csv("data_finalizing_test.csv")
del(data['movie_title'])
data['release_date'] = pd.to_datetime(data['release_date'])


##encoding

##encoder=ce.HashingEncoder(cols='voice-actor',n_components=6)
##encoder.fit_transform(data)
## target Encoder

encoder = TargetEncoder()
x = encoder.fit_transform(data[pp.features], data['revenue'])
y = data['revenue']


#Calculating Correlation
corr = data.corr()
top_feature = corr.index[abs(corr['revenue'])>0.3]

Features = top_feature.to_list()
Features.remove('revenue')

x = x[Features]
## Polynomial Model
##data.release_date = pd.to_datetime(data.release_date)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, shuffle=False)

polyfeature = PolynomialFeatures(degree=3)

xtrainPoly = polyfeature.fit_transform(xtrain)
polyModel = linear_model.LinearRegression()

polyModel.fit(xtrainPoly, ytrain)

trainPrediction = polyModel.predict(xtrainPoly)
testPrediction = polyModel.predict(polyfeature.fit_transform(xtest))
trainPrediction = polyModel.predict(polyfeature.fit_transform(xtrain));
print("Mean Sqaure Error : " , metrics.mean_squared_error(ytest , testPrediction))
print("Mean Square Error : " , metrics.mean_squared_error(ytrain , trainPrediction));
true_player_value=np.asarray(ytest)[0]
predicted_player_value=trainPrediction[0]
print('True Value : ' + str(true_player_value))
print('Predicted  : ' + str(predicted_player_value))





