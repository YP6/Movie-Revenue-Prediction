from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import pandas as pd
import category_encoders as ce


data = pd.read_csv("data_finalizing_test.csv")
del(data['movie_title'])
data['release_date'] = pd.to_datetime(data['release_date'])
del(data['release_date'])

print(data);

##encoding

##encoder=ce.HashingEncoder(cols='voice-actor',n_components=6)
##encoder.fit_transform(data)
## target Encoder

encoder=ce.TargetEncoder(cols='voice-actor')
data['voice-actor'] = encoder.fit_transform(data['voice-actor'],data['revenue'])
encoder=ce.TargetEncoder(cols='MPAA_rating')
data['MPAA_rating'] = encoder.fit_transform(data['MPAA_rating'],data['revenue'])
encoder=ce.TargetEncoder(cols='director')
data['director'] = encoder.fit_transform(data['director'],data['revenue'])
encoder=ce.TargetEncoder(cols='character')
data['character'] = encoder.fit_transform(data['character'],data['revenue'])
encoder=ce.TargetEncoder(cols='genre')
data['genre'] = encoder.fit_transform(data['genre'],data['revenue'])

## Polynomial Model
##data.release_date = pd.to_datetime(data.release_date)

print(data)
x = data.loc[:, data.columns != 'revenue']
##x.release_date = pd.to_datetime(x.release_date)
y = data['revenue']
xtrain , xtest , ytrain , ytest = train_test_split(x,y,test_size=0.2,shuffle=False)

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





