import numpy as np
import pandas as pd

from numpy import asarray
from sklearn.model_selection import train_test_split
from sklearn import svm
from category_encoders.target_encoder import TargetEncoder
from sklearn.preprocessing import LabelEncoder
from category_encoders.ordinal import OrdinalEncoder
from category_encoders.woe import WOEEncoder
from category_encoders.target_encoder import TargetEncoder
from category_encoders.sum_coding import SumEncoder
from category_encoders.m_estimate import MEstimateEncoder
from category_encoders.leave_one_out import LeaveOneOutEncoder
from category_encoders.helmert import HelmertEncoder
from category_encoders.cat_boost import CatBoostEncoder
from category_encoders.james_stein import JamesSteinEncoder
from category_encoders.one_hot import OneHotEncoder
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier


data = pd.read_csv('preprocessed_data.csv')


encodedLabel = []
# 0 1 2 ...
for i in data['MovieSuccessLevel']:
    if(i == 'S'):
        encodedLabel.append(0)
    elif(i == 'A'):
        encodedLabel.append(1)
    elif(i == 'B'):
        encodedLabel.append(2)
    elif(i == 'C'):
        encodedLabel.append(3)
    elif(i == 'D'):
        encodedLabel.append(4)

encodedLabel = pd.DataFrame(encodedLabel)

features = ['release_date', 'genre', 'MPAA_rating', 'director', 'character', 'voice-actor']
data = data[features]

data.release_date = pd.to_datetime(data.release_date)


#------------------------------------------------------------------------------------------
#Encoder Phase
JSE_encoder = JamesSteinEncoder()
encodedData = JSE_encoder.fit_transform(data[features], encodedLabel)
#------------------------------------------------------------------------------------------


encodedData['release_date'] = encodedData['release_date'].dt.year

x = encodedData[features]
y = np.squeeze(pd.DataFrame(encodedLabel))

#10 , 2
X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=10)



#------------------------------------------------------------------------------------------
#Model Phase
svm_kernel_ovo = OneVsOneClassifier(SVC(kernel='linear', C=1)).fit(X_train, y_train)

accuracy1 = svm_kernel_ovo.score(X_train, y_train)
accuracy = svm_kernel_ovo.score(X_test, y_test)
print('Linear Kernel OneVsOne SVM Train accuracy: ' + str(accuracy1*100) + "%")
print('Linear Kernel OneVsOne SVM Test accuracy: ' + str(accuracy*100) + "%")
#------------------------------------------------------------------------------------------