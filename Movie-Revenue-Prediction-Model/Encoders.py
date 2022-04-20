import pandas as pd
import category_encoders as ce
import Preprocessing

encoder= ce.TargetEncoder()
data = pd.read_csv("data_finalizing_test.csv")
del(data['movie_title'])
