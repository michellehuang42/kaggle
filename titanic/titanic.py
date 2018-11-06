import pandas as pd
from sklearn.tree import DecisionTreeRegressor
# from sklearn.metrics import mean_absolute_error
import ipdb

train_path = './data/train.csv'
train_data = pd.read_csv(train_path)
test_path = './data/test.csv'
test_data = pd.read_csv(test_path)

ipdb.set_trace()
# print(training_data.columns)

train_y = train_data.Survived

features = ['Sex', 'Age', 'Fare']

train_X = train_data[features]


# print(train_X.describe())

titanic_model = DecisionTreeRegressor(random_state=1)
titanic_model.fit(train_X, train_y)

# test_y = 