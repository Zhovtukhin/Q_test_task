import numpy as np # linear algebra
import pandas as pd # data processing
from sklearn.preprocessing import MinMaxScaler
import pickle

# Set random seed for reproducibility
np.random.seed(42)

# load train data for MinMaxScaler
train_raw = pd.read_csv("train.csv")

# Drop the 8th column
train = train_raw.drop(labels=["8"], axis=1)

# Normalize the data
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train.iloc[:, :-1])

# load test data
test_raw = pd.read_csv("hidden_test.csv")

# Transforl like train data
test = test_raw.drop(labels=["8"], axis=1)
test_scaled = scaler.transform(test)

# load model
with open('mlp_model.pkl', 'rb') as f:
    mlp_model = pickle.load(f)

# predict
y_pred = mlp_model.predict(test_scaled)*100

# save csv
df = pd.DataFrame(y_pred, columns=['target'])
df.to_csv('hidden_test_target.csv', index=False)
