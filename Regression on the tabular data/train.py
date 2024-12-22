import numpy as np # linear algebra
import pandas as pd # data processing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import root_mean_squared_error
import pickle

# Set random seed for reproducibility
np.random.seed(42)

train_raw = pd.read_csv("train.csv")

# Drop the 8th column
train = train_raw.drop(labels=["8"], axis=1)

# Normalize the data 
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train.iloc[:, :-1])


# Train-Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(train_scaled, train['target']/100,
                                                    test_size=0.2, random_state=42)



# Train MLP Regressor (Multi-layer Perceptron)
mlp_model = MLPRegressor(hidden_layer_sizes=(500,), max_iter=10000, activation='relu', solver='adam', random_state=42)
mlp_model.fit(X_train, y_train)

# Make predictions
y_pred = mlp_model.predict(X_test)

# Since the output range is expected between 0 and 1, we can clip predictions to ensure this
y_pred = np.clip(y_pred, 0, 1)

# Evaluate performance
rmse = root_mean_squared_error(y_test, y_pred)
print(f"MLP Regressor Root Mean Squared Error on Test: {rmse:.4f}")

# Test on train data
y_pred = mlp_model.predict(X_train)
rmse_train = root_mean_squared_error(y_train, y_pred)
print(f"MLP Regressor Root Mean Squared Error on Train: {rmse_train:.4f}")


# save
with open('mlp_model.pkl','wb') as f:
    pickle.dump(mlp_model,f)

